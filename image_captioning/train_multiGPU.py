from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import sys
import os

from utils.config import Config
from models import utils, caption
from datasets.coco_caption import prepare_dataset
from torch.utils.data import DataLoader, DistributedSampler
from engine import train_one_epoch, evaluate
import torch.distributed as dist   
import torch.multiprocessing as mp
from utils.log_utils import create_logger, get_model_summary
from tensorboardX import SummaryWriter
import pprint

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def check_dir(opts):

    if not os.path.isdir(opts.dir):
        raise ValueError('Data Dir shouldnot be empty')

    os.makedirs(opts.logDir, exist_ok=True)
    os.makedirs(opts.modelDir, exist_ok=True)

def train_caption(rank, world_size):

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(rank)
    opts = Config()

    check_dir(opts)
    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # device = torch.device(opts.device)
    seed = opts.seed + utils.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # CUDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    model, criterion = caption.build_model(opts)
    model.to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module

    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(pprint.pformat("Number of params: {}".format(n_parameters)))

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": opts.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=opts.lr, weight_decay=opts.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opts.lr_drop)

    dataset_train = prepare_dataset(opts, mode='train')
    dataset_val = prepare_dataset(opts, mode='val')
    logger.info(pprint.pformat("Train: {}, Valid: {}".format(len(dataset_train), len(dataset_val))))

    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)

    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, opts.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)
    data_loader_val = DataLoader(dataset_val, opts.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=opts.num_workers)

    if os.path.exists(opts.checkpoint):
        logger.info(pprint.pformat("Loading Checkpoint..."))
        checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        opts.start_epoch = checkpoint['epoch'] + 1


    logger.info(pprint.pformat("Start Training.."))
    for epoch in range(opts.start_epoch, opts.epochs):
        
        logger.info(pprint.pformat("Epoch: {}".format(epoch)))
        
        sampler_train.set_epoch(epoch)
        epoch_loss = train_one_epoch(
            model_without_ddp, criterion, data_loader_train, optimizer, rank, epoch, opts.clip_max_norm)
        lr_scheduler.step()

        logger.info(pprint.pformat("Training Loss: {}".format(epoch_loss)))
        global_steps = writer_dict['train_global_steps']
        writer_dict['writer'].add_scalar('train_loss', epoch_loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        validation_loss = evaluate(model_without_ddp, criterion, data_loader_val, rank)
        
        logger.info(pprint.pformat("Validation Loss: {}".format(validation_loss)))
        global_steps = writer_dict['valid_global_steps']
        writer_dict['writer'].add_scalar('val_loss', validation_loss, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, opts.checkpoint)

        print()

def main():
    world_size = 2
    mp.spawn(train_caption,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__ == '__main__':

    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()


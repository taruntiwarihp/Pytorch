from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Local files
from model.model import createDeepLabv3
from dataset.dataset import SegmentationDataset, get_transform
from utils import create_logger, get_model_summary

# Libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist   
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

import argparse, os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Train Classification')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='weights')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='logs')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='human_segmentation')
    parser.add_argument('--model_type',
                        help='Types of model backbone',
                        type=str,
                        default='resnet101',
                        choices=['mobilenet_v3_large', 'resnet101', 'resnet50'])
    parser.add_argument('--split_ratio',
                        help='train set valid set ratio',
                        type=float,
                        default=0.8)
    parser.add_argument('--batch_size',
                        help='Number of batch for data loader',
                        type=int,
                        default=4)
    parser.add_argument('--num_workers',
                        help='number of processor',
                        type=int,
                        default=8)
    parser.add_argument('--keep_feature_extract',
                        help='update the feature parameters',
                        type=int,
                        default=True)
    parser.add_argument('--n_class',
                        help='Total Class',
                        type=int,
                        default=2)
    parser.add_argument('--epochs',
                        help='Total Epochs',
                        type=int,
                        default=100)
    parser.add_argument('--max_lr',
                        help='maximum Learning Rate',
                        type=float,
                        default=3e-5)
    parser.add_argument('--weight_decay',
                        help='Weight Decay',
                        type=float,
                        default=1e-4)

    args = parser.parse_args()

    return args


def check_dir(opts):

    if not os.path.isdir(opts.dataDir):
        raise ValueError('Data Dir shouldnot be empty')

    os.makedirs(opts.logDir, exist_ok=True)
    os.makedirs(opts.modelDir, exist_ok=True)


def train_segmentation(rank, world_size):

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(rank)

    opts = parse_args()
    check_dir(opts)

    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    model = createDeepLabv3(opts)
    model.to(rank)
    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module

    n_parameters = sum(p.numel()
                       for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info(pprint.pformat("Number of params: {}".format(n_parameters)))
    print(n_parameters)


    dataset_train = SegmentationDataset(opts.dataDir, get_transform(train=True))
    dataset_val = SegmentationDataset(opts.dataDir, get_transform(train=False))

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    split_idx = int(0.2 * len(dataset_train))

    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-split_idx])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-split_idx:])
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)

    logger.info(pprint.pformat("Train: {}, Valid: {}".format(len(sampler_train), len(sampler_val))))

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, opts.batch_size, drop_last=True
    )

    train_loader = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)
    val_loader = DataLoader(dataset_val, opts.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=opts.num_workers)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), opts.max_lr, weight_decay=opts.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, opts.max_lr, epochs=opts.epochs, steps_per_epoch=len(train_loader))

    val_acc = 1e10
    for epoch in range(opts.epochs):

        model_without_ddp.train()
        losses = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description("Epoch {}".format(epoch))

                optimizer.zero_grad()
                pred = model_without_ddp(data.to(rank))

                loss = criterion(pred, target.to(rank))
                loss.backward()
                optimizer.step()
                sched.step()

                losses += loss.item()
                tepoch.set_postfix(loss=loss.item())

        train_loss = losses/len(train_loader)

        msg = 'Train Epoch : {}\t Training Loss : {}'.format(epoch, train_loss)
        logger.info(msg)

        global_steps = writer_dict['train_global_steps']
        writer_dict['writer'].add_scalar('train_loss', train_loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        model_without_ddp.eval()
        losses = 0
        pred_list = []
        tar_list = []

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for data, target, _ in tepoch:
                    tepoch.set_description("Epoch {}".format(epoch))
                    pred = model_without_ddp(data.to(rank))

                    _, predicted = torch.max(pred.data, 1)

                    tar_list.append(target.numpy())
                    pred_list.append(predicted.cpu().numpy())

                    loss = criterion(pred, target.to(rank))

                    losses += loss.item()

                    tepoch.set_postfix(loss=loss.item())



        mats = {
            'loss': losses/len(val_loader),
            'accuracy' : accuracy_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten()),
            'precision' : precision_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten(), average='macro'),
            'f1' : f1_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten(), average='macro'),
        }

        val_loss = losses/len(val_loader)

        # Logs
        msg = 'Validation Epoch : {}\t Validation Loss : {}\t Accuracy : {}\t Precision : {}\t F1-Score : {}\t'.format(
                        epoch, val_loss, mats['accuracy'], mats['precision'], mats['f1'])
        logger.info(msg)

        global_steps = writer_dict['valid_global_steps']
        writer_dict['writer'].add_scalar('val_loss', val_loss, global_steps)
        writer_dict['writer'].add_scalar('accuracy', mats['accuracy'], global_steps)
        writer_dict['writer'].add_scalar('precision', mats['precision'], global_steps)
        writer_dict['writer'].add_scalar('f1', mats['f1'], global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': sched.state_dict(),
            'epoch': epoch,
            'val_loss' : val_loss
        }
        
        torch.save(ckpt, '{}/{}_best_model.pt'.format(opts.modelDir, opts.model_type))
 

def main():
    world_size = 1
    mp.spawn(train_segmentation,
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


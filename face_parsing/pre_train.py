from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Local files
from model.base_model import BaseFeatureExtractor
from utils.utils import create_logger, get_model_summary

# Libraries
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Facial Feature Classification')

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
                        default='hairs_dataset')
    parser.add_argument('--model_type',
                        help='Types of model',
                        type=str,
                        default='efficientnet_v2_l',
                        choices=['efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s'])
    parser.add_argument('--split_ratio',
                        help='train set valid set ratio',
                        type=float,
                        default=0.9)
    parser.add_argument('--n_class',
                        help='Total Class',
                        type=int,
                        default=45)
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

def main():
    opts = parse_args()

    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    model = BaseFeatureExtractor(config = opts.model_type, n_class = opts.n_class)

    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, 256, 256)
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input, verbose=True))

    # Load Pretrained model Here
    ckpt = torch.load('weights/efficientnet_v2_l_best_model.pt')
    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()

    # Preparing Dataset
    trans = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    data = ImageFolder('data/scrapped_filtered', transform=trans)
    data_test = ImageFolder('data/scrapped_filtered', transform=trans)

    torch.manual_seed(1)
    indices = torch.randperm(len(data)).tolist()
    split_idx = int(0.2 * len(data))

    trainset = torch.utils.data.Subset(data, indices[:-split_idx])
    valset = torch.utils.data.Subset(data_test, indices[-split_idx:])


    train_loader = DataLoader(trainset, batch_size=24, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(valset, batch_size=24, shuffle=True, pin_memory=True, drop_last=True, num_workers=8)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), opts.max_lr, weight_decay=opts.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, opts.max_lr, epochs=opts.epochs, steps_per_epoch=len(train_loader))

    val_acc = 0.0
    for epoch in range(opts.epochs):

        model.train()
        losses = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description("Epoch {}".format(epoch))

                optimizer.zero_grad()
                pred = model(data.cuda())

                if opts.model_type == 'xception':
                    pred = pred[0]

                loss = loss_fn(pred, target.cuda())
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

        model.eval()
        losses = 0
        pred_list = []
        tar_list = []

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for data, target in tepoch:
                    tepoch.set_description("Epoch {}".format(epoch))
                    pred = model(data.cuda())

                    _, predicted = torch.max(pred.data, 1)

                    tar_list.append(target.numpy())
                    pred_list.append(predicted.cpu().numpy())

                    loss = loss_fn(pred, target.cuda())

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

        if mats['accuracy'] > val_acc:
            ckpt = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'val_loss' : val_loss,
                'acc': mats['accuracy'],
                'precision' : mats['precision'],
                'f1' : mats['f1'],

            }
            ckpt.update(mats)
            torch.save(ckpt, '{}/{}_best_model.pt'.format(opts.modelDir, opts.model_type))
            val_acc = mats['accuracy']

if __name__ == '__main__':
    main()
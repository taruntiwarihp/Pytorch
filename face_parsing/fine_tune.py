from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Local files
from model import *
from utils.utils import create_logger, get_model_summary
from model.utils import criterion, calculate_matrix

# Libraries
import torch
from torch.utils.data import DataLoader
from dataset.dataset import MultiFeatureFaceData
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

    logger, tb_log_dir = create_logger(opts, phase = 'finetune')
    logger.info(pprint.pformat(opts))

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    base_model = BaseFeatureExtractor(config = opts.model_type, n_class = opts.n_class)
    base_model.load_state_dict(torch.load('weights/efficientnet_v2_l_best_model.pt', map_location='cpu')['model'])
    base_model = torch.nn.Sequential(*list(base_model.backbone.children())[:-2])

    model = HairFeature(base_model)
    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, 256, 256)
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    # logger.info(get_model_summary(model, dump_input, verbose=True))

    # # Load Pretrained model Here
    # ckpt = torch.load('weights/efficientnet_v2_l_best_model.pt')
    # model.load_state_dict(ckpt['model'])
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()

    # Preparing Dataset
    trans = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    data = MultiFeatureFaceData('data/scrapped_filtered', transform=trans)
    data_test = MultiFeatureFaceData('data/scrapped_filtered', transform=trans)

    torch.manual_seed(1)
    indices = torch.randperm(len(data)).tolist()
    split_idx = int(0.2 * len(data))

    trainset = torch.utils.data.Subset(data, indices[:-split_idx])
    valset = torch.utils.data.Subset(data_test, indices[-split_idx:])


    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True, num_workers=8)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), opts.max_lr, weight_decay=opts.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, opts.max_lr, epochs=opts.epochs, steps_per_epoch=len(train_loader))

    val_acc = 0.0
    for epoch in range(opts.epochs):

        model.train()
        losses = 0
        with tqdm(train_loader, unit="batch") as tloader:
            for batch in tloader:
                tloader.set_description("Epoch {}".format(epoch))

                data = batch['image']
                target = batch['label']

                optimizer.zero_grad()
                pred = model(data.cuda())

                loss = criterion(loss_fn, pred, target)
                # loss = loss_fn(pred, target.cuda())
                loss.backward()
                optimizer.step()
                sched.step()

                losses += loss.item()

                tloader.set_postfix(loss=loss.item())

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
            with tqdm(val_loader, unit="batch") as tloader:
                for batch in tloader:
                    tloader.set_description("Epoch {}".format(epoch))
                    
                    data = batch['image']
                    target = batch['label']
                    
                    pred = model(data.cuda())

                    # _, predicted = torch.max(pred.data, 1)

                    tar_list.append(target)
                    pred_list.append(pred)

                    loss = criterion(loss_fn, pred, target)
                    # loss = loss_fn(pred, target.cuda())

                    losses += loss.item()

                    tloader.set_postfix(loss=loss.item())


        mats = calculate_matrix(pred_list, tar_list)
        # mats = {
        #     'loss': losses/len(val_loader),
        #     'accuracy' : accuracy_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten()),
        #     'precision' : precision_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten(), average='macro'),
        #     'f1' : f1_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten(), average='macro'),
        # }
        
        tmp = [mats[key]['accuracy'] for key in ['hair', 'beard', 'mustache']]
        avg_acc = sum(tmp)/len(tmp)
        tmp = [mats[key]['precision'] for key in ['hair', 'beard', 'mustache']]
        avg_precision = sum(tmp)/len(tmp)
        tmp = [mats[key]['f1'] for key in ['hair', 'beard', 'mustache']]
        avg_f1 = sum(tmp)/len(tmp)
        val_loss = losses/len(val_loader)

        # Logs
        msg = 'Validation Epoch : {}\t Validation Loss : {}\t Accuracy : {}\t Precision : {}\t F1-Score : {}\t'.format(
                        epoch, val_loss, avg_acc, avg_precision, avg_f1)

        logger.info(msg)
        logger.info(pprint.pformat(mats))

        global_steps = writer_dict['valid_global_steps']
        writer_dict['writer'].add_scalar('val_loss', val_loss, global_steps)
        writer_dict['writer'].add_scalar('accuracy', avg_acc, global_steps)
        writer_dict['writer'].add_scalar('precision', avg_precision, global_steps)
        writer_dict['writer'].add_scalar('f1', avg_f1, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        if avg_acc > val_acc:
            ckpt = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'val_loss' : val_loss

            }
            ckpt.update(mats)
            torch.save(ckpt, '{}/{}_best_model_finetune.pt'.format(opts.modelDir, opts.model_type))
            val_acc = avg_acc

if __name__ == '__main__':
    main()
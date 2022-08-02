from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.atten_classifier import AttentionClassifier
from dataset import HairDataset, get_transform
from utils import get_model_summary, create_logger
import subprocess
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import os
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Attention_Classifier')

    parser.add_argument('--modelDir', help='model Dir to save', type=str, default='weights')
    parser.add_argument('--logDir', help='log Dir to save logs and tensorboard', type=str, default='logs')
    parser.add_argument('--dataDir', help='Dataset dir', type=str, default='hairs_dataset')
    parser.add_argument('--model_type', help='Model arch', type=str, default='resnet', choices=['base', 'wide_res', 'efficientnet_b7', 'resnet'])
    parser.add_argument('--n_class', help='number of classes', type=int, default=14)
    parser.add_argument('--epochs', help='Total epoches', type=int, default=500)
    parser.add_argument('--max_lr', help='Maximum Learning Rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', help='Weight Decay per epoch', type=float, default=1e-4)
    parser.add_argument('--drop_rate', help='Dropout Probability', type=float, default=0.2)

    # Hyperparameters for Self Correction
    parser.add_argument('--beta', help='Ratio of high importance group in one mini-batch', type=float, default=0.7)
    parser.add_argument('--margin_1', help='Rank Regularization margin', type=float, default=0.15)
    parser.add_argument('--margin_2', help='Relable margin', type=float, default=0.2)
    parser.add_argument('--relabel_epoch', help='Relabeling samples', type=int, default=500)

    args = parser.parse_args()

    return args


def check_dir(opts):

    if not os.path.isdir(opts.dataDir):
        raise ValueError('Data Dir shouldnot be empty')

    os.makedirs(opts.logDir, exist_ok=True)
    os.makedirs(opts.modelDir, exist_ok=True)

def main(mode='male'):

    opts = parse_args()

    if mode == 'male':
        opts.n_class = 11
    else:
        opts.n_class = 14

    

    check_dir(opts)

    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # CUDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    model = AttentionClassifier(drop_rate=opts.drop_rate, n_class=opts.n_class, model_type=opts.model_type)

    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, 224, 224)
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input, verbose=True))

    model.cuda()

    # Data Preparation
    data = HairDataset(opts.dataDir, get_transform(train=True), mode=mode)
    data_test = HairDataset(opts.dataDir, get_transform(train=False), mode=mode)

    torch.manual_seed(1)
    indices = torch.randperm(len(data)).tolist()
    split_idx = int(0.2 * len(data))

    trainset = torch.utils.data.Subset(data, indices[:-split_idx])
    valset = torch.utils.data.Subset(data_test, indices[-split_idx:])


    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)

    # Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimise
    optimizer = torch.optim.Adam(model.parameters(), opts.max_lr, weight_decay = opts.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

    for epoch in range(opts.epochs):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0

        model.train()
        val_acc = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for imgs, targets, indexes in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                
                imgs = imgs.cuda()
                targets = targets.cuda()

                batch_sz = imgs.size(0)
                iter_cnt += 1
                tops = int(batch_sz * opts.beta)

                optimizer.zero_grad()
                att_weights, outs = model(imgs)

                # Rank Regularization
                _, top_idx = torch.topk(att_weights.squeeze(), tops)
                _, down_idx = torch.topk(att_weights.squeeze(), batch_sz - tops, largest=False)

                high_group = att_weights[top_idx]
                low_group = att_weights[down_idx]

                high_mean = torch.mean(high_group)
                low_mean = torch.mean(low_group)

                diff = low_mean - high_mean + opts.margin_1

                if diff > 0:
                    RR_loss = diff
                else:
                    RR_loss = 0.0

                loss = loss_fn(outs, targets) + RR_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.data
                _, predicts = torch.max(outs, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num

                if epoch >= opts.relabel_epoch:
                    sm = torch.softmax(outs, dim = 1)
                    P_max, predicted_labels = torch.max(sm, 1)
                    P_gt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()

                    flag = P_max - P_gt > opts.margin_2

                    if torch.all(flag==False):
                        continue

                    update_idx = flag.nonzero().squeeze()
                    lbl_idx = indexes[update_idx]
                    relabels = predicted_labels[update_idx]
                    train_loader.dataset.dataset.valid_labels[lbl_idx.cpu().numpy()] = relabels.cpu().numpy().astype(int)
        
        sched.step()
        acc = correct_sum.float() / float(train_loader.dataset.__len__())
        running_loss = running_loss/iter_cnt

        msg = 'Train Epoch : {}\t Training Loss : {}\t Train Accuracy : {}'.format(epoch, running_loss, acc)
        logger.info(msg)

        global_steps = writer_dict['train_global_steps']
        writer_dict['writer'].add_scalar('train_loss', running_loss, global_steps)
        writer_dict['writer'].add_scalar('train_accuracy', acc, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1        

        with torch.no_grad():
            running_loss = 0.0
            pred_list = []
            tar_list = []
            iter_cnt = 0
            final_cnt = 0
            samp_cnt = 0
            model.eval()

            with tqdm(val_loader, unit="batch") as tepoch:
                for imgs, targets, _ in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")

                    imgs = imgs.cuda()
                    targets = targets.cuda()

                    _, outs = model(imgs)
                    loss = loss_fn(outs, targets)

                    running_loss += loss.data
                    iter_cnt += 1

                    _, preds = torch.max(outs, 1)

                    tar_list.append(targets.cpu().numpy())
                    pred_list.append(preds.cpu().numpy())

                    correct_num = torch.eq(preds, targets)
                    final_cnt += correct_num.sum().cpu()
                    samp_cnt += outs.size(0)

                running_loss = running_loss/iter_cnt
                
                
        accuracy = accuracy_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten())
        precision = precision_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten(), average='macro')
        f1 = f1_score(np.vstack(pred_list).flatten(), np.vstack(tar_list).flatten(), average='macro')



        msg = 'Validation Epoch : {}\t Validation Loss : {}\t Accuracy : {}\t Precision : {}\t F1-Score : {}\t'.format(
            epoch, running_loss, accuracy, precision, f1)
        logger.info(msg)

        global_steps = writer_dict['valid_global_steps']
        writer_dict['writer'].add_scalar('val_loss', running_loss, global_steps)
        writer_dict['writer'].add_scalar('val_accuracy', accuracy, global_steps)
        writer_dict['writer'].add_scalar('precision', precision, global_steps)
        writer_dict['writer'].add_scalar('f1', f1, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        if acc > val_acc:
            val_acc = acc

            ckpt = {
                'epoch': epoch,
                'model':  model.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': accuracy
            }

            torch.save(ckpt, os.path.join(opts.modelDir, 'best_model_{}_{}.pt'.format(mode, opts.model_type)))


    writer_dict['writer'].close()

    print('Training Completed')

def auto_shutdown(t=0):
    time.sleep(t)
    cmdCommand = "shutdown -h" # now
    process = subprocess.Popen(cmdCommand.split(), stdout=subprocess.PIPE)

if __name__ == '__main__':
    # main(mode='male')
     
    # time.sleep(600)
    main(mode='female')
    auto_shutdown(t=1200)
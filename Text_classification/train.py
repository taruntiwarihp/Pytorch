from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
import numpy as np
import pprint
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from transformers import get_linear_schedule_with_warmup, AdamW, BertTokenizer

from model import Base_Model, RoBERT_Model
from dataset.dataset import Twitchdataset
from utils.logging import create_logger, get_model_summary
from utils.utils import my_collate, prepare_batch_for_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Texture_swin')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='weights')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='logs')
    parser.add_argument('--bert_config',
                        help='Bert Model Config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-large-uncased', 'distilbert-base-uncased']) # experiment
    parser.add_argument('--root',
                        help='Root Location',
                        type=str,
                        default='data/chats_data_sampled_combined.xlsx - Combined_round_4k.csv')
    parser.add_argument('--model_type',
                        help='Types of model',
                        type=str,
                        default='robert',
                        choices=['robert']) 
    parser.add_argument('--lstm_hidden_layer',
                        help='LSTM Hidden Layer on RoBert Model',
                        type=int,
                        default=100)
    parser.add_argument('--n_class',
                        help='Total number of Labels',
                        type=int,
                        default=3) 
    parser.add_argument('--kfold',
                        help='K-Fold Cross Validation',
                        type=int,
                        default=10) 
    parser.add_argument('--split_ratio',
                        help='train set valid set ratio',
                        type=float,
                        default=0.8)
    parser.add_argument('--chunk_len',
                        help='Base Chunk Length',
                        type=int,
                        default=10)                    
    parser.add_argument('--overlap_len',
                        help='Overlap token length between two Chunks',
                        type=int,
                        default=5)
    parser.add_argument('--batch_size',
                        help='train set valid set Batch size',
                        type=int,
                        default=16)
    parser.add_argument('--pretrain_epochs',
                        help='Total Epochs for Pretraining',
                        type=int,
                        default=50)
    parser.add_argument('--finetune_epochs',
                        help='Total Epochs for Finetuning',
                        type=int,
                        default=50)
    parser.add_argument('--lr',
                        help='Learning Rate',
                        type=float,
                        default=3e-5)
    parser.add_argument('--pretrain_warmup_steps',
                        help='Pretraining Warmup Step',
                        type=int,
                        default=300)

    args = parser.parse_args()

    return args

def main():
    opts = parse_args()

    # Create Weight dic
    os.makedirs(opts.modelDir, exist_ok=True)

    logger, tb_log_dir = create_logger(opts)
    logger.info(pprint.pformat(opts))

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    base_model = Base_Model(bert_config=opts.bert_config, n_class=opts.n_class)
    logger.info(pprint.pformat(base_model))
    base_model.cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'base_train_global_steps': 0,
        'main_train_global_steps': 0,
        'main_valid_global_steps': 0,
    }


    # Data Prep
    bert_tokenizer = BertTokenizer.from_pretrained(opts.bert_config, do_lower_case=True)
    dataset = Twitchdataset(
        root=opts.root,
        tokenizer=bert_tokenizer,
        chunk_len = opts.chunk_len,
        overlap_len = opts.overlap_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        collate_fn=my_collate,
        drop_last=True,
    )

    logger.info('Starting Pretraining')

    num_training_steps = int(len(dataset) / opts.batch_size * opts.pretrain_epochs)
    optimizer=AdamW(base_model.parameters(), lr=opts.lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = opts.pretrain_warmup_steps,
        num_training_steps = num_training_steps
    )
    loss_fun = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 1+opts.pretrain_epochs):

        base_model.train()
        losses = 0
        with tqdm(dataloader, unit="batch") as tloader:
            for batch in tloader:
                tloader.set_description("Epoch {}".format(epoch))
                ids, mask, token_type_ids, targets, _ = prepare_batch_for_model(batch, fine_tune=False)

                optimizer.zero_grad()
                outputs = base_model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = loss_fun(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses += loss.item()
                tloader.set_postfix(loss=loss.item())

        train_loss = losses/len(dataloader)
        msg = 'Train Epoch : {}\t Training Loss : {}'.format(epoch, train_loss)
        logger.info(msg)

        global_steps = writer_dict['base_train_global_steps']
        writer_dict['writer'].add_scalar('base_train_loss', train_loss, global_steps)
        writer_dict['base_train_global_steps'] = global_steps + 1

    logger.info('Pretraining Completed')
    torch.save(base_model, os.path.join(opts.modelDir, 'base_model.pt'))
    logger.info('Pretrained model saved at {}'.format(os.path.join(opts.modelDir, 'base_model.pt')))
    
    logger.info('Starting kFold Cross Validation for finetuning')

    kfold = KFold(n_splits=opts.kfold, shuffle=True)

    fold_mats = {
        'accuracy' : {},
        'precision' : {},
        'f1' : {},
        'recall' : {},
    }
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

        val_acc_track = 0.0

        logger.info('Starting Fold {}'.format(fold+1))

        writer_dict['main_train_global_steps'] = 0
        writer_dict['main_valid_global_steps'] = 0

        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(val_ids)

        train_data_loader=DataLoader(
            dataset,
            batch_size=opts.batch_size,
            sampler=train_sampler,
            collate_fn=my_collate,
            )

        valid_data_loader=DataLoader(
            dataset,
            batch_size=opts.batch_size,
            sampler=valid_sampler,
            collate_fn=my_collate)

        # Define Model
        child_model=torch.load(os.path.join(opts.modelDir, 'base_model.pt'))
        model=RoBERT_Model(bertFineTuned=list(child_model.children())[0], lstm_hidden_layer = opts.lstm_hidden_layer, n_class = opts.n_class).to("cuda")
 
        # optimizer
        num_training_steps = int(len(dataset) / opts.batch_size * opts.finetune_epochs)
        optimizer=AdamW(model.parameters(), lr=opts.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps
        )

        loss_fun = torch.nn.CrossEntropyLoss()

        for epoch in range(1, 1+opts.finetune_epochs):

            current_loss = 0.0
            model.train()

            with tqdm(train_data_loader, unit="batch") as tloader:
                for batch in tloader:
                    
                    tloader.set_description("Epoch {}".format(epoch))
                    ids, mask, token_type_ids, targets, lengt = prepare_batch_for_model(batch)

                    optimizer.zero_grad()
                    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
                    loss = loss_fun(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    current_loss += loss.item()
                    tloader.set_postfix(loss=loss.item())

            train_loss = current_loss/len(train_data_loader)
            msg = 'Fold : {}\t Epoch : {}\t Training Loss : {}'.format(fold, epoch, train_loss)
            logger.info(msg)

            global_steps = writer_dict['main_train_global_steps']
            writer_dict['writer'].add_scalar('fold_{}_main_train_loss'.format(fold), train_loss, global_steps)
            writer_dict['main_train_global_steps'] = global_steps + 1

            model.eval()
            current_loss = 0.0

            fin_targets = []
            fin_outputs = []

            with torch.no_grad():
                with tqdm(valid_data_loader, unit="batch") as tloader:
                    for batch in tloader:

                        tloader.set_description("Epoch {}".format(epoch))
                        ids, mask, token_type_ids, targets, lengt = prepare_batch_for_model(batch)

                        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, lengt=lengt)
                        loss = loss_fun(outputs, targets)
                        current_loss += loss.item()
                        tloader.set_postfix(loss=loss.item())

                        _, pred = torch.max(outputs.data, 1)

                        fin_targets.append(targets.cpu().detach().numpy())
                        # fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
                        fin_outputs.append(pred.cpu().detach().numpy())

            val_loss = current_loss/len(valid_data_loader)
            target = np.concatenate(fin_targets)
            predicted = np.concatenate(fin_outputs)
            # true_label_mask = [1 if (np.argmax(x)-target[i]) == 0 else 0 for i, x in enumerate(predicted)]
            # nb_prediction = len(true_label_mask)
            # true_prediction = sum(true_label_mask)
            # false_prediction = nb_prediction-true_prediction
            # accuracy = true_prediction/nb_prediction
            cm = confusion_matrix(target, predicted)
            class_acc = cm.diagonal()/cm.sum(axis=1)
            class_precision = precision_score(predicted, target, average=None)
            class_f1 = f1_score(predicted, target, average=None)
            class_recall = recall_score(predicted, target, average=None)

            accuracy = accuracy_score(predicted, target)
            precision = precision_score(predicted, target, average='micro')
            f1 = f1_score(predicted, target, average='micro')
            recall = recall_score(predicted, target, average='micro')

            logger.info('Class Accuracy')
            logger.info(pprint.pformat(class_acc))
            logger.info('Class Precision')
            logger.info(pprint.pformat(class_precision))
            logger.info('Class F1 Score')
            logger.info(pprint.pformat(class_f1))
            logger.info('Class Recall')
            logger.info(pprint.pformat(class_recall))

            msg = 'Fold : {}\t Epoch : {}\t Validation Loss {}\t Accuracy : {}\t Precision : {}\t F1_score : {}\t Recall : {}\t'.format(
                fold, epoch, val_loss, accuracy, precision, f1, recall
            )
            logger.info(msg)

            global_steps = writer_dict['main_valid_global_steps']
            writer_dict['writer'].add_scalar('fold_{}_main_val_loss'.format(fold), val_loss, global_steps)
            writer_dict['writer'].add_scalar('fold_{}_main_accuracy'.format(fold), accuracy, global_steps)
            writer_dict['writer'].add_scalar('fold_{}_main_precision'.format(fold), precision, global_steps)
            writer_dict['writer'].add_scalar('fold_{}_main_f1_score'.format(fold), f1, global_steps)
            writer_dict['writer'].add_scalar('fold_{}_main_recall'.format(fold), recall, global_steps)
            writer_dict['main_valid_global_steps'] = global_steps + 1

            # Save Weights
            if accuracy > val_acc_track:
                ckpt = {
                    'fold': fold,
                    'model_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'eval_loss': val_loss,
                    'accuracy': accuracy,
                    'precision' : precision,
                    'f1_score' : f1,
                    'recall' : recall,
                    'class_acc' : class_acc,
                    'class_precision' : class_precision,
                    'class_f1' : class_f1,
                    'class_recall' : class_recall,
                }

                save_path = os.path.join(opts.modelDir, 'fold_{}_RoBERT_best_model.pt'.format(fold))
                torch.save(ckpt, save_path)

                logger.info('Best Model saved with accuracy {}'.format(accuracy))
                val_acc_track = accuracy
        
        logger.info('Fold {} Finished.'.format(fold))
        logger.info('Best eval matrix.')

        logger.info('Class Accuracy')
        logger.info(pprint.pformat(ckpt['class_acc']))
        logger.info('Class Precision')
        logger.info(pprint.pformat(ckpt['class_precision']))
        logger.info('Class F1 Score')
        logger.info(pprint.pformat(ckpt['class_f1']))
        logger.info('Class Recall')
        logger.info(pprint.pformat(ckpt['class_recall']))
        
        fold_mats['accuracy'][str(fold + 1)] = ckpt['accuracy']
        fold_mats['precision'][str(fold + 1)] = ckpt['precision']
        fold_mats['f1'][str(fold + 1)] = ckpt['f1_score']
        fold_mats['recall'][str(fold + 1)] = ckpt['recall']

        # over_all_acc += val_acc_track

    msg = 'All folds are finished '
    for k in list(fold_mats.keys()):
        tmp = 0
        for fold in range(1, 1+ opts.kfold):
            tmp += fold_mats[k][str(fold)]
        
        tmp = tmp / opts.kfold
        tmp_msg = 'overall {} is {} '.format(k, tmp)

        msg += tmp_msg

    logger.info(msg)
    # over_all_acc = over_all_acc / opts.kfold
    # logger.info('All folds are finished overall accuracy is {}'.format(over_all_acc))
    logger.info(pprint.pformat(fold_mats))

    logger.info('Finished Training')

if __name__ == '__main__':
    main()    
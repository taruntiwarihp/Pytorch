from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from tqdm import tqdm
import numpy as np
import pprint
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import get_linear_schedule_with_warmup, AdamW, BertTokenizer

from models import CharacterBertModel
# from datasets.phishing_url import PhishingURLChar, PhishingURL
from datasets.multi_url import URLIdentificationDataset
from utils.logger import create_logger


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
                        default='data/URL')
    parser.add_argument('--model_type',
                        help='Types of model',
                        type=str,
                        default='charbert',
                        choices=['charbert', 'bert']) 
    parser.add_argument('--n_class',
                        help='Total number of Labels',
                        type=int,
                        default=5) 
    parser.add_argument('--split_ratio',
                        help='train set valid set ratio',
                        type=float,
                        default=0.2)
    parser.add_argument('--max_length',
                        help='Max length in one description',
                        type=int,
                        default=256)
    parser.add_argument('--batch_size',
                        help='train set valid set Batch size',
                        type=int,
                        default=32)
    parser.add_argument('--finetune_epochs',
                        help='Total Epochs for Finetuning',
                        type=int,
                        default=50)
    parser.add_argument('--learning_rate',
                        help='Learning Rate',
                        type=float,
                        default=5e-5)
    parser.add_argument('--weight_decay',
                        help='Weight decay if we apply some.',
                        type=float,
                        default=0.1)
    parser.add_argument('--warmup_ratio',
                        help='Linear warmup over warmup_ratio*total_steps.',
                        type=float,
                        default=0.1)
    parser.add_argument('--adam_epsilon',
                        help='Epsilon for Adam optimizer.',
                        type=float,
                        default=1e-8)                     

    args = parser.parse_args()

    return args

# writer.add_scalar("loss/train/g/total", loss, train_step)
def main():
    opts = parse_args()

    os.makedirs(opts.modelDir, exist_ok=True)
    os.makedirs(os.path.join(opts.modelDir, opts.model_type), exist_ok=True)

    logger, tb_dir = create_logger(opts)
    writer = SummaryWriter(log_dir=tb_dir)

    logger.info(pprint.pformat(opts))

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Define dataset
    torch.manual_seed(1)
    bert_tokenizer = BertTokenizer.from_pretrained(opts.bert_config, do_lower_case=True)
    data = URLIdentificationDataset(
        root=opts.root,
        tokenizer=bert_tokenizer,
        max_len=opts.max_length,
    )
    indices = torch.randperm(len(data)).tolist()
    split_idx = int(opts.split_ratio * len(data))

    trainset = torch.utils.data.Subset(data, indices[:-split_idx])
    valset = torch.utils.data.Subset(data, indices[-split_idx:])

    train_loader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(valset, batch_size=opts.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

    logger.info('Train set length {}\t Eval set length {}'.format(len(trainset), len(valset)))

    # Model
    model = CharacterBertModel(
        bert_config=opts.bert_config,
        num_labels=opts.n_class
    ).to('cuda:0')

    logger.info(model)

    # Prepare optimizer and schedule (linear warmup and decay)

    num_training_steps = int(len(trainset) / opts.batch_size * opts.finetune_epochs)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": opts.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.learning_rate, eps=opts.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(opts.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )

    val_acc_track = 0.0

    for epoch in range(opts.finetune_epochs):

        current_loss = 0.0
        model.train()

        with tqdm(train_loader, unit="batch") as tloader:
            for batch in tloader:
                batch = {k: v.to(device='cuda:0', non_blocking=True) for k, v in batch.items()}
                
                tloader.set_description("Epoch {}".format(epoch))

                optimizer.zero_grad()

                outputs = model(**batch)
                loss = outputs[0]

                loss.backward()
                optimizer.step()
                scheduler.step()
                current_loss += loss.item()
                tloader.set_postfix(loss=loss.item())

        train_loss = current_loss/len(train_loader)

        msg = 'Train Epoch : {}\t Training Loss : {}'.format(epoch, train_loss)
        logger.info(msg)

        # Tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)

        model.eval()
        current_loss = 0.0

        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tloader:
                for batch in tloader:
                    
                    batch = {k: v.to(device='cuda:0', non_blocking=True) for k, v in batch.items()}

                    targets = batch['labels']
                    tloader.set_description("Epoch {}".format(epoch))

                    outputs = model(**batch)
                    loss = outputs[0]
                    current_loss += loss.item()
                    tloader.set_postfix(loss=loss.item())

                    _, pred = torch.max(outputs[1], 1)

                    fin_targets.append(targets.cpu().detach().numpy())
                    fin_outputs.append(pred.cpu().detach().numpy())

        val_loss = current_loss/len(val_loader)
        target = np.concatenate(fin_targets)
        predicted = np.concatenate(fin_outputs)

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

        msg = 'Eval Epoch : {}\t Validation Loss {}\t Accuracy : {}\t Precision : {}\t F1_score : {}\t Recall : {}\t'.format(
            epoch, val_loss, accuracy, precision, f1, recall
        )
        logger.info(msg)

        writer.add_scalar("eval/loss", val_loss, epoch)
        writer.add_scalar("eval/accuracy", accuracy, epoch)
        writer.add_scalar("eval/precision", precision, epoch)
        writer.add_scalar("eval/f1", f1, epoch)
        writer.add_scalar("eval/recall", recall, epoch)

        # Save Weights
        if accuracy > val_acc_track:
            ckpt = {
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'eval_loss': val_loss,
                'epoch': epoch,
            }

            mats = {
                'accuracy': accuracy,
                'precision' : precision,
                'f1_score' : f1,
                'recall' : recall,
                'class_acc' : class_acc,
                'class_precision' : class_precision,
                'class_f1' : class_f1,
                'class_recall' : class_recall,
            }

            ckpt.update(mats)

            
            save_path = os.path.join(opts.modelDir, opts.model_type, 'best_model.pt')
            torch.save(ckpt, save_path)

            logger.info('Best Model saved')
            logger.info(pprint(mats))
            val_acc_track = accuracy

    logger.info('==>Training done!\nBest accuracy: %.3f' % (val_acc_track))
    logger.info('Best Evaluation matrix')
    logger.info(pprint(mats))

if __name__ == '__main__':
    main()
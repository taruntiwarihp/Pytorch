# import torch
# import torchvision.models as models
# from PIL import Image
# import torchvision.transforms as T
# from models.model import HairClassificationModelMobile

# def get_transform(train):
# 	transforms = []
# 	transforms.append(T.Resize((224, 224)))
# 	transforms.append(T.ToTensor())
# 	transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

# 	if train:
# 		transforms.append(T.RandomHorizontalFlip(0.5))

# 	return T.Compose(transforms)

# MALE_CLASSES = ['afro', 'buzz', 'curly', 'classic spikes with fade', 'dreadlocks', 'flattop', 'pompadour', 'ponytail', 'side slick', 'slick back', 'spiky']


# #img =  Image.open('hairs_dataset/afro/0be3a98cde.jpg')
# img = Image.open('hairs_dataset/classic spikes with fade/0abff3aa78.jpg')
# trans_fn = get_transform(False)
# img_trans = trans_fn(img)
# print(img_trans.shape)

# img_trans = torch.unsqueeze(img_trans, 0)
# print(img_trans.shape)

# model = HairClassificationModelMobile(11)
# ckpt = torch.load('weights/mobile_net_best_model.pt')
# model.load_state_dict(ckpt['model'])

# pred = model(img_trans)

# pred_prob = torch.softmax(pred, dim=1).cpu()
# print(pred_prob)

# prob, idx = torch.topk(pred_prob, k=1)
# print(idx)

# from models import create_model

# class Arg(object):

# 	def __init__(self):
# 		pass

# args = Arg()
# args.model_type = 'efficientnet_b0'
# args.n_class = 11

# model = create_model(args)

from atten_main import parse_args

from models.atten_classifier import AttentionClassifier
from dataset import HairDataset, get_transform
from torch.utils.data import DataLoader
import torch
import numpy as np


data = HairDataset("hairs_dataset", get_transform(train=True))
data_test = HairDataset("hairs_dataset", get_transform(train=False))
torch.manual_seed(1)
indices = torch.randperm(len(data)).tolist()
split_idx = int(0.2 * len(data))

trainset = torch.utils.data.Subset(data, indices[:-split_idx])
valset = torch.utils.data.Subset(data_test, indices[-split_idx:])
train_loader = DataLoader(trainset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(valset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)

imgs, targets, indexes = next(iter(train_loader))

imgs = imgs.cuda()
targets = targets.cuda()

opts = parse_args()

running_loss = 0.0
correct_sum = 0
iter_cnt = 0
val_acc = 0.0
batch_sz = imgs.size(0)

iter_cnt += 1
tops = int(batch_sz * opts.beta)

model = AttentionClassifier(drop_rate=0.2, model_type='resnet', n_class=11)
optimizer = torch.optim.Adam(model.parameters(), opts.max_lr, weight_decay = opts.weight_decay)
sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
model.cuda()
model.train()

for i, (imgs, targets, indexes) in enumerate(train_loader):

    imgs = imgs.cuda()
    targets = targets.cuda()

    att, outs = model(imgs)

    _, top_idx = torch.topk(att.squeeze(), tops)
    _, down_idx = torch.topk(att.squeeze(), batch_sz - tops, largest=False)

    high_group = att[top_idx]
    low_group = att[down_idx]

    high_mean = torch.mean(high_group)
    low_mean = torch.mean(low_group)

    diff = low_mean - high_mean + opts.margin_1

    if diff > 0:
        RR_loss = diff
    else:
        RR_loss = 0.0


    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outs, targets) + RR_loss
    loss.backward()
    optimizer.step()

    running_loss += loss.data
    _, predicts = torch.max(outs, 1)
    correct_num = torch.eq(predicts, targets).sum()
    correct_sum += correct_num

    sm = torch.softmax(outs, dim = 1)
    P_max, predicted_labels = torch.max(sm, 1)
    P_gt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()

    flag = P_max - P_gt > opts.margin_2

    print(i, torch.all(flag==False), flag)

    if ~torch.all(flag==False):



        update_idx = flag.nonzero().squeeze()
        lbl_idx = indexes[update_idx]
        relabels = predicted_labels[update_idx]
        print(lbl_idx.cpu().numpy())
        train_loader.dataset.dataset.valid_labels[np.array(lbl_idx.cpu().numpy(), dtype=np.int8)] = relabels.cpu().numpy()

        
from dataset.dataset import SegmentationDataset, get_transform
import torchvision.transforms as T
from model.model import createDeepLabv3
from train_gpu import parse_args
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import torch
from PIL import Image
# class DeepLabV3Wrapper(torch.nn.Module):
#     def __init__(self, model):
#         super(DeepLabV3Wrapper, self).__init__()
#         self.model = model

#     def forward(self, input):
#         output = self.model(input)['out']
#         return output

# def initialize_model(num_classes, keep_feature_extract=False, use_pretrained=False):
#     """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
#     """
#     model_deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=use_pretrained, progress=True)
#     model_deeplabv3.aux_classifier = None
#     if keep_feature_extract:
#         for param in model_deeplabv3.parameters():
#             param.requires_grad = False

#     model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

#     return model_deeplabv3




data = 'human_segmentation'
trans = get_transform(False)

# transforms = T.Compose([
# 			T.RandomHorizontalFlip(),
# 			T.RandomVerticalFlip(),
# 			# T.RandomResizedCrop((512, 512)),
# 			T.RandomCrop((224, 224)),
# 			T.ToTensor(),
# 			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 		])

Seg = SegmentationDataset(root = data, transforms = trans)
opts = parse_args()
train_loader = DataLoader(Seg, batch_size=opts.batch_size, shuffle=True)
print(train_loader)
# for img, mask in train_loader:
#     print(img.shape)
#     print(mask.shape)

# i, t  = Seg[1]
# print(i)
# print(t.shape)
# print(i.unsqueeze(0).shape)
# label = t.unsqueeze(0)
# label = label * 255
# label = label.long().squeeze()


# input_batch = i.unsqueeze(0)

# model(input_batch.to('cuda'))

dataset_train = SegmentationDataset(opts.dataDir, get_transform(train=True))
dataset_val = SegmentationDataset(opts.dataDir, get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset_train)).tolist()
split_idx = int(0.2 * len(dataset_train))
print(split_idx)
trainset = torch.utils.data.Subset(dataset_train, indices[:-split_idx])
valset = torch.utils.data.Subset(dataset_val, indices[-split_idx:])
# print(len(trainset), len(valset))

train_loader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=True)
# val_loader = DataLoader(valset, batch_size=opts.batch_size)

# print(len(train_loader))
# # print(len(val_loader))
# print(train_loader.next())
for img in train_loader:
    print(img)


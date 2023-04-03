from .efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, efficientnet_b0, efficientnet_b4, efficientnet_b7
from .conv_trans import ConvTransformer
from .fmnet import FMNet
from .vgg import VGG
from .alexnet import AlexNet
from .resnet import ResNet
from .mobilenet import MobileNetV2
from .inception import InceptionV3
from .efficient_att import EfficientFormer
from .fmnet import FMNet
from .swin import SwinTransformer

from torch import nn
import torch
from torch.nn import functional as F

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class BaseFeatureExtractor(nn.Module):
    def __init__(self, config='efficientnet_v2_l', img_dim=(256, 256), n_class=5):
        super().__init__()

        if config.lower() == 'efficientnet_v2_l':
            self.backbone = efficientnet_v2_l(pretrained=True)

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1280, out_features=n_class, bias=True)
            )

        elif config.lower() == 'efficientnet_v2_s':
            self.backbone = efficientnet_v2_s(pretrained=True)

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1280, out_features=n_class, bias=True)
            )

        elif config.lower() == 'efficientnet_v2_m':
            self.backbone = efficientnet_v2_m(pretrained=True)

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1280, out_features=n_class, bias=True)
            )

        elif config.lower() == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=False)

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1280, out_features=n_class, bias=True)
            )

        elif config.lower() == 'efficientnet_b4':
            self.backbone = efficientnet_b4(pretrained=False)

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1792, out_features=n_class, bias=True)
            )

        elif config.lower() == 'efficientnet_b7':
            self.backbone = efficientnet_b7(pretrained=False)

            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=2560, out_features=n_class, bias=True)
            )

        elif config.lower() == 'conv_trans':
            num_blocks = [2, 2, 12, 28, 2] 
            channels = [192, 192, 384, 768, 1536]

            self.backbone = ConvTransformer(img_dim, 3, num_blocks, channels, num_classes=n_class)

        elif config.lower() == 'fmnet':
            self.backbone = FMNet(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], num_classes=n_class)

        elif config.lower() == 'vgg':
            self.backbone = VGG(num_classes=n_class)

        elif config.lower() == 'alexnet':
            self.backbone = AlexNet(num_classes=n_class)

        elif config.lower() == 'resnet':
            self.backbone = ResNet(num_classes=n_class)

        elif config.lower() == 'mobilenet':
            self.backbone = MobileNetV2(num_classes=n_class)

        elif config.lower() == 'inception':
            self.backbone = InceptionV3(num_classes=n_class)

        elif config.lower() == 'efficient_att':
            self.backbone = EfficientFormer(
                layers=[3, 2, 6, 4],
                embed_dims=[48, 96, 224, 448],
                downsamples=[True, True, True, True],
                vit_num=1,
                num_classes=5,
                distillation=False
            )

        elif config.lower() == 'fmnet':
            self.backbone = FMNet(num_classes=n_class)

        elif config.lower() == 'swin':
            self.backbone = SwinTransformer(num_classes=n_class)

        # elif config.lower() == 'conv_trans':
        #     self.backbone = InceptionV3(num_classes=n_class)


        else:
            raise ValueError('Not supported')



    def forward(self, x):
        return self.backbone(x)

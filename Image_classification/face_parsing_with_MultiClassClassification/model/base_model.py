from .efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l 
from torch import nn
import torch

class BaseFeatureExtractor(nn.Module):
    def __init__(self, config='efficientnet_v2_l', n_class=45):
        super().__init__()

        if config.lower() == 'efficientnet_v2_l':
            self.backbone = efficientnet_v2_l(pretrained=True)
        elif config.lower() == 'efficientnet_v2_s':
            self.backbone = efficientnet_v2_s(pretrained=True)
        elif config.lower() == 'efficientnet_v2_m':
            self.backbone = efficientnet_v2_m(pretrained=True)
        else:
            raise ValueError('Not supported')

        # if n_class:
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=n_class, bias=True)
        )

    def forward(self, x):
        return self.backbone(x)

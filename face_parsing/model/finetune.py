import torch
from torch import nn
from .utils import MLPBlock

class HairFeature(nn.Module):

    def __init__(self, base_model, hair_classes=19, beard_classes=10, mustache_classes=9):
        super().__init__()

        self.base_extractor = base_model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.hair_mlp = MLPBlock(in_dim=1280, mlp_dim=512, out_dim=hair_classes, dropout=0.25)
        self.beard_mlp = MLPBlock(in_dim=1280, mlp_dim=512, out_dim=beard_classes, dropout=0.25)
        self.mustache_mlp = MLPBlock(in_dim=1280, mlp_dim=512, out_dim=mustache_classes, dropout=0.25)

    def forward(self, x):
        with torch.no_grad():
            features = self.base_extractor(x)
        
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        hair_prob = self.hair_mlp(features)
        beard_prob = self.beard_mlp(features)
        mustache_prob = self.mustache_mlp(features)

        # return hair_prob, beard_prob, mustache_prob

        return {
            'hair': hair_prob,
            'beard': beard_prob,
            'mustache': mustache_prob,
        }

class UniqueFeature(nn.Module):
    def __init__(self, base_model, classes=18):
        # hair 18, beard_classes=9, mustache_classes=8
        super().__init__()

        self.base_extractor = base_model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = MLPBlock(in_dim=1280, mlp_dim=1024, out_dim=classes, dropout=0.25)

    def forward(self, x):
        with torch.no_grad():
            features = self.base_extractor(x)
        
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        prob = self.mlp(features)

        return prob
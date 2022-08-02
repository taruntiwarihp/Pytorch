import torch
from torch import nn

from .resnet import build_resnet50, build_wrs50
# from .inception import build_inception_v3
from .efficientnet import build_efficientnet_model

# class Identity(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x

class AttentionClassifier(nn.Module):

    def __init__(self, model_type, pretrained = True, n_class=11, drop_rate=0.2):
        super(AttentionClassifier, self).__init__()

        if model_type == 'resnet':
            model = build_resnet50(pretrained=pretrained)

            self.features = nn.Sequential(*list(model.children())[:-1])
            fc_in_dim = list(model.children())[-1].in_features
        
        elif model_type == 'wide_res':
            model = build_wrs50(pretrained=pretrained)

            self.features = nn.Sequential(*list(model.children())[:-1])
            fc_in_dim = list(model.children())[-1].in_features
        
        # elif model_type == 'inception':
        #     model = build_inception_v3(pretrained=pretrained)
        #     fc_in_dim = model.fc.in_features
        #     model.fc = Identity()
        #     self.features = model

        elif model_type.startswith('efficient'):
            model_type = model_type.replace('_', '-')
            model = build_efficientnet_model(model_name=model_type, pretrained=pretrained)
            self.features = model
            fc_in_dim = model._fc.out_features

        self.model_type = model_type
        self.drop_rate = drop_rate
        
        self.drop = nn.Dropout(self.drop_rate)

        self.fc = nn.Linear(fc_in_dim, n_class) 

        self.att_layer = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = self.drop(x)

        x = x.view(x.size(0), -1)

        attention_weights = self.att_layer(x)
        out = attention_weights * self.fc(x)

        return attention_weights, out
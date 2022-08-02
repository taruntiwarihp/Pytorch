import torch
from torch import nn
from torchvision.models import resnet50

class AttentionResnet(nn.Module):
    """
    Self Correction Class
    """

    def __init__(self, pretrained=True, n_class=1000, drop_rate=0.2):
        super(AttentionResnet, self).__init__()

        # Pretrain model
        base_model = resnet50(pretrained=True)

        # Last layer (FC)
        self.feature = nn.Sequential(*list(base_model.children())[:-1])
        fc_in_dim = list(base_model.children())[-1].in_features

        # Drop out layer
        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)
        
        # Classifier (Logistic Prob.)
        self.fc = nn.Linear(fc_in_dim, n_class)

        # Added Attention Layer
        self.att_layer = nn.Sequential(
            nn.Linear(in_features = fc_in_dim, out_features = 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.feature(x)

        x = self.drop(x)   
        
        x = x.view(x.size(0), -1)

        attention_weights = self.att_layer(x)
        out = attention_weights * self.fc(x)

        return attention_weights, out
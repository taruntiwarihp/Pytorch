# from torchvision.ops.misc import ConvNormActivation
import warnings
import torch
from typing import Callable, Optional, Any
from types import FunctionType
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
# from .base_model import BaseFeatureExtractor
from collections import OrderedDict
from PIL import Image
import io
# from dataset.dataset import SingleFeatureDataset

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )

class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")


class MLPBlock(nn.Sequential):
    """MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act1 = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, mlp_dim)
        self.act2 = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_3 = nn.Linear(mlp_dim, out_dim)
        # self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)
        nn.init.normal_(self.linear_3.bias, std=1e-6)


def criterion(loss_fn, preds, gts):
    losses = 0
    for key in preds:
        losses += loss_fn(preds[key], gts[key].long().cuda())

    return losses

def calculate_matrix(pred_list, gt_list):

    mats = {
        'hair' : {
            'pred' : [],
            'gt' : []
        },
        'beard' : {
            'pred' : [],
            'gt' : []
        },
        'mustache' : {
            'pred' : [],
            'gt' : []
        }
    }

    evaluation_report = {
        'hair' : {},
        'beard' : {},
        'mustache' : {}
    }

    for (pred, gt) in zip(pred_list, gt_list):

        for key in ['hair', 'beard', 'mustache']:
            mats[key]['gt'].append(gt[key].numpy())

            _, predicted = torch.max(pred[key].data, 1)

            mats[key]['pred'].append(predicted.cpu().numpy())

    for key in ['hair', 'beard', 'mustache']:
        accuracy = accuracy_score(np.vstack(mats[key]['pred']).flatten(), np.vstack(mats[key]['gt']).flatten())
        precision = precision_score(np.vstack(mats[key]['pred']).flatten(), np.vstack(mats[key]['gt']).flatten(), average='macro')
        f1 = f1_score(np.vstack(mats[key]['pred']).flatten(), np.vstack(mats[key]['gt']).flatten(), average='macro')

        evaluation_report[key]['accuracy'] = accuracy
        evaluation_report[key]['precision'] = precision
        evaluation_report[key]['f1'] = f1

    return evaluation_report

class Classifier(torch.nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.mlp = MLPBlock(in_dim=1280, mlp_dim=1024, out_dim=n_class, dropout=0.25)

    def forward(self, features):
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        prob = self.mlp(features)

        return prob

class CLassifierProb:

    male_hair_classes = [
        'Afro', 'FlatTop', 'Wavy', 'ClassicSpike', 'Curly',  'Pompadour',
        'Buzz', 'Bun', 'Spiky', 'Ponytail', 'Bald', 'Fringe', 'SidePart',
        'Dreadlocks', 'Slickback'
    ]

    female_hair_classes = [
        'Wavy', 'Fringe', 'Buzz', 'Straight', 'Spiky', 'Bun', 'Dreadlocks', 
        'Slickback', 'Braid', 'Pompadour', 'Afro', 'Ponytail', 'Curly', 'Bob'
    ]

    def __init__(
        self, base_model, hair_ckpt_path, beard_ckpt_path, mustache_ckpt_path, 
        hair_classes, beard_classes, mustache_classes, trans_fn
    ):
        self.trans_fn = trans_fn
        # self.data = SingleFeatureDataset()
        self.hair_classes = hair_classes
        self.beard_classes = beard_classes
        self.mustache_classes = mustache_classes

        self.feature_extractor = base_model
        self.hair_model = self.get_keys(hair_ckpt_path, hair_classes)
        self.beard_model = self.get_keys(beard_ckpt_path, beard_classes)
        self.mustache_model = self.get_keys(mustache_ckpt_path, mustache_classes)

    def get_keys(self, ckpt_path, classes):
        n_class = len(classes)
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        model = Classifier(n_class)
        new_state_dict = OrderedDict()

        for key in model.state_dict().keys():
            new_state_dict[key] = ckpt[key]
        
        model.load_state_dict(new_state_dict)
        model = model.to('cuda:1')
        model.eval()

        return model

    def get_prob(self, model, feature, classes, k=5):
        output = model(feature)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, k)
        pred_cls = [classes[top5_catid[i]] for i in range(top5_prob.size(0))]

        # print(dict(zip(top5_catid, top5_prob)))

        return pred_cls

    def __call__(self, img_bytes, gender=None):
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = self.trans_fn(pil_img).unsqueeze(0).to('cuda:1')

        with torch.no_grad():
            base_feature = self.feature_extractor(img)
            
            hair_class = self.get_prob(self.hair_model, base_feature, self.hair_classes)
            beard_class = self.get_prob(self.beard_model, base_feature, self.beard_classes, k=1)
            mustache_class = self.get_prob(self.mustache_model, base_feature, self.mustache_classes, k=1)

        if gender == 'female':
            beard_class = None
            mustache_class = None

            for c in hair_class:
                if c in self.female_hair_classes:
                    break
        else:
            beard_class = beard_class[0]
            mustache_class = mustache_class[0]
            for c in hair_class:
                if c in self.female_hair_classes:
                    break

        temp = {
            'hair_class' : c,
            'beard_class' : beard_class,
            'mustache_class' : mustache_class
        }

        return temp

        
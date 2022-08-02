from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(opts):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    if opts.model_type == 'mobilenet_v3_large':

        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True,
                                                        progress=True)

        model.aux_classifier = None
        if opts.keep_feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier = DeepLabHead(2048, opts.n_class)

    if opts.model_type == 'resnet101':

        model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                        progress=True)

        model.aux_classifier = None
        if opts.keep_feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier = DeepLabHead(2048, opts.n_class)

    if opts.model_type == 'resnet50':

        model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                        progress=True)
 
        model.aux_classifier = None
        if opts.keep_feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier = DeepLabHead(2048, opts.n_class)

    return model

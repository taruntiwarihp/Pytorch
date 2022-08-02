from .wideResnet import WideResNet

def build_wideresnet(config):

    model = WideResNet(num_classes=config.num_classes,
                       depth=config.depth,
                       widen_factor=config.widen_factor,
                       dropout=0,
                       dense_dropout=config.dense_dropout)
    return model
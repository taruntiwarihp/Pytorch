from model import build_wideresnet
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Classifier')

    parser.add_argument('-')
    parser.add_argument('--device', help='Training Device', type=str, default='cpu') # torch.device('cpu')
    parser.add_argument('--modelDir', help='model Dir to save', type=str, default='weights')
    parser.add_argument('--logDir', help='log Dir to save logs and tensorboard', type=str, default='logs')
    parser.add_argument('--dataDir', help='Dataset dir', type=str, default='hairs_dataset')
    parser.add_argument('--model_type', help='Model arch', type=str, default='resnet', choices=['base', 'wide_res', 'efficientnet_b7', 'resnet'])
    parser.add_argument('--num_classes', help='number of classes', type=int, default=101)
    parser.add_argument('--depth', help='Depth of Network', type=int, default=28)
    parser.add_argument('--widen_factor', help='widen factor', type=int, default=8)
    parser.add_argument('--dense_dropout', help='', type=int, default=101)


    parser.add_argument('--epochs', help='Total epoches', type=int, default=50)
    parser.add_argument('--max_lr', help='Maximum Learning Rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', help='Weight Decay per epoch', type=float, default=5e-4)
    parser.add_argument('--momentum', help='Momentum Rate', type=float, default=0.9)
    parser.add_argument('--drop_rate', help='Dropout Probability', type=float, default=0.2)

    args = parser.parse_args()

    return args



config = ()

model = build_wideresnet(config)
model_params = sum(p.numel() for p in model.parameters())
print('Total parameters {} M'.format(model_params/1e6))
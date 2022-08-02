import torch
from PIL import Image
from torch.nn.modules import Module
from models import create_model
import torchvision.transforms as T
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    # philly
    parser.add_argument('--model_type',
                        help='model type',
                        type=str,
                        default='wide_res')
    parser.add_argument('--n_class',
                        help='Number of Classes',
                        type=int,
                        default='11')

    args = parser.parse_args()

    return args


def get_transform(img_dim=224):
	transforms = []
	transforms.append(T.Resize((img_dim, img_dim)))
	transforms.append(T.ToTensor())
	transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

	return T.Compose(transforms)



opts = parse_args()
device = 'cpu'
MALE_CLASSES = ['afro', 'buzz', 'curly', 'classic spikes with fade', 'dreadlocks', 'flattop', 'pompadour', 'ponytail', 'side slick', 'slick back', 'spiky']

# PATH = "weights/mobile_net_best_model.pt"
PATH = "weights/wide_res_best_model.pt"

model = create_model(opts)
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
print(checkpoint.keys())
import torch
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from torch.nn import functional as F
from yaml import parse
from train import parse_args
from model.model import createDeepLabv3


opts = parse_args()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open('WIN_20220109_20_52_35_Pro.jpg')
model = createDeepLabv3(opts)

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out']
output_predictions = output.argmax(0)


out = output_predictions.cpu().numpy() 
print(np.unique(out))
# person_m = np.zeros(out.shape, dtype=np.uint8)
# person_m[out==15] = 200


# cv2.imwrite('test.png', person_m)
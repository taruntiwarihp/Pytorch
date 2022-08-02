from models import EyebrowModel

import torch
from torchvision import transforms

from PIL import Image
import numpy as np
from glob import glob

transformations = transforms.Compose([
	transforms.Resize((16, 32), interpolation=transforms.InterpolationMode.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize([0.5], [0.5])
])

device = torch.device('cpu')

model = EyebrowModel()

ckpt = torch.load('checkpoints/eyebrow_siameseNet_batchnorm50.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])

model.to(device)
model.eval()

# print(ckpt.keys())

ref_img = Image.open('classes/soft/132crop.jpg').convert('L')
# test_img = Image.open('/content/images_background/Alphabet_of_the_Magi/character01/0709_14.png')
# test_img = Image.open('eyebrow_class/flat/73crop.jpg').convert('L')

ref_img_t = transformations(ref_img).unsqueeze(0).to(device)
# test_img_t = transformations(test_img).unsqueeze(0).to(device)

# with torch.no_grad():
#     output = model(ref_img_t, test_img_t)

# print(output)

# all_flat_img = glob('eyebrow_class/round/*.jpg')

# for img_path in all_flat_img:
# 	test_img = Image.open(img_path).convert('L')

# 	test_img_t = transformations(test_img).unsqueeze(0).to(device)

# 	with torch.no_grad():
# 		output = model(ref_img_t, test_img_t)

# 	print(img_path, output)

all_img = glob('classesTEST/*/imagecrop.jpg')

for img_path in all_img:
	test_img = Image.open(img_path).convert('L')

	test_img_t = transformations(test_img).unsqueeze(0).to(device)

	with torch.no_grad():
		output = model(ref_img_t, test_img_t)

	print(img_path.split('/')[1], output)
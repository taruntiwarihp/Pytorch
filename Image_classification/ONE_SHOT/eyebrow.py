import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

class Eyebrowdataset(Dataset):

	def __init__(self, root, setSize, transform=None):

		self.root = root
		self.categories = os.listdir(root)
		self.transform = transform
		self.setSize = setSize

	def __len__(self):
		return self.setSize

	def __getitem__(self, idx):
		img1 = None
		img2 = None
		label = None

		if idx % 2 == 0: # select the same cat for both images
			category = random.choice(self.categories)
			imgDir = os.path.join(self.root, category)
			# img1Name, img2Name = random.sample(os.listdir(imgDir), 2)
			img1Name = random.choice(os.listdir(imgDir))
			img2Name = random.choice(os.listdir(imgDir))
			img1 = Image.open(os.path.join(imgDir, img1Name)).convert('L')
			img2 = Image.open(os.path.join(imgDir, img2Name)).convert('L')
			label = 1.0

		else:
			category1, category2 = random.sample(self.categories, 2)
			img1Dir, img2Dir = os.path.join(self.root, category1), os.path.join(self.root, category2)
			img1Name = random.choice(os.listdir(img1Dir))
			img2Name = random.choice(os.listdir(img2Dir))
			img1 = Image.open(os.path.join(img1Dir, img1Name)).convert('L')
			img2 = Image.open(os.path.join(img2Dir, img2Name)).convert('L')
			label = 0.0
		
		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32)) 

class NWayEvalSet(Dataset):

	def __init__(self, root, setSize, numWay, transform=None):

		self.root = root
		self.categories = os.listdir(root)
		self.transform = transform
		self.setSize = setSize
		self.numWay = numWay

	def __len__(self):
		return self.setSize

	def __getitem__(self, idx):
		category = random.choice(self.categories)
		imgDir = os.path.join(self.root, category)
		imgName = random.choice(os.listdir(imgDir))
		mainImg = Image.open(os.path.join(imgDir, imgName))

		if self.transform:
			mainImg = self.transform(mainImg)
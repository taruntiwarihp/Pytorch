import os
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import optim
import torch.distributed as dist

from dataset import OmniglotDataset, NWayOneShotEvalSet
from eyebrow import Eyebrowdataset
from models import OmniglotModel, EyebrowModel
from utils import save_checkpoint

# os.environ['LOCAL_RANK'] = 4
# os.environ['WORLD_SIZE'] = 1

def main():

	# gpu = 3
	# torch.cuda.set_device(gpu)


	# dist_backend = 'nccl'
	# dist_url = 'env://'
	# world_size = 1
	# rank = 4
	# torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
	# 										world_size=world_size, rank=rank)

	# torch.distributed.barrier()

	root_dir = 'images_background/'

	categories = [[folder, os.listdir(root_dir + folder)] for folder in os.listdir(root_dir)  if not folder.startswith('.') ]
	# print(categories)

	dataSize = 10000 # self-defined dataset size
	TRAIN_PCT = 0.8 # percentage of entire dataset for training
	train_size = int(dataSize * TRAIN_PCT)
	val_size = dataSize - train_size

	transformations = transforms.Compose(
		[transforms.ToTensor()]) 

	omniglotDataset = OmniglotDataset(categories, root_dir, dataSize, transformations)
	train_set, val_set = random_split(omniglotDataset, [train_size, val_size])
	train_loader = DataLoader(train_set, batch_size=256, num_workers=8)
	val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=True)

	# create the test set for final testing
	testSize = 5000 
	numWay = 20
	test_set = NWayOneShotEvalSet(categories, root_dir, testSize, numWay, transformations)
	test_loader = DataLoader(test_set, batch_size = 1, num_workers = 2, shuffle=True)

	device = torch.device('cuda:0')
	model = OmniglotModel()
	model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
	# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3])
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr = 0.0006)

	num_epochs = 50
	criterion = torch.nn.BCEWithLogitsLoss()

	# training and validation after every epoch
	def train(model, train_loader, val_loader, num_epochs, criterion, save_name):
		best_val_loss = float("Inf") 
		train_losses = []
		val_losses = []
		cur_step = 0
		for epoch in range(num_epochs):
			running_loss = 0.0
			model.train()
			print("Starting epoch " + str(epoch+1))
			for img1, img2, labels in train_loader:
				
				# Forward
				img1 = img1.to(device)
				img2 = img2.to(device)
				labels = labels.to(device)
				outputs = model(img1, img2)
				loss = criterion(outputs, labels)
				
				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				running_loss += loss.item()

			avg_train_loss = running_loss / len(train_loader)
			train_losses.append(avg_train_loss)
			
			val_running_loss = 0.0
			with torch.no_grad():
				model.eval()
				for img1, img2, labels in val_loader:
					img1 = img1.to(device)
					img2 = img2.to(device)
					labels = labels.to(device)
					outputs = model(img1, img2)
					loss = criterion(outputs, labels)
					val_running_loss += loss.item()
			avg_val_loss = val_running_loss / len(val_loader)
			val_losses.append(avg_val_loss)
			
			print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
				.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				save_checkpoint(save_name, model, optimizer, best_val_loss)
		
		print("Finished Training")  
		return train_losses, val_losses  

	# evaluation metrics
	def eval(model, test_loader):
		with torch.no_grad():
			model.eval()
			correct = 0
			print('Starting Iteration')
			count = 0
			for mainImg, imgSets, label in test_loader:
				mainImg = mainImg.to(device)
				predVal = 0
				pred = -1
				for i, testImg in enumerate(imgSets):
					testImg = testImg.to(device)
					output = model(mainImg, testImg)
					if output > predVal:
						pred = i
						predVal = output
				label = label.to(device)
				if pred == label:
					correct += 1
				count += 1
				if count % 20 == 0:
					print("Current Count is: {}".format(count))
					print('Accuracy on n way: {}'.format(correct/count))

	save_path = os.path.join('checkpoints','siameseNet-batchnorm50.pt')
	train_losses, val_losses = train(model, train_loader, val_loader, num_epochs, criterion, save_path)


	# Figure
	x = np.arange(50)
	fig, axs = plt.subplots(2)
	fig.suptitle('Loss Plot')
	axs[0].plot(x, train_losses)
	axs[1].plot(x, val_losses)

	fig.savefig('full_figure.png')

def main_eyebrow():

	root_dir = 'eyebrow_class/'

	# categories = [[folder, os.listdir(root_dir + folder)] for folder in os.listdir(root_dir)  if not folder.startswith('.') ]
	# print(categories)
	# categories = os.listdir(root_dir)

	dataSize = 10000 # self-defined dataset size
	TRAIN_PCT = 0.8 # percentage of entire dataset for training
	train_size = int(dataSize * TRAIN_PCT)
	val_size = dataSize - train_size

	transformations = transforms.Compose([
		transforms.Resize((16, 32), interpolation=transforms.InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])


	omniglotDataset = Eyebrowdataset(root_dir, dataSize, transformations)
	train_set, val_set = random_split(omniglotDataset, [train_size, val_size])
	train_loader = DataLoader(train_set, batch_size=32, num_workers=8)
	val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=True)

	# create the test set for final testing
	# testSize = 5000 
	# numWay = 20
	# test_set = NWayOneShotEvalSet(categories, root_dir, testSize, numWay, transformations)
	# test_loader = DataLoader(test_set, batch_size = 1, num_workers = 2, shuffle=True)

	device = torch.device('cuda:0')
	model = EyebrowModel()
	model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
	# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3])
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr = 0.0006)

	num_epochs = 100
	criterion = torch.nn.BCEWithLogitsLoss()

	# training and validation after every epoch
	def train(model, train_loader, val_loader, num_epochs, criterion, save_name):
		best_val_loss = float("Inf") 
		train_losses = []
		val_losses = []
		cur_step = 0
		for epoch in range(num_epochs):
			running_loss = 0.0
			model.train()
			# print("Starting epoch " + str(epoch+1))
			for img1, img2, labels in tqdm(train_loader):
				
				# Forward
				img1 = img1.to(device)
				img2 = img2.to(device)
				labels = labels.to(device)
				outputs = model(img1, img2)
				loss = criterion(outputs, labels)
				
				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				running_loss += loss.item()

			avg_train_loss = running_loss / len(train_loader)
			train_losses.append(avg_train_loss)
			
			val_running_loss = 0.0
			with torch.no_grad():
				model.eval()
				for img1, img2, labels in tqdm(val_loader):
					img1 = img1.to(device)
					img2 = img2.to(device)
					labels = labels.to(device)
					outputs = model(img1, img2)
					loss = criterion(outputs, labels)
					val_running_loss += loss.item()
			avg_val_loss = val_running_loss / len(val_loader)
			val_losses.append(avg_val_loss)
			
			print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
				.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				save_checkpoint(save_name, model, optimizer, best_val_loss)
		
		print("Finished Training")  
		return train_losses, val_losses  

	# evaluation metrics
	def eval(model, test_loader):
		with torch.no_grad():
			model.eval()
			correct = 0
			print('Starting Iteration')
			count = 0
			for mainImg, imgSets, label in test_loader:
				mainImg = mainImg.to(device)
				predVal = 0
				pred = -1
				for i, testImg in enumerate(imgSets):
					testImg = testImg.to(device)
					output = model(mainImg, testImg)
					if output > predVal:
						pred = i
						predVal = output
				label = label.to(device)
				if pred == label:
					correct += 1
				count += 1
				if count % 20 == 0:
					print("Current Count is: {}".format(count))
					print('Accuracy on n way: {}'.format(correct/count))

	save_path = os.path.join('checkpoints','eyebrow_siameseNet_batchnorm50_new.pt')
	train_losses, val_losses = train(model, train_loader, val_loader, num_epochs, criterion, save_path)


	# Figure
	x = np.arange(num_epochs)
	fig, axs = plt.subplots(2)
	fig.suptitle('Loss Plot')
	axs[0].plot(x, train_losses)
	axs[1].plot(x, val_losses)

	fig.savefig('full_figure_eyebrow_new.png')

if __name__ == '__main__':
	# main()
	main_eyebrow()
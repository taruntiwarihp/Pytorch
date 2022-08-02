import torch
import os
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import transforms
from PIL import Image



MALE_CLASSES = ['afro', 'buzz', 'curly', 'classic spikes with fade', 'dreadlocks', 'flattop', 'pompadour', 'ponytail', 'side slick', 'slick back', 'spiky']
FEMALE_CLASSES = ['afro', 'bob', 'braid', 'bun', 'buzz', 'curly', 'fringe', 'dreadlocks', 'pompadour', 'ponytail', 'slick back', 'spiky', 'straight', 'wavy']

class HairDataset(Dataset):

    def __init__(self, root_dir, transform=None, mode='male'):

        if mode == 'male':
            CLASSES = MALE_CLASSES
        else:
            CLASSES = FEMALE_CLASSES
        
        self.valid_img_files = []
        self.valid_labels = []

        for class_index, class_name in enumerate(CLASSES):
            all_files = glob(os.path.join(root_dir, class_name, '*.jpg'))
            all_labels = [class_index] * len(all_files)

            self.valid_img_files.extend(all_files)
            self.valid_labels.extend(all_labels)
        
            
        self.transform = transform

    def __len__(self):
        return len(self.valid_img_files)

    def __getitem__(self, idx):
        img = Image.open(self.valid_img_files[idx])

        if self.transform is not None:
            img = self.transform(img)

        label = self.valid_labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label, idx

def get_transform(train, opts):
	transforms = []
	transforms.append(T.Resize((224, 224)))
	transforms.append(T.ToTensor())
	transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))

	return T.Compose(transforms)
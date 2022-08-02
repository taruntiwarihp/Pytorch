from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torchvision.transforms.functional as TF
import random
from glob import glob 
import os

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


class SegmentationDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(SegmentationDataset, self).__init__()

        self.images = sorted(glob(os.path.join(root, 'images/*')))
        self.masks = sorted(glob(os.path.join(root, 'masks/*')))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        image_np = np.asarray(img)
        mask_np = np.asarray(mask)
        new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
        image_and_label_np = np.zeros(new_shape, image_np.dtype)
        image_and_label_np[:, :, 0:3] = image_np
        image_and_label_np[:, :, 3] = mask_np

        # Convert to PIL
        image_and_label = Image.fromarray(image_and_label_np)

        image_and_label = self.transforms(image_and_label)

        # Extract image and label
        img = image_and_label[0:3, :, :]
        mask = image_and_label[3, :, :].unsqueeze(0)

        # Normalize back from [0, 1] to [0, 255]
        mask = mask * 255
        #  Convert to int64 and remove second dimension
        # mask = mask.long().squeeze()
        return img, mask


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize((224, 224)))
    transforms.append(T.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1]))

    if train:

        transforms.append(RandomRotation())

    return T.Compose(transforms)

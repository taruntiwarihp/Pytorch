from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms as T

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer
from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)

train_transform = T.Compose([
    RandomRotation(),
    T.Lambda(under_max),
    T.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = T.Compose([
    T.Lambda(under_max),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='train'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id']), val['caption']) for val in ann['annotations']]

        if mode == 'val':
            self.annot = self.annot
        if mode == 'train':
            self.annot = self.annot[: limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)

        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
             caption, max_length=self.max_length,
             pad_to_max_length=True, return_attention_mask=True, 
             return_token_type_ids=False, truncation=True
        )

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])
        ).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask

def prepare_dataset(opts, mode='train'):
    if mode == 'train':
        train_dir = os.path.join(opts.dir, 'train2017')
        train_file = os.path.join(
            opts.dir, 'annotations', 'captions_train2017.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=opts.max_position_embeddings, limit=opts.limit, transform=train_transform, mode='train')
        
        return data
        
    elif mode == 'val':
        val_dir = os.path.join(opts.dir, 'val2017')
        val_file = os.path.join(opts.dir, 'annotations', 'captions_val2017.json')

        data = CocoCaption(val_dir, read_json(
            val_file), max_length=opts.max_position_embeddings, limit=opts.limit, transform=val_transform, mode='val')
        
        return data

    else:
        raise ValueError('{} Not supported yet'.format(mode))
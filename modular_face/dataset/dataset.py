from torch.utils.data import Dataset
import torch

import os
from PIL import Image
from glob import glob

class MultiFeatureFaceData(Dataset):

    hair_classes = [
        '__ignore__', 'afro', 'bald', 'bob', 'braid', 'bun', 'buzz', 'classicspike', 'curly', 'dreadlocks', 
        'flattop', 'fringe', 'pompadour', 'ponytail', 'sidepart', 'slickback', 'spiky', 'straight', 'wavy'
    ]

    mustache_classes = [
        '__ignore__', 'chevron', 'clean', 'english', 'fumanchu', 'handlebar', 'horseshoe', 'pencil', 'walrus'
    ]

    beard_classes = [
        '__ignore__', 'anchor', 'bandholz', 'circlebeard', 'clean', 'ducktail', 'dutch', 'goatee', 'muttonchops', 'stubble'
    ]

    def __init__(self, root='data/scrapped_filtered', transform=None):
        
        self.transform = transform
        _class = sorted(os.listdir(root))

        self.all_valid_data = []

        for cls in _class:
            
            if cls.split('_')[0] in self.hair_classes:
                for f in os.listdir(os.path.join(root, cls)):
                    base_dict = {}
                    base_dict['img_file'] = os.path.join(root, cls, f)
                    base_dict['hair'] = self.hair_classes.index(cls.split('_')[0])
                    base_dict['beard'] = 0
                    base_dict['mustache'] = 0

                    self.all_valid_data.append(base_dict)
            
            elif cls.split('_')[0] in self.beard_classes:
                for f in os.listdir(os.path.join(root, cls)):
                    base_dict = {}
                    base_dict['img_file'] = os.path.join(root, cls, f)
                    base_dict['beard'] = self.beard_classes.index(cls.split('_')[0])
                    base_dict['hair'] = 0
                    base_dict['mustache'] = 0

                    self.all_valid_data.append(base_dict)

            elif cls.split('_')[0] in self.mustache_classes:
                for f in os.listdir(os.path.join(root, cls)):
                    base_dict = {}
                    base_dict['img_file'] = os.path.join(root, cls, f)
                    base_dict['mustache'] = self.mustache_classes.index(cls.split('_')[0])
                    base_dict['hair'] = 0
                    base_dict['beard'] = 0

                    self.all_valid_data.append(base_dict)

            else:
                raise ValueError('Not Valid')

    def __len__(self):
        return len(self.all_valid_data)

    def __getitem__(self, idx):

        base_dict = self.all_valid_data[idx]

        img = Image.open(base_dict['img_file']).convert('RGB')
        img = self.transform(img)

        sample = {
            'image': img,
            'label': {
                'hair' : base_dict['hair'],
                'beard' : base_dict['beard'],
                'mustache' : base_dict['mustache'],
            }
        }

        return sample

class SingleFeatureDataset(Dataset):
    hair_classes = [
        'afro', 'bald', 'bob', 'braid', 'bun', 'buzz', 'classicspike', 'curly', 'dreadlocks', 
        'flattop', 'fringe', 'pompadour', 'ponytail', 'sidepart', 'slickback', 'spiky', 'straight', 'wavy'
    ]

    mustache_classes = [
        'chevron', 'clean', 'english', 'fumanchu', 'handlebar', 'horseshoe', 'pencil', 'walrus'
    ]

    beard_classes = [
        'anchor', 'bandholz', 'circlebeard', 'clean', 'ducktail', 'dutch', 'goatee', 'muttonchops', 'stubble'
    ]

    def __init__(self, mode='hair', root='data/scrapped_filtered', transform=None):

        self.transform = transform
        all_base_classes = os.listdir(root)
        if mode == 'hair':
            mode_class = self.hair_classes
            all_valid_classes = [f for f in all_base_classes if f.split('_')[0] in mode_class]

        elif mode == 'beard':
            mode_class = self.beard_classes
            all_valid_classes = [f for f in all_base_classes if f.split('_')[0] in mode_class]

        elif mode == 'mustache':
            mode_class = self.mustache_classes
            all_valid_classes = [f for f in all_base_classes if f.split('_')[0] in mode_class]

        else:
            raise ValueError('Not Valid')

        self.valid_img_files = []
        self.valid_labels = []

        for _, class_name in enumerate(all_valid_classes):
            cls = class_name.split('_')[0]
            class_index = mode_class.index(cls)
            all_files = glob(os.path.join(root, class_name, '*.jpg'))
            all_labels = [class_index] * len(all_files)

            self.valid_img_files.extend(all_files)
            self.valid_labels.extend(all_labels)
            
    def __len__(self):
        return len(self.valid_img_files)

    def __getitem__(self, idx):
        img = Image.open(self.valid_img_files[idx])
        img = self.transform(img)

        label = self.valid_labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return img, label
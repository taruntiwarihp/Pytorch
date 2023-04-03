# from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# from torchvision.models.segmentation import fcn_resnet50
# from datasets.mango import build_transform
# from models import BaseFeatureExtractor
# import torch
# from PIL import Image

# model = BaseFeatureExtractor('efficientnet_v2_l', n_class=5)
# model_path = 'weights/efficientnet_v2_l/best_model.pt'

# ckpt = torch.load(model_path)

# model.load_state_dict(ckpt['model_dict'])

# model.backbone.classifier = torch.nn.Identity()

# model.eval()


# trans = build_transform(input_size=224, is_train=True)
# img = Image.open('data/Mango/Anthracnose/1.jpg').convert('RGB')
# img = trans(img).unsqueeze(0)

# print(img.shape)

# features = model(img)

# print(features.shape)

# agg_model = fcn_resnet50(pretrained=True)

# print(agg_model.classifier)
from glob import glob

im = sorted(glob("data/celeb_classification/*"))
print(len(im))
# # # from model import *
# # # from model.utils import MLPBlock
# # # from collections import OrderedDict

# # # import torch

# # # base_model = BaseFeatureExtractor(n_class = 45)
# # # base_model.load_state_dict(torch.load('weights/efficientnet_v2_l_best_model.pt', map_location='cpu')['model'])
# # # base_model = torch.nn.Sequential(*list(base_model.backbone.children())[:-2])

# # # class Classifier(torch.nn.Module):

# # #     def __init__(self, n_class):
# # #         super().__init__()

# # #         self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
# # #         self.mlp = MLPBlock(in_dim=1280, mlp_dim=1024, out_dim=n_class, dropout=0.25)

# # #     def forward(self, features):
# # #         features = self.avgpool(features)
# # #         features = torch.flatten(features, 1)
# # #         prob = self.mlp(features)

# # #         return prob

# # # hair_ckpt = torch.load('weights/efficientnet_v2_l_hair_best_model.pt')['model']
# # # hair_model = Classifier(18)

# # # new_state_dict = OrderedDict()
# # # for key in hair_model.state_dict().keys():
# # #     new_state_dict[key] = hair_ckpt[key]

# # # print(hair_model.load_state_dict(new_state_dict))

# # from asyncio import base_futures
# # from model.utils import CLassifierProb
# # from torchvision import transforms as T
# # from PIL import Image
# # from model import BaseFeatureExtractor
# # import torch
# # from model.utils import Classifier
# # from collections import OrderedDict
# # import time

# # trans = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# # base_model = BaseFeatureExtractor(n_class = 45)
# # base_model.load_state_dict(torch.load('weights/efficientnet_v2_l_best_model.pt', map_location='cpu')['model'])
# # base_model = torch.nn.Sequential(*list(base_model.backbone.children())[:-2]).cuda()
# # base_model.eval()

# # cls = CLassifierProb(
# #     base_model = base_model,
# #     hair_ckpt_path = 'weights/efficientnet_v2_l_hair_best_model.pt',
# #     beard_ckpt_path = 'weights/efficientnet_v2_l_beard_best_model.pt',
# #     mustache_ckpt_path = 'weights/efficientnet_v2_l_mustache_best_model.pt',
# #     hair_classes = 18, 
# #     beard_classes = 9, 
# #     mustache_classes = 8, 
# #     trans_fn = trans,
# # )

# # img_file = 'data/scrapped_filtered/afro_female_hair/0a13d66e8d.jpg'
# # img = Image.open(img_file).convert('RGB')
# # # img = trans(img).unsqueeze(0).cuda()
# # tic = time.time()
# # print(cls(img))
# # print(time.time() - tic)
# # # ckpt_path = 'weights/efficientnet_v2_l_hair_best_model.pt'
# # # classes = 18
# # # ckpt = torch.load(ckpt_path, map_location='cpu')['model']
# # # model = Classifier(classes)
# # # new_state_dict = OrderedDict()
# # # print('adchidhfusgvhbcsdfvbhjsdhcvj')
# # # for key in model.state_dict().keys():
# # #     new_state_dict[key] = ckpt[key]

# # # model.load_state_dict(new_state_dict)
# # # model = model.cuda()
# # # model.eval()

# # # base_feature = base_model(img)
# # # # print(base_futures.shape)
# # # prob = model(base_feature)

# # # print(prob.shape)

# # # _, predicted = torch.max(prob.data, 1)

# # # print()


# from utils.face68_class import FacePoints
# import dlib
# import cv2
# import numpy as np
# from matplotlib.path import Path
# from utils.dlib_handler import DlibHandler
# import time

# LOWER_LIP_INDICES = [48] + [i for i in range(59, 53, -1)] + [i for i in range(64, 68, 1)] + [60, 48]

# def region_avg(img, face_68_points, indices):

    
#     vertices = face_68_points[indices]
#     # print(vertices)
#     path = Path(vertices)

#     x, y = np.mgrid[:img.shape[1], :img.shape[0]]
#     points = np.vstack((x.ravel(), y.ravel())).T
#     mask = path.contains_points(points)

#     # reshape mask for display
#     img_mask = mask.reshape(x.shape).T
#     avg_pixel = np.count_nonzero(img_mask)
#     region_only = img.copy()

#     region_only[img_mask==False] = 0
#     avg_color = [np.sum(region_only[:,:,a]) for a in range(3)]
#     avg_color = [int(t/avg_pixel) for t in avg_color]
#     avg_color_rgb = avg_color[::-1]

#     return '%02x%02x%02x' % tuple(avg_color_rgb)

# def region_avg_(img, vertices):

    
#     # vertices = face_68_points[indices]
#     # print(vertices)
#     path = Path(vertices)

#     x, y = np.mgrid[:img.shape[1], :img.shape[0]]
#     points = np.vstack((x.ravel(), y.ravel())).T
#     mask = path.contains_points(points)

#     # reshape mask for display
#     img_mask = mask.reshape(x.shape).T
#     avg_pixel = np.count_nonzero(img_mask)
#     region_only = img.copy()

#     region_only[img_mask==False] = 0
#     avg_color = [np.sum(region_only[:,:,a]) for a in range(3)]
#     avg_color = [int(t/avg_pixel) for t in avg_color]
#     avg_color_rgb = avg_color[::-1]

#     return '%02x%02x%02x' % tuple(avg_color_rgb)

# tic = time.time()
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')
# face_points = FacePoints(detector, predictor)
# dh = DlibHandler(detector, predictor)
# print()
# # print('Loading time is ', time.time()- tic)

# img_path = 'data/scrapped_filtered/afro_female_hair/0cfe175ba2.jpg'
# # img_path = 'hills-2836301_1920.jpg'
# # img_path = 'How-Give-Kindle-book-multiple-recipients.jpg'
# # img_bgr = cv2.imread(img_path)
# # gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# # rects = detector(gray, 1)
# # rect = [r for r in rects][0]
# # shape = predictor(gray, rect)
# # shape_np = np.zeros((68, 2), dtype="int")
# # for i in range(0, 68):
# #     shape_np[i] = (shape.part(i).x, shape.part(i).y)

# # lip_color = region_avg(img_bgr, shape_np, LOWER_LIP_INDICES)

# # print('#{}'.format(lip_color))

# # # print(LOWER_LIP_INDICES)

# print(face_points.get_points(img_path))
# # # print(face_points.lips)

# # vertices = [face_points.lips['lip_{}'.format(i)] for i in [48] + [i for i in range(59, 53, -1)]] + [face_points.mouths['mouth_{}'.format(i)] for i in [i for i in range(64, 68, 1)]] + [face_points.mouths['mouth_60'], face_points.lips['lip_48']]
# # # print(vertices)

# # c = region_avg_(img_bgr, vertices)
# # print('#{}'.format(c))
# # tic = time.time()
# # ff = dh.handle_face_features(img_path)

# # print(ff)

# # print('Time ', time.time()- tic)

import requests
import time
from PIL import Image
import io

img_path = 'data/scrapped_filtered/afro_female_hair/0cfe175ba2.jpg'
img_path = 'How-Give-Kindle-book-multiple-recipients.jpg'
pil_image = Image.open(img_path).convert('RGB')
buf = io.BytesIO()
pil_image.save(buf, format='JPEG')
img_bytes = buf.getvalue()

tic = time.time()
# files=[
#   ('file',('file.jpg', open('data/scrapped_filtered/afro_female_hair/0cfe175ba2.jpg', 'rb'), 'image/jpeg'))
# ]
files = [('file', img_bytes)]
payload = {}
headers = {}

url = "http://34.123.222.223:5400/dlib"

# response = requests.request("POST", url, headers=headers, data=payload, files=files)

# print(response.text)

# print(time.time()-tic)
tic = time.time()

# files=[
#   ('file',('file.jpg', open('data/scrapped_filtered/afro_female_hair/0cfe175ba2.jpg', 'rb'), 'image/jpeg'))
# ]

files = [('file', img_bytes)]
url = "http://34.123.222.223:5500/faceParse"

# response = requests.request("POST", url, headers=headers, data=payload, files=files)

# print(response.text)
# print(time.time()-tic)

payload={'gender': 'female'}
url = "http://34.123.222.223:5800/modularFace"

response = requests.request("POST", url, headers=headers, data=payload, files=files)
print(response.text)
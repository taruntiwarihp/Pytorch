import torch
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from torch.nn import functional as F
from base import VideoInference

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# print(input_tensor.shape)
# print(input_image.size)
# print(input_batch.shape)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)


# create a color pallette, selecting a color for each class
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
# r.putpalette(colors)
# image = cv2.imread(filename)
# print(image.shape)
# out = output_predictions.cpu().numpy() 
# person_m = np.zeros(out.shape, dtype=np.uint8)
# person_m[out==15] = 200
# mask = np.expand_dims(person_m, axis=2)
# mask = np.tile(mask, (1,1,3))
# mask = mask.astype(np.uint8)
# image_alpha = cv2.add(image, mask)
# cv2.imwrite('img.png', image_alpha)

# print(np.unique(out))

model.to('cuda')

def get_human(image):
    with torch.no_grad():
        # input_batch = image.unsqueeze(0)
        output = model(image)['out'][0]

    output_predictions = output.argmax(0)
    out = output_predictions.byte().cpu().numpy() #.reshape((720, 1280, 3))
    # print(out.shape)
    person_m = np.zeros(out.shape, dtype=np.uint8)
    person_m[out==15] = 200

    return person_m
        
inference = VideoInference(
    model=get_human,
    video_path='test.mp4',
    input_size=320,
    use_cuda=True,
    draw_mode='matting',
)
inference.run()
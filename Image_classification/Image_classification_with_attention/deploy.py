import io
import json
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
from flask import Flask, jsonify, request
from models.atten_classifier import AttentionClassifier


app = Flask(__name__)
MALE_CLASSES = ['afro', 'buzz', 'curly', 'classic spikes with fade', 'dreadlocks', 'flattop', 'pompadour', 'ponytail', 'side slick', 'slick back', 'spiky']
# FEMALE_CLASSES = ['afro', 'bob', 'braid', 'bun', 'buzz', 'curly', 'fringe', 'dreadlocks', 'pompadour', 'ponytail', 'slick back', 'spiky', 'straight', 'wavy']

device='cpu'

model = AttentionClassifier('resnet')

checkpoint = torch.load('weights/best_model_male_resnet.pt',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])

model.to(device).eval()


def transform_image(image_bytes):

    my_transforms = T.Compose([T.Resize((224, 224)),
                               T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])

    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    atten, outputs = model(tensor.to(device))
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return MALE_CLASSES[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return class_name

if __name__ == '__main__':
    app.run(port=8080)



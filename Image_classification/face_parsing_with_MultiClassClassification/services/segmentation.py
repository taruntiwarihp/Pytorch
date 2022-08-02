from flask import Flask, jsonify, request
import numpy as np
import cv2
from PIL import Image
import io
from torchvision import transforms as T
import sys
import torch

sys.path.append('/home/bigthinx/modular_face/')
from utils.face_parsing import color_extraction

from model.bisenet import BiSeNet

app = Flask(__name__)

face_parsing_model = BiSeNet(n_classes=19)
fp_ckpt_path = 'weights/face_parsing_BiSeNet.pth'
ckpt = torch.load(fp_ckpt_path)
face_parsing_model.load_state_dict(ckpt)
face_parsing_model.to('cuda:1')
face_parsing_model.eval()

img_dim=(512, 512)

face_parsing_trans_fn = T.Compose([
    T.Resize(img_dim),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/faceParse', methods=['POST'])
def predict_parse():
    if request.method == 'POST':
        imagefile = request.files.get('file', False)
        img_pil = Image.open(imagefile).convert('RGB')
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        img_bytes = buf.getvalue()

        temp = color_extraction(img_bytes, face_parsing_model, face_parsing_trans_fn)
        return jsonify(temp)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5500, threaded=True)
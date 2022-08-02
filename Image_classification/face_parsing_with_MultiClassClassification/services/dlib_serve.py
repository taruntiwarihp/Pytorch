import dlib
from flask import Flask, jsonify, request
import sys
import io
from PIL import Image

import torchvision.transforms as T
import os
import torch

sys.path.append('/home/bigthinx/modular_face/')
from model.eyebrowResnet import EyebrowResnet18
from utils.dlib_handler import DlibHandler
from utils.one_shot import  eyebrow_roi, eyebrow_get_max_feature

def get_eyebrow_transform():
    transformations = []
    transformations.append(T.Resize((84, 256), interpolation=T.InterpolationMode.BICUBIC))
    transformations.append(T.ToTensor())
    transformations.append(T.Normalize([0.5], [0.5]))

    return T.Compose(transformations)

app = Flask(__name__)

eye_brow_model = EyebrowResnet18()
ckpt_path = 'weights/eyebrow_resnet18_high.pt'
ckpt = torch.load(ckpt_path)
eye_brow_model.load_state_dict(ckpt['model_state_dict'])
eye_brow_model.to('cuda:1')
eye_brow_model.eval()

eyebrow_trans_fn = get_eyebrow_transform()
eyebrow_ref_root = 'ref_img'
eyebrow_classes = [f.split('.')[0] for f in sorted(os.listdir(eyebrow_ref_root))]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')

DH = DlibHandler(detector, predictor)

ref_eyebrow_imgs = []
# path = '{}/*.jpg'.format(self.eyebrow_ref_img)
for f in eyebrow_classes:

    f = os.path.join(eyebrow_ref_root, f + '.jpg')

    temp = Image.open(f).convert('L')
    temp_t = eyebrow_trans_fn(temp).unsqueeze(0)
    ref_eyebrow_imgs.append(temp_t)

ref_eyebrow_imgs = torch.cat(ref_eyebrow_imgs, dim=0).to('cuda:1')

@app.route('/dlib', methods=['POST'])
def predict():
    if request.method == 'POST':
        imagefile = request.files.get('file', False)
        img_pil = Image.open(imagefile).convert('RGB')
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        img_bytes = buf.getvalue()

        temp = DH.handle_face_features(img_bytes)

        if isinstance(temp, str):
            return jsonify({'error': temp})

        eyebrow_class = eyebrow_get_max_feature(
            model=eye_brow_model,
            trans_fn=eyebrow_trans_fn,
            img_=eyebrow_roi(detector, predictor, img_bytes),
            ref_batch = ref_eyebrow_imgs,
            cats = eyebrow_classes
        )

        temp['eyebrow'] = eyebrow_class

        return jsonify(temp)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5400, threaded=True)
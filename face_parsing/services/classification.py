from flask import Flask, jsonify, request
from PIL import Image
import io
from torchvision import transforms as T
import sys
import torch

sys.path.append('/home/bigthinx/modular_face/')

from model.utils import CLassifierProb
from model import BaseFeatureExtractor, HairFeature

app = Flask(__name__)

base_ckpt_path = 'weights/efficientnet_v2_l_best_model.pt'
hair_ckpt_path = 'weights/efficientnet_v2_l_hair_best_model.pt'
beard_ckpt_path = 'weights/efficientnet_v2_l_beard_best_model.pt'
mustache_ckpt_path = 'weights/efficientnet_v2_l_mustache_best_model.pt'

hair_classes = [
    'Afro', 'Bald', 'Bob', 'Braid', 'Bun', 'Buzz', 'ClassicSpike', 'Curly', 'Dreadlocks', 
    'FlatTop', 'Fringe', 'Pompadour', 'Ponytail', 'SidePart', 'Slickback', 'Spiky', 'Straight', 'Wavy'
]
beard_classes = [
    'Anchor', 'Bandholz', 'CircleBeard', 'Clean', 'Ducktail', 'Dutch', 'Goatee', 'MuttonChops', 'Stubble'
]
mustache_classes = [
    'Chevron', 'Clean', 'English', 'FuManchu', 'Handlebar', 'Horseshoe', 'Pencil', 'Walrus'
]


base_model = BaseFeatureExtractor(n_class = 45)
base_model.load_state_dict(torch.load(base_ckpt_path, map_location='cpu')['model'])
base_model = torch.nn.Sequential(*list(base_model.backbone.children())[:-2])

model = HairFeature(base_model).to('cuda:1')

model.eval()
trans_fn = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

classifier = CLassifierProb(
    base_model=base_model,
    hair_ckpt_path=hair_ckpt_path,
    beard_ckpt_path=beard_ckpt_path,
    mustache_ckpt_path=mustache_ckpt_path,
    hair_classes=hair_classes,
    beard_classes=beard_classes,
    mustache_classes=mustache_classes,
    trans_fn=trans_fn
)


@app.route('/hair', methods=['POST'])
def predict_parse():
    if request.method == 'POST':
        gender = request.form.get('gender', False) 
        imagefile = request.files.get('file', False)
        img_pil = Image.open(imagefile).convert('RGB')
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        img_bytes = buf.getvalue()

        temp = classifier(img_bytes, gender)

        # temp = {
        #     'gender': gender,
        #     'hair_class': hair_class,
        #     'beard_class': beard_class,
        #     'mustache_class': mustache_class
        # }
        return jsonify(temp)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5300, threaded=True)
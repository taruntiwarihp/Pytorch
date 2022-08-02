from flask import Flask, jsonify, request
from PIL import Image
import io
from torchvision import transforms as T
import os
import torch
import requests
import json
import pprint
from utils.utils import null_json
from utils.get_logger import create_logger
# from queue import Queue
# from threading import Thread

# create logs
logDir = 'logs'
os.makedirs(logDir, exist_ok=True)

logger = create_logger()

logger.info("Creating Flask Application")
app = Flask(__name__)

logger.info(" main client initiated")

# que = Queue() 
# threads_list = []


@app.route('/modularFace', methods=['POST'])
def predict_parse():
    if request.method == 'POST':
        gender = request.form.get('gender', False) 
        imagefile = request.files.get('file', False)

        logger.info('Got a new request with gender {}'.format(gender))
        img_pil = Image.open(imagefile).convert('RGB')
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        img_bytes = buf.getvalue()

        results = null_json()

        # Dlib
        files = [('file', img_bytes)]
        payload = {}
        headers = {}

        logger.info('Sending request to Dlib Server http://34.68.5.165:5400/dlib')
        url = "http://34.68.5.165:5400/dlib"
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        
        # print(response.text)
        result = json.loads(response.text)
        logger.info('Received results from Dlib')
        logger.info(pprint.pformat(result))
        
        if 'error' in result.keys():
            results['error']['message'] = result['error']
            logger.error(result['error'])
            return jsonify(results)

        results['eye']['style'] = result['eye_feature']
        results['eye']['color'] = result['eye_color']

        results['eyebrow']['style'] = result['eyebrow']

        results['face']['style'] = result['face_color']
        results['face']['color'] = result['face_shape_feature']

        results['lip']['style'] = result['lip_feature']
        results['lip']['color'] = result['lip_color']

        results['nose']['style'] = result['nose_feature']
        results['nose']['color'] = result['face_color']

        # Segmentation
        files = [('file', img_bytes)]
        payload = {}
        headers = {}

        logger.info('Sending request to faceParse Server http://34.68.5.165:5500/faceParse')
        url = "http://34.68.5.165:5500/faceParse"
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        logger.info('Received results from Face Parse')
        result = json.loads(response.text)
        logger.info(pprint.pformat(result))

        results['eyebrow']['color'] = result['eyebrow_color']
        results['hair']['color'] = result['hair_color']
        results['facialHair']['color'] = result['hair_color']

        # CLassification
        files = [('file', img_bytes)]
        payload={'gender': gender}
        headers = {}

        logger.info('Sending request to Classification Server http://34.68.5.165:5300/hair')
        url = "http://34.68.5.165:5300/hair"
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        # print(response.text)
        logger.info('Received results from Classification')

        result = json.loads(response.text)
        logger.info(pprint.pformat(result))

        results['hair']['style'] = result['hair_class']
        results['facialHair']['beardStyle'] = result['beard_class']
        results['facialHair']['moustacheStyle'] = result['mustache_class']
        logger.info('Sending Final result')
        logger.info(pprint.pformat(results))
        return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5800, threaded=True)
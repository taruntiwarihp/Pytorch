from PIL import Image
import torch
import cv2
import numpy as np
from imutils import face_utils
import io

def eyebrow_roi(detector, predictor, img_path):
    if isinstance(img_path, str):
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = Image.open(io.BytesIO(img_path)).convert('RGB')
        frame = np.array(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame_copy = frame.copy()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = detector(gray)[0]
    shape = predictor(gray,rect)
    point_array = face_utils.shape_to_np(shape)
    x_max, x_min, y_max, y_min = max(point_array[:,0]), min(point_array[:,0]), max(point_array[:,1]), min(point_array[:,1])
    WIDTH = abs(x_max - x_min)
    HIGHT = abs(y_max - y_min)
    a = WIDTH//38
    b = HIGHT//38
    (x, y, w, h) = cv2.boundingRect(np.array([point_array[17:22]]))
    roi_eyebrow = frame_copy[y-b:y+h+b, x-a:x+w+a]
    roi_eyebrow = cv2.cvtColor(roi_eyebrow, cv2.COLOR_BGR2RGB)

    return roi_eyebrow

def eyebrow_get_max_feature(model, trans_fn, img_, ref_batch, cats):
    
    img = Image.fromarray(img_).convert('L') 
    img_t = trans_fn(img).unsqueeze(0)
    test_batch = torch.cat([img_t] * 10, dim=0).to('cuda:1')
    with torch.no_grad():
        output = model(test_batch, ref_batch).to('cuda:1')

    catagory = cats[output.argmax(0)]

    # Name Editing
    cat_map = {
        'roundLowArch': 'RoundLowArch',
        'hard' : 'SoftAngledHighArch',
        'flat' : 'Flat',
        'softLowArch' : 'SoftAngledLowArch',
        'softHighArch' : 'SoftAngledHighArch',
        'roundMidArch' : 'RoundLowArch',
        'sShape': 'SShaped',
        'roundHighArch' : 'RoundHighArch',
        'straight': 'Straight',
        'softMidArch': 'SoftAngledLowArch',
    }

    return cat_map[catagory]
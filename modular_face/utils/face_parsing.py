import cv2
from PIL import Image
import io
import numpy as np
import torch

def color_extraction(img_path, face_parsing_model, trans_fn):

    l_eyebrow_idx = 2
    r_eyebrow_idx = 3
    hair_idx = 17
    result = {}

    if isinstance(img_path, str):
        img_bgr = cv2.imread(img_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        image = Image.open(img_path).convert('RGB')
    else:
        image = Image.open(io.BytesIO(img_path)).convert('RGB')
        # image = np.array(image)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # trans_fn = get_general_transform(img_dim=(512, 512))

    img_t = trans_fn(image).unsqueeze(0).to('cuda:1')
    img_512 = image.resize((512, 512), Image.BILINEAR)
    img_512_np = np.array(img_512)

    with torch.no_grad():
        output = face_parsing_model(img_t)[0]
        mask = output.squeeze(0).cpu().numpy().argmax(0)

    all_possible_classes = np.unique(mask)
    if ((l_eyebrow_idx in all_possible_classes) or (r_eyebrow_idx in all_possible_classes)):

        # eyebrow avg color
        debug_img = img_512_np.copy()
        debug_mask = mask.copy()

        try:

            debug_img[debug_mask!=l_eyebrow_idx] = 0
            all_pixel = np.count_nonzero(np.array(debug_mask==l_eyebrow_idx, dtype=np.uint8))
            avg_color_rgb = [np.sum(debug_img[:,:,a]) for a in range(3)] # [::-1]
            avg_color_rgb = [int(t/all_pixel) for t in avg_color_rgb]

            eyebrow_color = tuple(avg_color_rgb)
            eyebrow_color = '%02x%02x%02x' % tuple(avg_color_rgb)

        except:
            eyebrow_idx = 3
            debug_img[debug_mask!=eyebrow_idx] = 0
            all_pixel = np.count_nonzero(np.array(debug_mask==eyebrow_idx, dtype=np.uint8))
            avg_color_rgb = [np.sum(debug_img[:,:,a]) for a in range(3)] # [::-1]
            avg_color_rgb = [int(t/all_pixel) for t in avg_color_rgb]

            eyebrow_color = tuple(avg_color_rgb)
            eyebrow_color = '%02x%02x%02x' % tuple(avg_color_rgb)

        result['eyebrow_color'] = '#{}'.format(eyebrow_color)

    if (hair_idx in all_possible_classes):

        # hair avg color
        debug_img = img_512_np.copy()
        debug_mask = mask.copy()

        debug_img[debug_mask!=hair_idx] = 0
        all_pixel = np.count_nonzero(np.array(debug_mask==hair_idx, dtype=np.uint8))
        avg_color_rgb = [np.sum(debug_img[:,:,a]) for a in range(3)] # [::-1]
        avg_color_rgb = [int(t/all_pixel) for t in avg_color_rgb]

        hair_color = tuple(avg_color_rgb)
        hair_color = '%02x%02x%02x' % tuple(avg_color_rgb)

        result['hair_color'] = '#{}'.format(hair_color)

    return result
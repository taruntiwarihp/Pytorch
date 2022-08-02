from .face68_class import FacePoints
from .general_cal import angles_from_four_points 

import numpy as np
import cv2
from scipy.spatial import distance as dist
from matplotlib.path import Path
from PIL import Image
import io
from mtcnn.mtcnn import MTCNN

class DlibHandler(object):

    def __init__(self, detector, predictor):
        self.face_points = FacePoints(detector, predictor)
        self.mtcnn = MTCNN()
    
    def handle_face_features(self, img_path):
        error = self.face_points.get_points(img_path)

        if error:
            return error

        nose_feature = self.nose_handle()
        eyes_feature = self.eye_handle()
        lips_feature = self.lip_handle()
        face_shape_feature = self.face_handle()
        lip_color = self.get_lip_color(img_path)
        face_color = self.get_face_color(img_path)
        eye_color = self.get_eye_color(img_path)

        face_features = {
            'nose_feature': nose_feature,
            'eye_feature': eyes_feature,
            'lip_feature': lips_feature,
            'face_shape_feature': face_shape_feature,
            'lip_color' : lip_color,
            'face_color' : face_color,
            'eye_color': eye_color,
        }

        return face_features

    def activate_prob(self, prob):
        if prob >= 1:
            return 1.0
        elif prob <=0:
            return 0.0
        else:
            return prob

    def nose_handle(self):

        area = [
            self.face_points.noses['nose_30'],
            self.face_points.noses['nose_31'],
            self.face_points.noses['nose_33'],
            self.face_points.noses['nose_35']
        ]

        nose_area = cv2.contourArea(
            np.around(
                np.array([[pt] for pt in area])
                    ).astype(np.int32)
        )

        nose_area = nose_area /  (self.face_points.HIGHT * self.face_points.WIDTH)

        nose_width = dist.euclidean(
            self.face_points.noses['nose_31'],
            self.face_points.noses['nose_35']
        ) / self.face_points.WIDTH

        nose_height = dist.euclidean(
            self.face_points.noses['nose_27'],
            self.face_points.noses['nose_33'],
        ) / self.face_points.HIGHT


        if nose_width > 0.20:
            return 'Wide'
        elif nose_width < 0.17:
            return 'Narrow'
        else:
            return 'Natural'

    def eye_handle(self):

        EYE_CLASSES = ["Round", "Almond", "Closeset", "Wideset", "Upturned", "Downturned", "Hoodied", "Monolid" ]
        
        first_half = dist.euclidean(
            self.face_points.jaws['jaw_0'],
            self.face_points.noses['nose_27'])
        second_half = dist.euclidean(
            self.face_points.noses['nose_27'], 
            self.face_points.jaws['jaw_16'])

        if first_half > second_half:

            a = dist.euclidean(
                self.face_points.right_eyes['eye_43'],
                self.face_points.right_eyes['eye_46']
            ) / self.face_points.HIGHT

            b = dist.euclidean(
                self.face_points.right_eyes['eye_44'],
                self.face_points.right_eyes['eye_47']
            ) / self.face_points.HIGHT

            eye_hight = (a + b) / 2

            eye_angle = angles_from_four_points(
                self.face_points.left_eyes['eye_36'], 
                self.face_points.left_eyes['eye_39'],
                self.face_points.noses['nose_27'],
                self.face_points.jaws['jaw_8'])

            eyeBrow_dist = dist.euclidean(
                self.face_points.left_eyebrows['eyebrow_20'],
                self.face_points.left_eyes['eye_38']
                )

            eyeBrow_dist1 = dist.euclidean(
                self.face_points.left_eyebrows['eyebrow_21'],
                self.face_points.left_eyes['eye_39']
            )

            eye_brow = (eyeBrow_dist + eyeBrow_dist1) / 2
        
        else:

            a = dist.euclidean(
                self.face_points.right_eyes['eye_43'],
                self.face_points.right_eyes['eye_46']
            ) / self.face_points.HIGHT

            b = dist.euclidean(
                self.face_points.right_eyes['eye_44'],
                self.face_points.right_eyes['eye_47']
            ) / self.face_points.HIGHT

            eye_hight = (a + b) / 2

            eye_angle = angles_from_four_points(
                self.face_points.right_eyes['eye_42'], 
                self.face_points.right_eyes['eye_45'],
                self.face_points.noses['nose_27'],
                self.face_points.jaws['jaw_8'])

            eyeBrow_dist = dist.euclidean(
                self.face_points.right_eyebrows['eyebrow_22'],
                self.face_points.right_eyes['eye_42']
                )

            eyeBrow_dist1 = dist.euclidean(
                self.face_points.right_eyebrows['eyebrow_23'],
                self.face_points.right_eyes['eye_43']
            )

            eye_brow = (eyeBrow_dist + eyeBrow_dist1) / 2
        
        eyes_gap = dist.euclidean(
            self.face_points.left_eyes['eye_39'],
            self.face_points.right_eyes['eye_42']
        ) / self.face_points.WIDTH

        ## Round eyes
        ROUND_RANGES = [0.1, 0.13]
        round_prob = (eye_hight - ROUND_RANGES[0]) / (ROUND_RANGES[1] - ROUND_RANGES[0])
        round_prob = self.activate_prob(round_prob)

        # Almond eyes
        ALMOND_RANGE = [0.05, 0.1]
        almond_prob = 1 - ((eye_hight - ALMOND_RANGE[0]) / (ALMOND_RANGE[1] - ALMOND_RANGE[0]))
        almond_prob = self.activate_prob(almond_prob)

        # close eyes
        CLOSE_RANGE = [0.18, 0.27]
        close_prob = 1 - ((eyes_gap - CLOSE_RANGE[0]) / (CLOSE_RANGE[1] - CLOSE_RANGE[0]))
        close_prob = self.activate_prob(close_prob)

        # wide eyes
        WIDE_RANGE = [0.27, 0.36]
        wide_prob = ((eyes_gap - WIDE_RANGE[0]) / (WIDE_RANGE[1] - WIDE_RANGE[0]))
        wide_prob = self.activate_prob(wide_prob)

        # upturn eyes
        UPTURN_RANGE = [75, 85]
        upturn_prob = 1 - ((eye_angle - UPTURN_RANGE[0]) / (UPTURN_RANGE[1] - UPTURN_RANGE[0]))
        upturn_prob = self.activate_prob(upturn_prob)

        # down eyes
        DOWN_RANGE = [85, 100]
        down_prob = ((eye_angle - WIDE_RANGE[0]) / (WIDE_RANGE[1] - WIDE_RANGE[0]))
        down_prob = self.activate_prob(down_prob)

        all_probs = np.array([
            round_prob, almond_prob, close_prob,
            wide_prob, upturn_prob, down_prob
        ])

        ## Default
        if all_probs.all() < 0.3:
            return "Almond"

 
        return EYE_CLASSES[np.argmax(all_probs)]
    
    def lip_handle(self):
        
        # Upperlips and lower lips
        upper_lips_dist1 = dist.euclidean(
            self.face_points.lips['lip_50'],
            self.face_points.mouths['mouth_61'])
        
        upper_lips_dist2 = dist.euclidean(
            self.face_points.lips['lip_52'],
            self.face_points.mouths['mouth_63'])

        upper_lips_dist = (upper_lips_dist1 + upper_lips_dist2) /2
        upper_lips_dist = upper_lips_dist / self.face_points.HIGHT

        lower_lips_dist = dist.euclidean(
            self.face_points.mouths['mouth_66'],
            self.face_points.lips['lip_57']
        ) / self.face_points.HIGHT

        # bratz and bow lips
        lips_dist = dist.euclidean(
            self.face_points.lips['lip_48'],
            self.face_points.lips['lip_54']
        ) / self.face_points.WIDTH

        lips_area = [
            self.face_points.lips['lip_50'],
            self.face_points.lips['lip_51'],
            self.face_points.lips['lip_52']]

        uperlips = cv2.contourArea(
            np.around(
                np.array([[pt] for pt in lips_area]
                )).astype(np.int32)) / (self.face_points.HIGHT * self.face_points.WIDTH)

        if uperlips > 100:
            return 'HeartShaped'

        if lower_lips_dist - upper_lips_dist < 0.028:
            if lips_dist < 0.36:
                return 'Round'
            else:
                return 'BowShaped'

        else:

            if lower_lips_dist > 0.070:
                return 'HeavyLower'

            elif upper_lips_dist > 0.040:
                return 'HeavyUpper  '
            else:
                return 'Natural'


    def face_handle(self):

        head_dist = dist.euclidean(
            self.face_points.left_eyebrows['eyebrow_17'],
            self.face_points.right_eyebrows['eyebrow_26']
        ) /  self.face_points.WIDTH

        jaw_dist = dist.euclidean(
            self.face_points.jaws['jaw_4'],
            self.face_points.jaws['jaw_12']
        ) / self.face_points.WIDTH


        ratio = jaw_dist - head_dist
        
        if ratio < 0 and ratio > -0.05:
            return 'Square'

        elif ratio < -0.06 and ratio > -0.09:
            return 'Round'

        elif ratio < -0.10:
            return 'Diamond'

        elif ratio > 0.020:
            return 'Oblong'
        
        elif ratio > 0 and ratio < 0.020:
            return 'Triangle'

        else: 
            return 'Oval'
        
    def region_avg(self, img, vertices):

        
        # vertices = face_68_points[indices]
        # print(vertices)
        path = Path(vertices)

        x, y = np.mgrid[:img.shape[1], :img.shape[0]]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = path.contains_points(points)

        # reshape mask for display
        img_mask = mask.reshape(x.shape).T
        avg_pixel = np.count_nonzero(img_mask)
        region_only = img.copy()

        region_only[img_mask==False] = 0
        avg_color = [np.sum(region_only[:,:,a]) for a in range(3)]
        avg_color = [int(t/avg_pixel) for t in avg_color]
        avg_color_rgb = avg_color[::-1]

        return '%02x%02x%02x' % tuple(avg_color_rgb)
    
    def get_lip_color(self, img_path):

        if isinstance(img_path, str):
            frame = cv2.imread(img_path)
        else:
            frame = Image.open(io.BytesIO(img_path)).convert('RGB')
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        vertices = [self.face_points.lips['lip_{}'.format(i)] for i in [48] + [i for i in range(59, 53, -1)]] + [self.face_points.mouths['mouth_{}'.format(i)] for i in [i for i in range(64, 68, 1)]] + [self.face_points.mouths['mouth_60'], self.face_points.lips['lip_48']]
        color = self.region_avg(frame, vertices)
        return '#{}'.format(color)

    def get_face_color(self, img_path):

        if isinstance(img_path, str):
            frame = cv2.imread(img_path)
        else:
            frame = Image.open(io.BytesIO(img_path)).convert('RGB')
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        vertices = [self.face_points.noses['nose_{}'.format(i)] for i in [27, 31, 35, 27]] 
        color = self.region_avg(frame, vertices)
        return '#{}'.format(color)
    
    def get_eye_color(self, img_path):

        if isinstance(img_path, str):
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = Image.open(io.BytesIO(img_path)).convert('RGB')
            frame = np.array(frame)
        
        result = self.mtcnn.detect_faces(frame)
        left_eye = result[0]['keypoints']['left_eye']
        right_eye = result[0]['keypoints']['right_eye']

        eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
        eye_radius = eye_distance/20

        f,s = left_eye[0] + int(eye_radius), left_eye[1]
        rgb = frame[(s,f)]

        color = '%02x%02x%02x' % tuple(rgb)

        return '#{}'.format(color)

            
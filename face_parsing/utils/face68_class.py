import dlib
import cv2
import numpy as np
from imutils import face_utils
from PIL import Image
import io
import matplotlib.pyplot as plt
# import matplotlib

# matplotlib.use('TkAgg')

class FacePoints(object):

    def __init__(self, detector=None, predictor=None):

        if detector:
            self.detector = detector
        else:
            self.detector = dlib.get_frontal_face_detector()
        if predictor:
            self.predictor = predictor
        else:
            self.predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')


    def get_points(self, img_path):

        if isinstance(img_path, str):
            frame = cv2.imread(img_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = Image.open(io.BytesIO(img_path)).convert('RGB')
            frame = np.array(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        self.all_points = []

        
        self.frame_copy = frame
        
        rect = self.detector(gray)

        if len(rect) == 0:
            return 'No person Detected'
        elif len(rect) > 1:
            return '{} faces are detected'.format(len(rect))
        else:
            rect = rect[0]

        shape = self.predictor(gray, rect)
        point_array = face_utils.shape_to_np(shape)

        x_max, x_min, y_max, y_min = max(point_array[:,0]), min(point_array[:,0]), max(point_array[:,1]), min(point_array[:,1])
        self.WIDTH = abs(x_max - x_min)
        self.HIGHT = abs(y_max - y_min)

        self.jaws = {}
        for i in range(17):
            self.jaws['jaw_' +str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        self.left_eyes = {}
        self.right_eyes = {}

        for i in range(36, 42):
            self.left_eyes['eye_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        for i in range(42, 48):
            self.right_eyes['eye_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        self.noses = {}
        for i in range(27, 36):
            self.noses['nose_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        self.left_eyebrows = {}
        self.right_eyebrows = {}

        for i in range(17, 22):
            self.left_eyebrows['eyebrow_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        for i in range(22, 27):
            self.right_eyebrows['eyebrow_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        self.lips = {}

        for i in range(48, 60):
            self.lips['lip_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        self.mouths = {}

        for i in range(60, 68):
            self.mouths['mouth_' + str(i)] = (point_array[i][0], point_array[i][1])
            self.all_points.append((point_array[i][0], point_array[i][1]))

        return None

    def plot_68_points(self, save_path=None):
        img_cp = self.frame_copy.copy()

        for (x,y) in self.all_points:
            img_cp = cv2.circle(img_cp, (int(x), int(y)), 1, (0, 0, 255), 1)

        if save_path == None:
            plt.imshow(img_cp)
            plt.imshow
        else:
            cv2.imwrite(save_path, img_cp)
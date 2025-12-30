import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ViewDetector:
    def __init__(self, model_path='checkpoints/golf_view_classifier.h5'):
        self.model = tf.keras.models.load_model(model_path)
        # self.verdicts = {"['down-the-line']": 0, "['face-on']": 1}
    
    def _get_image_view(self, image):
        '''Predict view from a single image or frame'''
        image = cv2.resize(image, (160, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = preprocess_input(image) # [-1, 1]
        img_batch = np.expand_dims(image, axis=0)
        pred = self.model.predict(img_batch, verbose=0)
        prob = pred[0][0]
        if prob >= 0.5:
            return 'face-on', prob
        else:
            return 'down-the-line', prob
    
if __name__ == '__main__':
    detector = ViewDetector()
    cap = cv2.VideoCapture(r"D:\Datathon\data\v1\TDTU-Golf-Pose-v1_fix_crop\Public Test\Ngoài trời - Outdoor\Band 2-4\Backside-8900-9_fix_crop.mov")

    sample_frames = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, num=7).astype(int)
    sample_frames = sample_frames[1:-1]

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, test_img = cap.read()
        pred = detector._get_image_view(test_img)
        print(pred)
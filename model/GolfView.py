import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ViewDetector:
    def __init__(self, model_path='checkpoints/golf_view_classifier (1).h5'):
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
        
    def get_view(self, path)-> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file {path} not found.")
        
        cap = cv2.VideoCapture(path)
        sample_frames = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, num=11).astype(int)
        sample_frames = sample_frames[1:-1]

        list_preds = []
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, test_img = cap.read()
            pred, prob = self._get_image_view(test_img)
            list_preds.append(pred)

        cap.release()
        return max(set(list_preds), key=list_preds.count)


if __name__ == '__main__':
    detector = ViewDetector()
    TEST_DIR = r"data/v1/TDTU-Golf-Pose-v1_fix_crop/"

    paths = []
    views = []
    predictions = []

    for root, dirs, files in os.walk(TEST_DIR):
        for fname in files:
            vid_path = os.path.join(root, fname)
            prediction = detector.get_view(vid_path)
            predictions.append(prediction)
            paths.append(vid_path)
            views.append('down-the-line' if vid_path.lower().find('backside') != -1 else 'face-on')
    
    import pandas as pd
    df = pd.DataFrame({
        'path': paths,
        'view': views,
        'prediction': predictions
    })
    
    print(df[df['view'] != df['prediction']]['path'].values)


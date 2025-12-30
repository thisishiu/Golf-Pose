import cv2
import os
import tqdm 
import pandas as pd
import numpy as np
from scipy.io import loadmat

x = loadmat('golfdb_master/data/golfDB.mat')
l = list(x['golfDB'][0])
d = dict()
for idx, k in enumerate(l):
    d["{:3d}".format(idx)] = list(l[idx])
df = pd.DataFrame(d).T
df.columns = ["id","youtube_id","player", "sex", "club","view","slow","events","bbox","split"]

TEMP_IMG_DIR = 'temp/training_frames'
VIDEO_DIR = 'data/raw/videos_160'
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

path_splits = []
views = []

def extract_frame_for_training(row):
    video_id = row['id'][0][0]
    vid_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(vid_path):
        print(f"Video {vid_path} not found, skipping.")
        return None
    print(f"Processing video {vid_path} - {row['view'][0]}")

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        frames_to_get = np.linspace(0, total_frames - 1, num=7).astype(int)
        frames_to_get = frames_to_get[1:-1]

        for frame_idx in frames_to_get:
            save_path = os.path.join(TEMP_IMG_DIR, f"{video_id}_{frame_idx}.jpg")
            if os.path.exists(save_path):
                path_splits.append(save_path)
                views.append(row['view'])
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (160, 160))
                cv2.imwrite(save_path, frame)
                path_splits.append(save_path)
                views.append(row['view'])

    cap.release()
    return None

print("[View Classifier] Extracting frames for training...")
df = df[df['view'] != 'other']
df.apply(extract_frame_for_training, axis=1)

df_splits = pd.DataFrame({
    'image_path': path_splits,
    'view': views
})

df_splits = df_splits.dropna(subset=['image_path'])
print(f"Num of samples added: {len(df_splits)}")
print(f"{df_splits['view'].value_counts()}")

df_splits['view'] = df_splits['view'].astype(str)

print(df_splits.sample(20))
exit()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df_splits, test_size=0.2, random_state=42, stratify=df_splits['view'])

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    preprocessing_function=preprocess_input,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    # rescale=1./255
    preprocessing_function=preprocess_input,
    )

# Flow from DataFrame
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path', # Cột chứa đường dẫn ảnh vừa tạo
    y_col='view',      # Cột nhãn
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary', # DTL và ONFACE
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='view',
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# MobileNetV2
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
# base_model.trainable = False # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=val_generator)

model.save('checkpoints/golf_view_classifier.h5')
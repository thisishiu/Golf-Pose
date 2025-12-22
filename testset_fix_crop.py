import os
import cv2
# import torch
# from model.Caddie import CaddieSetExtractor
# from torchvision import transforms
# from model.dataloader import *
from preprocessing.fix_crop_video import RobustSmartCropper

RAW_DIR = 'data/raw/TDTU-Golf-Pose-v1'
OUT_DIR = 'data/v1/TDTU-Golf-Pose-v1_fix_crop'

for root, _, files in os.walk(RAW_DIR):
    for fname in files:
        in_path = os.path.join(root, fname)
        rel_path = os.path.relpath(in_path, RAW_DIR)
        print(os.path.splitext(rel_path))
        out_path = os.path.join(OUT_DIR, os.path.splitext(rel_path)[0] + '_fix_crop.mov')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Đọc file đầu vào (tuỳ định dạng, ví dụ: với ảnh thì dùng PIL, numpy, ...)
        # data = ... đọc file ...
        # features = your_processing_function(data)
        # torch.save(features, out_path)

        # Ví dụ placeholder:
        # features = your_processing_function(in_path)
        # torch.save(features, out_path)
        # processor = RobustSmartCropper()
        # processor.process_video(in_path, out_path)

        processor = RobustSmartCropper()
        processor.process_video(in_path, out_path)


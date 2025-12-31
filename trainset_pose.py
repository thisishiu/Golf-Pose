import cv2
import time
import os
from ultralytics import YOLO

model_path = 'checkpoints/yolo11s-pose.pt' 
model = YOLO(model_path)

IN_DIR = 'data/raw/videos_160'
OUT_DIR = 'test/videos_160_skeleton'

i = 0
for root, _, files in os.walk(IN_DIR):
    for fname in files:
        in_path = os.path.join(root, fname)
        rel_path = os.path.relpath(in_path, IN_DIR)
        out_path = os.path.join(OUT_DIR, os.path.splitext(rel_path)[0] + '_skeleton.mov')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print(f"Không thể mở video {in_path}.")
            continue
        print(f"Đang xử lý video {in_path}...")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print(f"Kết thúc video {in_path}.")
                break
            results = model(frame, conf=0.5, verbose=False)
            
            print(results[0])

            annotated_frame = results[0].plot()

            if results[0].keypoints is not None:
                # Lấy tọa độ x, y của các khớp (shape: [num_people, 17, 2])
                keypoints = results[0].keypoints.xy.cpu().numpy()

            out.write(annotated_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        i += 1
        if i > 50:
            exit()

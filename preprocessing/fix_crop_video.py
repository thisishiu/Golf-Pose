import cv2
import numpy as np
from ultralytics import YOLO

class RobustSmartCropper:
    def __init__(self, model_path='checkpoints/yolov8n.pt', output_size=(160, 160)):
        self.model = YOLO(model_path)
        self.output_size = output_size

    def get_box_from_frame(self, frame):
        """Hàm phụ trợ: Lấy box từ 1 frame đơn lẻ"""
        # conf=0.5
        results = self.model(frame, classes=[0], verbose=False, conf=0.4)
        
        if len(results[0].boxes) == 0:
            return None
            
        # Lấy người có diện tích lớn nhất (tránh lấy nhầm người xem ở xa)
        best_box = None
        max_area = 0
        
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = box
                
        if best_box is None: return None

        x1, y1, x2, y2 = best_box
        
        # (Square Padding)
        cx = (x1 + x2) / 2 # center x
        cy = (y1 + y2) / 2 # center y
        w = x2 - x1
        h = y2 - y1
        
        # Mở rộng vùng nhìn 20%
        size = max(w, h) * 1.4
        
        final_x = int(cx - size / 2)
        final_y = int(cy - size / 2)
        final_size = int(size)
        
        return (final_x, final_y, final_size)

    def scan_for_crop_box(self, cap, num_samples=10):
        """
        CHIẾN LƯỢC QUÉT:
        Lấy mẫu rải rác 'num_samples' frame trải đều khắp video.
        Dừng ngay lập tức khi tìm thấy người.
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return None # error video

        # Tạo danh sách các index frame cần kiểm tra
        check_indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
        
        print(f"Finding golfer at frames: {check_indices}...")

        for idx in check_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            
            box = self.get_box_from_frame(frame)
            if box is not None:
                print(f"-> Found at {idx}! Box: {box}")
                return box # (Early Exit)
        
        print("[Fix Crop] Warning: No human detected in sampled frames.")
        return None

    def crop_fixed_region(self, frame, box):
        """Hàm crop và resize """
        x, y, size = box
        h_img, w_img = frame.shape[:2]
        
        # Safe crop logic (như cũ)
        top = max(0, -y)
        bottom = max(0, y + size - h_img)
        left = max(0, -x)
        right = max(0, x + size - w_img)
        
        img_crop = frame[max(0, y):min(h_img, y+size), max(0, x):min(w_img, x+size)]
        
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            img_crop = cv2.copyMakeBorder(img_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
            
        return cv2.resize(img_crop, self.output_size, interpolation=cv2.INTER_AREA)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return

        crop_box = self.scan_for_crop_box(cap, num_samples=10)
        
        # Fallback: Nếu quét nát video không thấy ai, dùng Center Crop
        if crop_box is None:
            ret, frame = cap.read() # Đọc frame đầu
            if ret:
                h, w = frame.shape[:2]
                size = min(h, w)
                crop_box = (int(w/2 - size/2), int(h/2 - size/2), size)
            else:
                return # Video hỏng hoàn toàn

        # Reset video về đầu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.output_size)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Chỉ tốn CPU cắt ghép, cực nhanh
            processed = self.crop_fixed_region(frame, crop_box)
            out.write(processed)
            
        cap.release()
        out.release()
        print(f"[Fix Crop] Done! Saved at: {output_path}")

# --- SỬ DỤNG ---
if __name__ == '__main__':
    processor = RobustSmartCropper()
    processor.process_video(r"D:\Datathon\data\raw\TDTU-Golf-Pose-v1\Public Test\Ngoài trời - Outdoor\Band 1-2\Backside-8897-2.mov", 'test\Backside-8897-2.mov_fix_crop.mov')
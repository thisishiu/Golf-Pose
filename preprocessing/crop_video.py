import cv2
import numpy as np
from ultralytics import YOLO

class SmartGolferCropper:
    def __init__(self, model_path='checkpoints\yolov8n.pt', output_size=(160, 160), smooth_factor=0.2):
        # Load YOLOv8 nano (nhanh, nhẹ)
        self.model = YOLO(model_path)
        self.output_size = output_size # (W, H)
        
        # Hệ số làm mượt (0 < alpha <= 1). Càng nhỏ càng mượt nhưng delay.
        self.smooth_factor = smooth_factor 
        self.prev_box = None # Lưu tọa độ box [cx, cy, w, h] của frame trước

    def get_smooth_box(self, current_box):
        """
        Áp dụng Exponential Moving Average (EMA) để làm mượt chuyển động của box.
        current_box: dạng [x1, y1, x2, y2]
        Trả về: dạng trung tâm [cx, cy, w, h] đã làm mượt
        """
        x1, y1, x2, y2 = current_box
        curr_w = x2 - x1
        curr_h = y2 - y1
        curr_cx = x1 + curr_w / 2
        curr_cy = y1 + curr_h / 2
        curr_smooth = np.array([curr_cx, curr_cy, curr_w, curr_h])

        if self.prev_box is None:
            self.prev_box = curr_smooth
            return curr_smooth
        
        # Công thức EMA: New = Alpha * Current + (1 - Alpha) * Previous
        smoothed_box = self.smooth_factor * curr_smooth + (1 - self.smooth_factor) * self.prev_box
        self.prev_box = smoothed_box
        return smoothed_box

    def crop_and_resize(self, frame, smooth_box):
        """
        Tạo vùng crop hình vuông từ box đã smooth và resize về 160x160
        """
        h_img, w_img, _ = frame.shape
        cx, cy, w, h = smooth_box
        
        # 1. Xác định cạnh hình vuông (lấy cạnh lớn nhất giữa w và h + thêm chút padding)
        max_side = max(w, h) * 1.2 # Thêm 20% padding để không bị sát người quá
        
        # 2. Tính tọa độ góc trên trái của hình vuông crop
        crop_x1 = int(cx - max_side / 2)
        crop_y1 = int(cy - max_side / 2)
        crop_x2 = int(crop_x1 + max_side)
        crop_y2 = int(crop_y1 + max_side)
        
        # 3. Xử lý biên (Padding nếu vùng crop vượt ra ngoài ảnh gốc)
        # Tạo một ảnh canvas đen lớn hơn để đảm bảo luôn crop được hình vuông
        pad_w = int(max_side // 2)
        pad_h = int(max_side // 2)
        padded_frame = cv2.copyMakeBorder(frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # Dịch chuyển tọa độ crop theo lượng padding vừa thêm
        crop_x1 += pad_w
        crop_y1 += pad_h
        crop_x2 += pad_w
        crop_y2 += pad_h

        # Thực hiện crop trên ảnh đã padding
        cropped = padded_frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 4. Resize về 160x160
        resized = cv2.resize(cropped, self.output_size, interpolation=cv2.INTER_AREA)
        return resized

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return
        
        # Setup video writer
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec cho mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, self.output_size)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # --- BƯỚC 1: DETECTION ---
            # Chỉ detect class 0 (person), verbose=False để tắt log
            results = self.model(frame, classes=[0], verbose=False) 
            
            found_person = False
            if len(results[0].boxes) > 0:
                # Lấy box có độ tin cậy cao nhất (thường là người chính)
                # box format: [x1, y1, x2, y2] (tensor -> numpy -> list)
                box = results[0].boxes[0].xyxy.cpu().numpy()[0]
                found_person = True
            else:
                # Nếu frame này không thấy người, dùng lại box của frame trước (nếu có)
                if self.prev_box is not None:
                    # Convert lại từ [cx, cy, w, h] sang [x1, y1, x2, y2] tạm để tái sử dụng logic
                    pcx, pcy, pw, ph = self.prev_box
                    box = [pcx - pw/2, pcy - ph/2, pcx + pw/2, pcy + ph/2]
                    found_person = True

            if found_person:
                # --- BƯỚC 2: SMOOTHING ---
                smooth_box_center = self.get_smooth_box(box)
                
                # --- BƯỚC 3 & 4: SQUARE CROP & RESIZE ---
                final_frame = self.crop_and_resize(frame, smooth_box_center)
            else:
                # Trường hợp tệ nhất: từ đầu đến giờ chưa thấy người nào
                # Trả về frame đen hoặc resize toàn bộ ảnh gốc
                final_frame = cv2.resize(frame, self.output_size)
                print(f"Warning: No person detected in frame {frame_count}")

            out.write(final_frame)
            
            # (Optional) Xem trước kết quả
            # cv2.imshow('Smart Crop', final_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Crop done! Save as: {output_path}")

# --- CHẠY THỬ ---
# Thay đường dẫn video của bạn vào đây
input_video = r"D:\Datathon\data\raw\TDTU-Golf-Pose-v1\Public Test\Trong nhà - Indoor\Band 8-10\Side-5999-1.mov"
output_video = r"test/Side-5999-1.mov"

cropper = SmartGolferCropper(model_path='checkpoints\yolov8n.pt', output_size=(160, 160), smooth_factor=0.15)
# Lần đầu chạy sẽ tự động tải yolov8n.pt
cropper.process_video(input_video, output_video)
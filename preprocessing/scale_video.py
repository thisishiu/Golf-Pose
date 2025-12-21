import cv2

def simple_scale_video(input_path, output_path, target_size=(160, 160)):
    # 1. Mở video gốc
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {input_path}")
        return

    # Lấy thông số FPS để video đầu ra có tốc độ đúng
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 2. Khởi tạo bộ ghi video (VideoWriter)
    # mp4v là codec chuẩn cho file .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    print(f"Process: {input_path} -> {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. Resize trực tiếp (Bóp méo hình ảnh để vừa khung 160x160)
        # INTER_AREA: Tốt nhất khi thu nhỏ ảnh (downsampling) để tránh bị nhiễu (aliasing)
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # 4. Ghi vào file mới
        out.write(resized_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    print("Done!")

# --- CHẠY THỬ ---
simple_scale_video(r"D:\Datathon\data\raw\TDTU-Golf-Pose-v1\Public Test\Trong nhà - Indoor\Band 8-10\Side-5999-1.mov", 'test\Side-5999-1_scale.mov')
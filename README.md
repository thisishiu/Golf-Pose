## Test (thử nghiệm riêng model)
### clone golfdb-master -> có SwingNet
- Tải các model pretrain
  - `golfdb-master\model.py`
    - đọc vào cpu, sửa mấy chỗ .cuda()

  - `golfdb-master\data\generate_splits.py`
    - #25, #31, #36 #4 đổi đường dẫn (thêm gốc)
    - chạy -> đã tạo ra các metadata trong `data/raw/video_160_split/`

## Test model (thử nghiệm theo pipeline và tiến hành thử nghiệm trên tập test)
- Dựa vào file test, code module lấy vector đặc trưng cho từng động tác. [bỏ, cần làm pipeline rõ]

- Từ model SwingNet  
  - lấy pose cho từng frame ở đầu ra swingnet  
  - -> Đâu ra không ổn lắm (có tìm được 8 phase nhưng cái sai cái đúng)

- Nhận thấy tác giả train SwingNet với video 160x160  
    - dùng yolo crop video thành 160x160, tỉ lệ swingnet đúng tăng lên ~100% <> thời gian lâu (`preprocessing\crop_video.py`)  
    - dùng scale thì nhanh nhưng không hề chính xác (`preprocessing\scale_video.py`)

    - => fix crop, vừa nhanh vừa chính xác (`preprocessing\fix_crop_video.py`) **[Xong phần SwingNet]**  
    (fix crop cho tập test xem ở `data/v1/TDTU-Golf-Pose-v1_fix_crop`)

- Test pose bằng yolo11n-pose.pt (kết hợp với **SwingNet** để ra features)
  - chạy file `testset_features.py` để lấy features cho từng cú đánh trong test
  - đã lấy được vector đặc trưng của 1 cú đánh, các đặc trưng dựa vào 15 features CaddieSet (xem ở paper CaddieSet)
  - (mỗi video được xuất đặc trưng xem ở `test/TDTU-Golf-Pose-v1_40ft` và `test/TDTU-Golf-Pose-v1_120ft` (sửa path trong file `testset_features.py`))

- Tiếp theo là lấy vector đặc trung cho 1 cú đánh diểm cao ~10, dựa vào golfdb (1400 video)
  - chạy `correct_posture_matrix.py` để lấy vector embedding cho tập train
  - (đã thu thập, xem ở `checkpoints/golf_pose_stats_*`)

- Chấm điểm các vector trong tập test => kết quả không ổn lắm **(dự tính chỉ 50% Acc)**
  - kết quả nằm trong file `eda/stats_40ft.*` và `eda/stats_120ft.*`
  - trong quá trình eda, statistic model trên:
    có vẻ như không gian vector đặc trưng được chia làm 2 cum rõ rệt (xem ở 2 `file eda/stats_40ft` và `eda/stats_120ft`)
  - => Dự đoán nguyên nhân là các cú đánh được quay từ 2 góc ('back' và 'side')

### Xử lí góc quay
- Ý tưởng là xử xí mỗi góc mỗi khác, không làm giống nhau
  - cần 1 model đơn giản detect góc quay ('back' or 'side')
  - dự đoán là dùng CNN / MobileNetV2 **[bỏ, dectect góc quay sau khi dự đoán khung xương bằng yolo]**
  - => tiến hành label tập train (golfdb), label góc quay là 'back' hay 'side' [bỏ, đã có label]
  - trong repo golfdb, tác giả đã label sẵn (`golfdb_master\data\golfDB.mat`)

- Lấy đầu ra YoloPose làm model view detect **[bỏ, cách này phá pipeline]**

- Từ video đầu vào, lấy ngẫu nhiên các frame (số lẻ) -> model CNN/MobileNetV2 để detect góc quay -> lấy mode (chế độ) của các kết quả đó làm góc quay của video

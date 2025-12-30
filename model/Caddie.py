import sys
sys.path.append('.')

import torch
import cv2
import numpy as np
from model.dataloader import Normalize
from ultralytics import YOLO

class CaddieSetExtractor:
    def __init__(self, model_path='checkpoints/yolo11n-pose.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[CaddieSetExtractor] Running on: {self.device}")

        # Load model và đẩy sang device
        self.pose_model = YOLO(model_path)
        
        # Mapping Keypoints
        self.KP = {
            'nose': 0, 'l_shoulder': 5, 'r_shoulder': 6,
            'l_elbow': 7, 'r_elbow': 8, 'l_wrist': 9, 'r_wrist': 10,
            'l_hip': 11, 'r_hip': 12, 'l_knee': 13, 'r_knee': 14,
            'l_ankle': 15, 'r_ankle': 16
        }

        self.FACEON_SELECTED_FEATURES = {
            0: ['STANCE-RATIO', 'SHOULDER-ANGLE', 'UPPER-TILT'],
            1: ['SHOULDER-LOC', 'UPPER-TILT', 'RIGHT-ARM-ANGLE', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HEAD-LOC'],
            2: ['SHOULDER-LOC', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HEAD-LOC'],
            3: ['SHOULDER-LOC', 'RIGHT-LEG-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HEAD-LOC'],
            4: ['SHOULDER-HANGING-BACK', 'RIGHT-ARMPIT-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HIP-HANGING-BACK', 'HEAD-LOC'],
            5: ['SHOULDER-HANGING-BACK', 'SHOULDER-ANGLE', 'RIGHT-ARM-ANGLE', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HIP-HANGING-BACK', 'HEAD-LOC'],
            6: ['WEIGHT-SHIFT', 'RIGHT-LEG-ANGLE', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HEAD-LOC'],
            7: ['HIP-SHIFTED', 'FINISH-ANGLE']
        }

        self.DTL_SELECTED_FEATURES = {}

        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _get_dist(self, a, b):
        return np.linalg.norm(a - b)

    def _get_angle_3p(self, a, b, c):
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_slope_angle(self, a, b):
        return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))

    def get_FACEON_metrics(self, k, address_k):
        kp = self.KP
        feats = {}
        
        nose = k[kp['nose']]
        l_sho, r_sho = k[kp['l_shoulder']], k[kp['r_shoulder']]
        l_elb, r_elb = k[kp['l_elbow']], k[kp['r_elbow']]
        l_wri, r_wri = k[kp['l_wrist']], k[kp['r_wrist']]
        l_hip, r_hip = k[kp['l_hip']], k[kp['r_hip']]
        r_kne = k[kp['r_knee']]
        l_ank, r_ank = k[kp['l_ankle']], k[kp['r_ankle']]

        if address_k is not None:
            addr_nose = address_k[kp['nose']]
            addr_mid_hip = (address_k[kp['l_hip']] + address_k[kp['r_hip']]) / 2
            ref_stride = self._get_dist(address_k[kp['l_ankle']], address_k[kp['r_ankle']]) + 1e-6
            ref_hip_angle = self._get_slope_angle(address_k[kp['l_hip']], address_k[kp['r_hip']])
        else:
            addr_nose = nose
            addr_mid_hip = (l_hip + r_hip) / 2
            ref_stride = self._get_dist(l_ank, r_ank) + 1e-6
            ref_hip_angle = self._get_slope_angle(l_hip, r_hip)

        curr_shoulder_width = self._get_dist(l_sho, r_sho) + 1e-6
        curr_hip_width = self._get_dist(l_hip, r_hip) + 1e-6
        
        feats['SHOULDER-ANGLE'] = self._get_slope_angle(r_sho, l_sho)
        feats['UPPER-TILT'] = curr_hip_width / curr_shoulder_width
        feats['STANCE-RATIO'] = curr_shoulder_width / ref_stride
        feats['HEAD-LOC'] = self._get_dist(nose, addr_nose) / ref_stride
        feats['SHOULDER-LOC'] = (l_sho[0] - min(l_ank[0], r_ank[0])) / (abs(r_ank[0]-l_ank[0]) + 1e-6)
        feats['LEFT-ARM-ANGLE'] = self._get_angle_3p(l_sho, l_elb, l_wri)
        feats['RIGHT-ARM-ANGLE'] = self._get_angle_3p(r_sho, r_elb, r_wri)
        feats['HIP-ROTATION'] = abs(self._get_slope_angle(l_hip, r_hip) - ref_hip_angle)
        feats['HIP-SHIFTED'] = self._get_dist((l_hip+r_hip)/2, addr_mid_hip) / ref_stride
        feats['RIGHT-LEG-ANGLE'] = self._get_angle_3p(r_hip, r_kne, r_ank)
        feats['SHOULDER-HANGING-BACK'] = self._get_dist(l_ank, l_sho) / ref_stride
        feats['HIP-HANGING-BACK'] = self._get_dist(l_ank, l_hip) / ref_stride
        feats['RIGHT-ARMPIT-ANGLE'] = self._get_angle_3p(r_elb, r_sho, r_hip)
        feats['WEIGHT-SHIFT'] = self._get_slope_angle(l_ank, l_hip)
        feats['FINISH-ANGLE'] = self._get_slope_angle(l_ank, r_hip)

        return feats
    

    def get_DTL_metrics(self, k, address_k):
        """
        Dựa trên input: k (keypoints hiện tại), address_k (keypoints tại frame 0)
        """
        kp = self.KP
        feats = {}

        nose = k[kp['nose']]
        l_sho, r_sho = k[kp['l_shoulder']], k[kp['r_shoulder']]
        l_elb, r_elb = k[kp['l_elbow']], k[kp['r_elbow']]
        l_wri, r_wri = k[kp['l_wrist']], k[kp['r_wrist']]
        l_hip, r_hip = k[kp['l_hip']], k[kp['r_hip']]
        l_kne, r_kne = k[kp['l_knee']], k[kp['r_knee']]
        l_ank, r_ank = k[kp['l_ankle']], k[kp['r_ankle']]

        # Tính các điểm trung gian
        mid_shoulder = (l_sho + r_sho) / 2
        mid_hip = (l_hip + r_hip) / 2
        
        # Xử lý thông tin tại Address (Frame 0)
        if address_k is not None:
            addr_mid_hip = (address_k[kp['l_hip']] + address_k[kp['r_hip']]) / 2
            # Reference height: Dùng chiều dài thân người tại address để chuẩn hóa khoảng cách
            addr_mid_shoulder = (address_k[kp['l_shoulder']] + address_k[kp['r_shoulder']]) / 2
            ref_len = self._get_dist(addr_mid_shoulder, addr_mid_hip) + 1e-6
            ref_hip_angle = self._get_slope_angle(address_k[kp['l_hip']], address_k[kp['r_hip']])
        else:
            addr_mid_hip = mid_hip
            ref_len = self._get_dist(mid_shoulder, mid_hip) + 1e-6
            ref_hip_angle = self._get_slope_angle(l_hip, r_hip)

        # 1. SPINE-ANGLE: Spine angle relative to horizontal
        # Góc tạo bởi đường nối giữa hông và vai so với phương ngang
        feats['SPINE-ANGLE'] = abs(self._get_slope_angle(mid_hip, mid_shoulder))

        # 2. LOWER-ANGLE: Angle formed by right pelvis, knee, and ankle (Góc chân phải)
        feats['LOWER-ANGLE'] = self._get_angle_3p(r_hip, r_kne, r_ank)

        # 3. SHOULDER-ANGLE: Shoulder angle relative to horizontal
        feats['SHOULDER-ANGLE'] = self._get_slope_angle(r_sho, l_sho)

        # 4. LEFT-ARM-ANGLE: Angle formed by left shoulder, elbow, and wrist
        feats['LEFT-ARM-ANGLE'] = self._get_angle_3p(l_sho, l_elb, l_wri)

        # 5. RIGHT-ARM-ANGLE: Angle formed by right shoulder, elbow, and wrist
        feats['RIGHT-ARM-ANGLE'] = self._get_angle_3p(r_sho, r_elb, r_wri)

        # 6. HIP-LINE: Movement of hip relative to Address ratio
        # Độ dịch chuyển của tâm hông so với vị trí ban đầu (thường xét theo trục X - tiến/lùi)
        feats['HIP-LINE'] = (mid_hip[0] - addr_mid_hip[0]) / ref_len

        # 7. HIP-ANGLE: Rotation degree of pelvis relative to Address degree
        # Độ xoay hông (độ dốc đường nối 2 hông) so với lúc setup
        feats['HIP-ANGLE'] = abs(self._get_slope_angle(l_hip, r_hip) - ref_hip_angle)

        # 8. RIGHT-DISTANCE: Gap between right elbow and the torso ratio
        # Khoảng cách từ khuỷu tay phải đến tâm hông (hoặc đường sống lưng). 
        feats['RIGHT-DISTANCE'] = self._get_dist(r_elb, mid_hip) / ref_len

        # 9. LEFT-LEG-ANGLE: Angle formed by left pelvis, knee, and ankle
        feats['LEFT-LEG-ANGLE'] = self._get_angle_3p(l_hip, l_kne, l_ank)

        return feats

    def process_sequence(self, sequence, view):
        """
        Input: Tensor (1, 8, 3, 160, 160) hoặc (8, 3, 160, 160)
        Output: Numpy array shape (40,)
        """
        sequence = sequence.to(self.device)
        
        sequence = (
            self.norm.inverse(sequence)
                # .mul(255.0)
                # .clamp(0, 255)
            )
        
        sequence = sequence.squeeze(0)  # (F, C, H, W)
        
        # Ultralytics nhận Tensor [B, C, H, W] trên GPU. Không cần chuyển về numpy
        results = self.pose_model(sequence, verbose=False, conf=0.1)
        
        # Tìm Address Keypoints (Frame 0)
        address_kp = None
        for res in results:
            if len(res.keypoints) > 0:
                # .cpu().numpy() ở đây là bắt buộc để tính toán với hàm calculate_metrics
                address_kp = res.keypoints.xy[0].cpu().numpy()
                break
        
        if address_kp is None: 
            print("[CaddieSetExtractor] Warning: No address_kp detected.")
            return np.zeros(40)

        final_feature_vector = []

        if view == 'FACEON':
            for i, res in enumerate(results):
                if len(res.keypoints) > 0:
                    k = res.keypoints.xy[0].cpu().numpy()
                    all_15_metrics = self.get_FACEON_metrics(k, address_kp)
                else:
                    all_15_metrics = self.get_FACEON_metrics(address_kp, address_kp)

                # chỉ lấy các metrics cần thiết
                metrics_to_take = self.FACEON_SELECTED_FEATURES.get(i, [])
                for metric_name in metrics_to_take:
                    value = all_15_metrics.get(metric_name, 0.0)
                    final_feature_vector.append(value)

                # lấy tất cả 15 metrics
                # for metric_to_take in all_15_metrics.keys():
                #     final_feature_vector.append(all_15_metrics[metric_to_take])
        elif view == 'DTL':
            for i, res in enumerate(results):
                if len(res.keypoints) > 0:
                    k = res.keypoints.xy[0].cpu().numpy()
                    all_9_metrics = self.get_DTL_metrics(k, address_kp)
                else:
                    all_9_metrics = self.get_DTL_metrics(address_kp, address_kp)

                # chỉ lấy các metrics cần thiết
                # metrics_to_take = self.DTL_SELECTED_FEATURES.get(i, [])
                # for metric_name in metrics_to_take:
                #     value = all_9_metrics.get(metric_name, 0.0)
                #     final_feature_vector.append(value)

                # lấy tất cả 9 metrics
                for metric_to_take in all_9_metrics.keys():
                    final_feature_vector.append(all_9_metrics[metric_to_take])

        else:
            print(f"[CaddieSetExtractor] Error: Unknown view type '{view}'.")
            return np.zeros(40)

        return np.array(final_feature_vector)

if __name__ == '__main__':
    extractor = CaddieSetExtractor()

    # Giả lập input tensor trên CPU (như dataloader thường trả về)
    dummy_sequence = torch.randn(1, 8, 3, 160, 160)
    
    # Hàm process sẽ tự đẩy lên GPU xử lý
    features = extractor.process_sequence(dummy_sequence)
    print("Extracted features shape:", features.shape)
    print("Values:", features)
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

        self.SELECTED_FEATURES = {
            0: ['STANCE-RATIO', 'SHOULDER-ANGLE', 'UPPER-TILT'],
            1: ['SHOULDER-LOC', 'UPPER-TILT', 'RIGHT-ARM-ANGLE', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HEAD-LOC'],
            2: ['SHOULDER-LOC', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HEAD-LOC'],
            3: ['SHOULDER-LOC', 'RIGHT-LEG-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HEAD-LOC'],
            4: ['SHOULDER-HANGING-BACK', 'RIGHT-ARMPIT-ANGLE', 'HIP-SHIFTED', 'HIP-ROTATION', 'HIP-HANGING-BACK', 'HEAD-LOC'],
            5: ['SHOULDER-HANGING-BACK', 'SHOULDER-ANGLE', 'RIGHT-ARM-ANGLE', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HIP-HANGING-BACK', 'HEAD-LOC'],
            6: ['WEIGHT-SHIFT', 'RIGHT-LEG-ANGLE', 'LEFT-ARM-ANGLE', 'HIP-SHIFTED', 'HEAD-LOC'],
            7: ['HIP-SHIFTED', 'FINISH-ANGLE']
        }

        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # --- Các hàm tính toán hình học (Giữ nguyên Numpy vì nhanh và linh hoạt cho số lượng ít) ---
    def _get_dist(self, a, b):
        return np.linalg.norm(a - b)

    def _get_angle_3p(self, a, b, c):
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_slope_angle(self, a, b):
        return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))

    def calculate_all_15_metrics(self, k, address_k):
        # ... (Giữ nguyên logic hàm này như cũ) ...
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

    def process_sequence(self, sequence):
        """
        Input: Tensor (1, 8, 3, 160, 160) hoặc (8, 3, 160, 160)
        Output: Numpy array shape (40,)
        """
        sequence = sequence.to(self.device)

        if sequence.dim() == 4:
            sequence = sequence.unsqueeze(0) # (B, F, C, H, W)
        
        sequence = (
            self.norm.inverse(sequence)
                # .mul(255.0)
                # .clamp(0, 255)
            )
        
        sequence = sequence.squeeze(0)  # (F, C, H, W)
        
        # Ultralytics nhận Tensor [B, C, H, W] trên GPU. Không cần chuyển về numpy
        results = self.pose_model(sequence, verbose=False, conf=0.1)
        # for i, r in enumerate(results):
            # print(f"result: {r.keypoints}")

        #Post-processing (Cần convert về CPU numpy để tính toán hình học)
        # Bước này nhẹ, nên làm ở CPU là ổn
        
        # Tìm Address Keypoints (Frame 0)
        address_kp = None
        for res in results:
            if len(res.keypoints) > 0:
                # .cpu().numpy() ở đây là bắt buộc để tính toán với hàm calculate_metrics
                address_kp = res.keypoints.xy[0].cpu().numpy()
                break
        
        if address_kp is None: 
            # print("[CaddieSetExtractor] Warning: No person detected.")
            return np.zeros(40)

        final_feature_vector = []

        for i, res in enumerate(results):
            if len(res.keypoints) > 0:
                k = res.keypoints.xy[0].cpu().numpy()
                all_15_metrics = self.calculate_all_15_metrics(k, address_kp)
            else:
                all_15_metrics = self.calculate_all_15_metrics(address_kp, address_kp)

            # metrics_to_take = self.SELECTED_FEATURES.get(i, [])
            # for metric_name in metrics_to_take:
            #     value = all_15_metrics.get(metric_name, 0.0)
                # final_feature_vector.append(value)
            for metric_to_take in all_15_metrics.keys():
                final_feature_vector.append(all_15_metrics[metric_to_take])

        return np.array(final_feature_vector)

if __name__ == '__main__':
    extractor = CaddieSetExtractor()

    # Giả lập input tensor trên CPU (như dataloader thường trả về)
    dummy_sequence = torch.randn(1, 8, 3, 160, 160)
    
    # Hàm process sẽ tự đẩy lên GPU xử lý
    features = extractor.process_sequence(dummy_sequence)
    print("Extracted features shape:", features.shape)
    print("Values:", features)
import torch
import cv2
import numpy as np
from model.dataloader import *
from model.Caddie import CaddieSetExtractor
from torch.utils.data import DataLoader
from torchvision import transforms
# from model.SwingNet import EventDetector
# from model.MobileNetV2 import MobileNetV2

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/raw/videos_160_split/golfDB.pkl',
                     vid_dir='data/raw/videos_160',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
    # tạm thời là batch_size=1 do CaddieSetExtractor chưa handle được batch size >1

    extractor = CaddieSetExtractor()

    good_pose_features = None
    for i, sample in enumerate(data_loader):
        if i % 100 == 0: 
            print(f"Sample {i}")

        sequence = sample['images']  # (B, F, C, H, W)
        features = extractor.process_sequence(sequence)
        
        # EWA
        alpha = 0.9
        good_pose_features = alpha * good_pose_features + (1 - alpha) * features if i > 0 else features
        good_pose_features = good_pose_features / (1 - alpha**(i + 1))

    torch.save(good_pose_features, 'checkpoints/good_posture_features.pt')
import torch
import cv2
import numpy as np
from model.dataloader import *
from model.Caddie import CaddieSetExtractor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
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
    collected_features = []
    for i, sample in enumerate(tqdm(data_loader)):
        try:
            sequence = sample['images']  # (B, F, C, H, W)
            features = extractor.process_sequence(sequence)

            if np.all(features == 0):
                continue

            collected_features.append(features)
#
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    if len(collected_features) > 0:
        feature_matrix = np.array(collected_features)
        mean_vector = np.mean(feature_matrix, axis=0)
        std_vector = np.std(feature_matrix, axis=0)
        std_vector[std_vector == 0] = 1e-6
        save_dict = {
            'mean': mean_vector,
            'std': std_vector, 
            'n_samples': len(collected_features),
            'table': feature_matrix
        }     
        torch.save(save_dict, 'checkpoints/golf_pose_stats.pt')
        print("Saved")
        print("Mean vector shape:", mean_vector.shape)
        print("Std vector shape:", std_vector.shape)
        print("len:", len(collected_features))
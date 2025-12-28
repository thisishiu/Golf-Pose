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
        # try:
            video = sample['images']  # (B, F, C, H, W)
            events = sample['events'].reshape(-1)  # (num_events,)     

            sequence = video[:, events, :, :, :]
            features = extractor.process_sequence(sequence)

            if np.all(features == 0):
                continue

            collected_features.append(features)
            # print(f"Processed sample {i}, shape: {features.shape}.")
#
        # except Exception as e:
        #     print(f"Error processing sample {i}: {e}")

    if len(collected_features) > 0:
        feature_matrix = np.array(collected_features)
        save_dict = {
            'table': feature_matrix
        }     
        torch.save(save_dict, 'checkpoints/golf_pose_stats (*).pt')
        print("Saved")
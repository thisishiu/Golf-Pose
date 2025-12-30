import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        view = a['view']
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        path = osp.join(self.vid_dir, '{}.mp4'.format(a['id']))
        cap = cv2.VideoCapture(path)
        # img = cap.read()[1]
        # print(img.min(), img.max())
        # exit()
        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {
            'images': np.asarray(images), 
            'labels': np.asarray(labels),
            'events': np.asarray(events[1:-1]),
            'path': path,
            'view': view
            }
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {
            'images': torch.from_numpy(images).float().div(255.),
            'labels': torch.from_numpy(labels).long(),
            'events': torch.from_numpy(sample['events']).long() if sample.get('events') is not None else None,
            'path': sample.get('path', None),
            'view': sample.get('view', None)
            }


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        # images: Tensor (T, C, H, W)
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {
            'images': images,
            'labels': labels,
            'events': sample['events'] if sample.get('events') is not None else None,
            'path': sample.get('path', None),
            'view': sample.get('view', None)
            }

    def inverse(self, tensor):
        """Denormalize a tensor image."""
        if tensor.dim() == 4:
            raise ValueError("[Normalize] Inverting expected tensor to be 5-D (B, F, C, H, W), got {}-D".format(tensor.dim()))
        device = tensor.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        tensor = tensor.clone()
        tensor.mul_(std[None, None, :, None, None]).add_(mean[None, None, :, None, None])
        return tensor

if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/raw/videos_160_split/train_split_1.pkl',
                     vid_dir='data/raw/videos_160',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print(sample)
        # print('{} events: {}'.format(len(events), events))




    





       


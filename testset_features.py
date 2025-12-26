import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from model.Caddie import CaddieSetExtractor
from model.dataloader import *
from model.SwingNet import EventDetector

def get_video(path):
    input_size = 160
    transform =transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cap = cv2.VideoCapture(path)

    frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    ratio = input_size / max(frame_size)

    new_size = tuple([int(x * ratio) for x in frame_size])
    delta_w = input_size - new_size[1]
    delta_h = input_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # preprocess and return frames
    images = []
    for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, img = cap.read()
        resized = cv2.resize(img, (new_size[1], new_size[0]))
        b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

        b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        images.append(b_img_rgb)
    cap.release()
    labels = np.zeros(len(images)) # only for compatibility with transforms
    sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
    if transform:
        sample = transform(sample)
    return sample

if __name__ == '__main__':

    model = EventDetector(pretrain=True,
                        width_mult=1.,
                        lstm_layers=1,
                        lstm_hidden=256,
                        bidirectional=True,
                        dropout=False)
    try:
        save_dict = torch.load('checkpoints/swingnet_1800.pth.tar', map_location=torch.device('cpu'))
    except:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")


    RAW_DIR = 'data/v1/TDTU-Golf-Pose-v1_fix_crop'
    OUT_DIR = 'test/TDTU-Golf-Pose-v1_features'

    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, RAW_DIR)
            print(os.path.splitext(rel_path))
            out_path = os.path.join(OUT_DIR, os.path.splitext(rel_path)[0] + '.pt')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"Processing file: {in_path}")

            sample = get_video(in_path)
            images = sample['images'].unsqueeze(0)
            seq_length = 64
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                # print(image_batch.shape)
                logits = model(image_batch)
                # logits = model(image_batch.cuda())
                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1

            events = np.argmax(probs, axis=0)[:-1]
            print(f"Event: {events}") # (8,)
            # print(images.shape) # torch.Size([1, 223, 3, 160, 160])
            sequence = images[:,events,:,:,:] # torch.Size([1, 8, 3, 160, 160])

            print(f"Loaded video frames shape: {sequence.shape}")  

            # Khởi tạo extractor
            extractor = CaddieSetExtractor()

            feature_vector = extractor.process_sequence(sequence)

            print(f"Input shape: {sequence.shape}")
            print(f"Output Feature Vector shape: {feature_vector.shape}")

            torch.save(feature_vector, out_path)

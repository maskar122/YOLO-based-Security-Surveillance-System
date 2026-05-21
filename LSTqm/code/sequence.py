import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Transform للفريمات
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class VideoDataset(Dataset):
    def __init__(self, frames_root, sequence_length=8):
        self.sequence_length = sequence_length
        self.samples = []
        self.categories = ["NonViolence", "violence"]

        for label, category in enumerate(self.categories):
            cat_path = os.path.join(frames_root, category)
            for video_folder in os.listdir(cat_path):
                folder_path = os.path.join(cat_path, video_folder)
                frames = sorted(os.listdir(folder_path))
                for i in range(0, len(frames) - sequence_length + 1):
                    seq = frames[i:i+sequence_length]
                    self.samples.append((seq, label, folder_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label, folder = self.samples[idx]
        imgs = []
        for f in seq:
            img_path = os.path.join(folder, f)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)  # shape: (sequence_length, 3, 224, 224)
        return imgs, label

# مثال عمل DataLoader
dataset = VideoDataset(r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\frames", sequence_length=8)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print("عدد sequences:", len(dataset))
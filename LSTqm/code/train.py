import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os

# -------------------------
# 1️⃣ إعداد الـ Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 2️⃣ Dataset
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class VideoDataset(Dataset):
    def __init__(self, frames_root, sequence_length=8):
        self.sequence_length = sequence_length
        self.samples = []
        self.categories = ["NonViolence", "violence"]  # تأكد الاسم صحيح

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
        imgs = torch.stack(imgs)  # (sequence_length, 3, 224, 224)
        return imgs, label

# -------------------------
# 3️⃣ DataLoader
# -------------------------
dataset = VideoDataset(r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\frames", sequence_length=8)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -------------------------
# 4️⃣ موديل CNN + LSTM
# -------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        features = []
        for i in range(t):
            f = self.cnn(x[:, i])  # (batch, 512,1,1)
            f = f.view(b, -1)
            features.append(f)
        features = torch.stack(features, dim=1)  # (batch, seq, 512)
        out, _ = self.lstm(features)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = CNN_LSTM(num_classes=2).to(device)

# -------------------------
# 5️⃣ Training setup
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5  # ممكن تزوده بعد التجربة

# -------------------------
# 6️⃣ Training loop
# -------------------------
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (seqs, labels) in enumerate(dataloader):
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx+1) % 50 == 0 or (batch_idx+1) == len(dataloader):
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} done. Average Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%\n")

# -------------------------
# 7️⃣ حفظ الموديل
# -------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_model.pth")
print("Model trained and saved successfully at models/lstm_model.pth")
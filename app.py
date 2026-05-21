import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ======================
# 1️⃣ Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 2️⃣ LSTM Model
# ======================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        features = []
        for i in range(t):
            f = self.cnn(x[:, i])
            f = f.view(b, -1)
            features.append(f)
        features = torch.stack(features, dim=1)
        out, _ = self.lstm(features)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# تحميل موديل LSTM
lstm_model = CNN_LSTM().to(device)
lstm_model.load_state_dict(torch.load(r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\data\models\lstm_model.pth"))
lstm_model.eval()

# ======================
# 3️⃣ YOLOv5 Models (FIXED)
# ======================
knife_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\bicak el dahil.v1i.yolov5pytorch\yolov5\runs\train\exp14\weights\best.pt",
    force_reload=True
)

gun_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\gun.v1i.yolov5pytorch\yolov5\runs\train\exp13\weights\best.pt",
    force_reload=True
)

# ======================
# 4️⃣ Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

sequence = []
sequence_length = 8

# ======================
# 5️⃣ Video
# ======================
cap = cv2.VideoCapture(r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\bicak el dahil.v1i.yolov5pytorch\12.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- YOLOv5 --------
    knife_results = knife_model(frame)
    gun_results = gun_model(frame)

    weapon_detected = False

    # knife
    if len(knife_results.xyxy[0]) > 0:
        weapon_detected = True

    # gun
    if len(gun_results.xyxy[0]) > 0:
        weapon_detected = True

    # -------- LSTM --------
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img)
    sequence.append(img)

    if len(sequence) == sequence_length:
        seq_tensor = torch.stack(sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            output = lstm_model(seq_tensor)
            pred = torch.argmax(output, dim=1).item()

        sequence.pop(0)
    else:
        pred = 0

    # -------- Decision --------
    if weapon_detected or pred == 1:
        color = (0,0,255)
        label = "DANGER"
    else:
        color = (0,255,0)
        label = "NORMAL"

    cv2.putText(frame, label, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Video Analysis", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO  # استيراد YOLO من ultralytics

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

# تحميل LSTM
lstm_model = CNN_LSTM().to(device)
lstm_model.load_state_dict(torch.load(
    r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\data\models\lstm_model.pth",
    map_location=device
))
lstm_model.eval()

# ======================
# 3️⃣ YOLO Models باستخدام ultralytics
# ======================
# تحميل نموذج YOLOv8 للتجزئة (Segment)
person_model = YOLO(r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\data\yolov8n-seg.pt")

# تحميل نماذج YOLOv5 للأسلحة (سنستخدم نفس الطريقة القديمة)
# ملاحظة: YOLOv5 و YOLOv8 يمكن أن يعملوا معاً
knife_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\bicak el dahil.v1i.yolov5pytorch\yolov5\runs\train\exp14\weights\best.pt",
    force_reload=False
)

gun_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\gun.v1i.yolov5pytorch\yolov5\runs\train\exp13\weights\best.pt",
    force_reload=False
)

# ضبط الثقة لنماذج YOLOv5
knife_model.conf = 0.4
gun_model.conf = 0.4

# نقل نماذج YOLOv5 إلى GPU
knife_model.to(device)
gun_model.to(device)

print("✅ Models loaded successfully")

# ======================
# 4️⃣ Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sequence_length = 8
person_sequences = {}

# ======================
# 5️⃣ Video Input/Output
# ======================
video_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\bicak el dahil.v1i.yolov5pytorch\12.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing video: {width}x{height}, FPS: {fps}")

# ======================
# 6️⃣ Processing Loop with Segmentation
# ======================
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # -------- YOLOv8 Segmentation for Person Detection --------
    results = person_model(frame, conf=0.4, device=device)  # تشغيل YOLOv8 segmentation
    
    # إنشاء نسخة من الإطار للرسم
    display_frame = frame.copy()
    
    # معالجة نتائج التجزئة
    if len(results) > 0:
        result = results[0]  # أول نتيجة (لأننا ندخل صورة واحدة)
        
        # استخراج bounding boxes والماسكات
        if result.masks is not None:
            boxes = result.boxes
            masks = result.masks
            
            # تحويل الماسكات إلى numpy array
            if masks is not None:
                masks_np = masks.data.cpu().numpy() if hasattr(masks.data, 'cpu') else masks.data.numpy()
                
                for i, (box, mask) in enumerate(zip(boxes, masks_np)):
                    # استخراج إحداثيات bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # class 0 = person in COCO dataset
                    if cls != 0:
                        continue
                    
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # التأكد من الإحداثيات
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # استخراج الـ crop باستخدام bounding box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue
                    
                    # -------- LSTM Prediction --------
                    try:
                        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        img = transform(img)
                        
                        # استخدام ID بسيط (يمكن تحسينه)
                        person_id = f"person_{x1}_{y1}_{frame_count//30}"
                        
                        if person_id not in person_sequences:
                            person_sequences[person_id] = []
                        
                        person_sequences[person_id].append(img)
                        
                        if len(person_sequences[person_id]) > sequence_length:
                            person_sequences[person_id].pop(0)
                        
                        if len(person_sequences[person_id]) == sequence_length:
                            seq_tensor = torch.stack(person_sequences[person_id]).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = lstm_model(seq_tensor)
                                pred = torch.argmax(output, dim=1).item()
                                confidence = torch.softmax(output, dim=1)[0][pred].item()
                        else:
                            pred = 0
                            confidence = 0.0
                            
                    except Exception as e:
                        print(f"Error in LSTM processing: {e}")
                        pred = 0
                        confidence = 0.0
                    
                    # -------- Weapon Detection --------
                    weapon = False
                    weapon_type = ""
                    
                    try:
                        knife_results = knife_model(crop)
                        if len(knife_results.xyxy[0]) > 0:
                            weapon = True
                            weapon_type = "KNIFE"
                        
                        gun_results = gun_model(crop)
                        if len(gun_results.xyxy[0]) > 0:
                            weapon = True
                            weapon_type = "GUN"
                            
                    except Exception as e:
                        print(f"Error in weapon detection: {e}")
                    
                    # -------- Decision & Drawing --------
                    if weapon or pred == 1:
                        color = (0, 0, 255)  # Red for danger
                        if weapon:
                            label = f"DANGER: {weapon_type}"
                        else:
                            label = "DANGER: SUSPICIOUS"
                    else:
                        color = (0, 255, 0)  # Green for normal
                        label = "NORMAL"
                    
                    # -------- رسم الماسك (Segmentation Mask) --------
                    # تغيير حجم الماسك ليتناسب مع الإطار
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # تطبيق الماسك على منطقة الشخص فقط
                    mask_region = mask_binary[y1:y2, x1:x2]
                    
                    # إنشاء overlay شفاف للتجزئة
                    overlay = display_frame.copy()
                    
                    # تلوين منطقة التجزئة حسب الحالة
                    if weapon or pred == 1:
                        # لون أحمر شفاف للخطر
                        overlay[y1:y2, x1:x2][mask_region == 1] = [0, 0, 255]
                    else:
                        # لون أخضر شفاف للطبيعي
                        overlay[y1:y2, x1:x2][mask_region == 1] = [0, 255, 0]
                    
                    # دمج الـ overlay مع الإطار الأصلي
                    cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
                    
                    # رسم bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # إضافة النص
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # عرض الثقة
                    if confidence > 0:
                        conf_text = f"LSTM: {confidence:.2f}"
                        cv2.putText(display_frame, conf_text, (x1, y2 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # إضافة عداد الإطارات
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # إضافة معلومات عن النموذج المستخدم
    cv2.putText(display_frame, "YOLOv8 Segmentation + LSTM", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # حفظ وعرض الفيديو
    out.write(display_frame)
    cv2.imshow("Advanced Security System - YOLOv8 Segmentation", display_frame)
    
    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("Manually stopped by user")
        break
    
    # عرض التقدم
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

# ======================
# 7️⃣ Cleanup
# ======================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video saved as output_segmentation.mp4")
print(f"Total frames processed: {frame_count}")
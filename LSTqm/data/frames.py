import cv2
import os

data_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\data"
frames_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate-progecr_2\lstm\frames"

categories = ["violence", "NonViolence"]

def video_to_frames(video_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(save_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_file, frame)
        count += 1
    cap.release()

for cat in categories:
    cat_path = os.path.join(data_path, cat)
    save_cat_path = os.path.join(frames_path, cat)
    for video_file in os.listdir(cat_path):
        video_file_path = os.path.join(cat_path, video_file)
        save_video_path = os.path.join(save_cat_path, video_file.split('.')[0])
        video_to_frames(video_file_path, save_video_path)
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import requests
import time

# CONFIG
DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224
TARGET_CLASSES = ["plastic", "paper", "metal", "clothes"]

# ESP32 Configuration
ESP32_IP = "172.16.61.142"  # ESP32 IP address
ROTATE_URL = f"http://{ESP32_IP}/rotate"


# MODEL LOAD
def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, checkpoint["config"]["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    class_names = list(checkpoint["class_to_idx"].keys())
    return model, class_names


model, class_names = load_model()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_image(cv2_frame):
    image = Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB))
    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    filtered = [(name, probs[i]) for i, name in enumerate(class_names)
                if name.lower() in TARGET_CLASSES]

    if not filtered: return "Unknown", 0.0

    names, scores = zip(*filtered)
    scores = np.array(scores)
    scores = scores / scores.sum()
    top_idx = np.argmax(scores)
    return names[top_idx], scores[top_idx]


# MAIN LOOP
def start_capture_mode():
    cap = cv2.VideoCapture(0)

    print("--- Capture Mode Started ---")
    print("Press 's' to Capture and Analyze")
    print("Press 'q' to Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preview Window
        cv2.imshow('Capture Mode - Press S to Scan', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\nCapturing...")
            label, confidence = classify_image(frame)
            result_text = f"Result: {label} ({confidence * 100:.1f}%)"
            print(result_text)

            if label.lower() == "plastic" and confidence > 0.7:
                print(">>> Sending Signal to ESP32...")
                try:
                    requests.get(ROTATE_URL, timeout=1)
                    print(">>> Signal Sent Successfully!")
                except:
                    print(">>> Error: Could not reach ESP32")

            result_frame = frame.copy()
            cv2.putText(result_frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('Analysis Result', result_frame)
            cv2.waitKey(2000)  # 2 sec for result showing
            cv2.destroyWindow('Analysis Result')

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_capture_mode()
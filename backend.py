import cv2
import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

# Configuration
ARDUINO_PORT = "/dev/cu.usbmodemFX2348N1"  # arduino port
arduino = serial.Serial(port=ARDUINO_PORT, baudrate=9600, timeout=.1)

DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224
TARGET_CLASSES = ["plastic", "paper", "metal", "clothes"]


# AI model Load
def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, checkpoint["config"]["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model, list(checkpoint["class_to_idx"].keys())


model, class_names = load_model()
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def start_system():
    cap = cv2.VideoCapture(0)  # USB Phone Webcam
    print("System Started. Press 's' to Scan.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('System Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # AI Inference
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_t = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_t)
                probs = F.softmax(logits, dim=1)[0].numpy()

            top_idx = np.argmax(probs)
            label = class_names[top_idx]
            conf = probs[top_idx]

            print(f"Detected: {label} ({conf * 100:.1f}%)")

            if conf > 0.7:
                print(f">>> Sending Command 'R' to Arduino for {label}...")
                try:
                    arduino.write(bytes('R', 'utf-8'))
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error sending to Arduino: {e}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_system()
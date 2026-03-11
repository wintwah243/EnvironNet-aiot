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
ARDUINO_PORT = "/dev/cu.usbmodemFX2348N1"
arduino = serial.Serial(port=ARDUINO_PORT, baudrate=9600, timeout=.1)

DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224

VALID_CLASSES = ["plastic", "paper", "metal"]

# ---------------- AI model Load ----------------
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

# ---------------- Main System ----------------
def start_system():
    cap = cv2.VideoCapture(0)
    print("System Started. Press 's' to Scan. Press 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('System Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("\nScanning Object...")
            # AI Inference
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_t = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_t)
                probs = F.softmax(logits, dim=1)[0].numpy()

            top_idx = np.argmax(probs)
            detected_label = class_names[top_idx].lower()
            conf = probs[top_idx]

            if detected_label in VALID_CLASSES:
                final_label = detected_label
            else:
                final_label = "trash"

            print(f"Result: {final_label.upper()} (Conf: {conf * 100:.1f}%)")

            if conf > 0.4:
                try:
                    if final_label == "paper":
                        print(">>> Sending 'P' (Paper - 0s Sequence)")
                        arduino.write(bytes('P', 'utf-8'))
                        time.sleep(5)

                    elif final_label == "metal":
                        print(">>> Sending 'M' (Metal - 0.6s Sequence)")
                        arduino.write(bytes('M', 'utf-8'))
                        time.sleep(5)

                    elif final_label == "plastic":
                        print(">>> Sending 'L' (Plastic - 1.9s Sequence)")
                        arduino.write(bytes('L', 'utf-8'))
                        time.sleep(5)

                    elif final_label == "trash":
                        print(">>> Sending 'T' (Trash - 2s Sequence)")
                        arduino.write(bytes('T', 'utf-8'))
                        time.sleep(5)

                    print(">>> Sequence Completed. Ready for next scan.")
                except Exception as e:
                    print(f"Error communicating with Arduino: {e}")
            else:
                print(">>> Confidence too low. Please try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_system()
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIGURATION ----------------
DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224
TARGET_CLASSES = ["plastic", "paper", "metal", "clothes"]

ESP32_IP = "172.16.61.142"
STREAM_URL = f"http://{ESP32_IP}:81/stream"
ROTATE_URL = f"http://{ESP32_IP}/rotate"


# --- AI MODEL LOAD ---
def load_model():
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, checkpoint["config"]["num_classes"])
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE).eval()
        class_names = list(checkpoint["class_to_idx"].keys())
        print("Model loaded successfully!")
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


model, class_names = load_model()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_frame(cv2_frame):
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
    if scores.sum() > 0:
        scores = scores / scores.sum()

    top_idx = np.argmax(scores)
    return names[top_idx], scores[top_idx]


# --- STREAMING LOOP ---
def start_stream():
    print(f"Connecting to ESP32 Stream: {STREAM_URL}")

    # OpenCV ကနေ ESP32 ရဲ့ Stream URL ကို တိုက်ရိုက်ချိတ်တာပါ
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Error: Could not open video stream. Check IP or WiFi.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame (stream end?). Retrying...")
            continue

        # AI Prediction လုပ်မယ်
        label, confidence = classify_frame(frame)

        # Result ကို ပုံပေါ်မှာ စာသားရေးမယ်
        text = f"{label}: {confidence * 100:.1f}%"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Monitor ပြမယ်
        cv2.imshow('EnvironNet ESP32-CAM Monitor', frame)

        # 'q' နှိပ်ရင် ပိတ်မယ်
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_stream()
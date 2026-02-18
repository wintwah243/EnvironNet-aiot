from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import io

# ---------------- CONFIG ----------------
DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224
TARGET_CLASSES = ["plastic", "paper", "metal", "clothes"]

app = FastAPI(title="EnvironNet API")

#  MODEL LOAD 
def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features, checkpoint["config"]["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()

    class_names = list(checkpoint["class_to_idx"].keys())
    return model, class_names

model, class_names = load_model()

#  TRANSFORM 
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# inference
def classify(image: Image.Image):
    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Keep only target classes
    filtered = [(name, probs[i]) for i, name in enumerate(class_names) 
                if name.lower() in TARGET_CLASSES]

    # Normalize
    names, scores = zip(*filtered)
    scores = np.array(scores)
    scores = scores / scores.sum()

    # Top class only
    top_idx = np.argmax(scores)
    return {"class": names[top_idx], "confidence": float(scores[top_idx])}

#  API ROUTE 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    top_result = classify(image)
    return top_result

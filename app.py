import streamlit as st
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
from collections import deque
import numpy as np

# config
DEVICE = "cpu"
CKPT_PATH = "environ_net.pt"
IMG_SIZE = 224

# --- NEW FILTER CONFIG ---
TARGET_CLASSES = ["plastic", "paper", "metal", "clothes"]


# -------------------------

@st.cache_resource
def load_model_and_classes():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, checkpoint["config"]["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    class_names = list(checkpoint["class_to_idx"].keys())
    return model, class_names


model, class_names = load_model_and_classes()

# Image transform
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Define your visible classes here ---
TARGET_CLASSES = ["plastic", "paper", "metal", "clothes"]


def classify_image(image: Image.Image, top_k=3):
    """
    Filters the 10-class output down to only the 4 target classes.
    """
    model.eval()
    img_t = base_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        # Get raw probabilities for all 10 classes
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        # Filter: Only keep indices that match our 4 target names
        filtered_results = []
        for i, name in enumerate(class_names):
            if name.lower() in [c.lower() for c in TARGET_CLASSES]:
                filtered_results.append((name, probs[i]))

        # Re-normalize: Make the 4 classes sum to 100% (professional look)
        names, scores = zip(*filtered_results)
        scores = np.array(scores)
        total_sum = scores.sum()

        if total_sum > 0:
            scores = scores / total_sum  # Distribute probability among the 4

        # Sort and return
        final_results = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

    return final_results[:top_k]


print([c for c in class_names if c.lower() in TARGET_CLASSES])


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Photo", "Take Photo"])

# Home
if page == "Home":
    st.title("🌍 EnvironNet")
    st.subheader("Trash Classification AI")
    st.write(
        f"EnvironNet is a deep learning model optimized to classify waste into **{len(TARGET_CLASSES)} categories**: "
        f"**{', '.join(TARGET_CLASSES).title()}.**"
    )

    st.markdown("---")
    # st.image("demo/environNet.jpg", width="stretch") # Commented out if file missing

    st.header("💡 How EnvironNet Helps")
    st.write(
        "Our AI model uses deep convolutional neural networks (CNNs) to analyze waste images. "
        "By focusing on specific recyclables like plastic and metal, we help streamline the sorting process."
    )

    st.markdown("---")
    st.header("🎯 Our Goals")
    goal_cols = st.columns(3)
    with goal_cols[0]:
        st.success("**Efficiency**\n\nFaster classification through AI integration.")
    with goal_cols[1]:
        st.success("**Impact**\n\nSupporting cleaner recycling streams.")
    with goal_cols[2]:
        st.success("**Awareness**\n\nHelping users identify household recyclables.")

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:gray;'>EnvironNet © 2025</div>", unsafe_allow_html=True)

# upload section
elif page == "Upload Photo":
    st.header(" 📤 Upload Image")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Selected Image", use_container_width=True)

        placeholder = st.empty()
        for i in range(3):
            placeholder.info(f"🔄 Scanning... {'.' * (i % 3 + 1)}")
            time.sleep(0.4)
        placeholder.empty()

        results = classify_image(image, top_k=3)
        if results:
            top_label, top_prob = results[0]
            st.success(f"✅ Predicted: **{top_label}** ({top_prob * 100:.2f}%)")
            for label, prob in results[1:]:
                st.write(f"**{label}**: {prob * 100:.2f}%")

elif page == "Take Photo":
    st.header("📸 Take Photo")
    img_file = st.camera_input("Take a picture")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captured Photo", use_container_width=True)

        placeholder = st.empty()
        for i in range(3):
            placeholder.info(f"🔄 Scanning... {'.' * (i % 3 + 1)}")
            time.sleep(0.4)
        placeholder.empty()

        results = classify_image(image, top_k=3)
        if results:
            top_label, top_prob = results[0]
            st.success(f"✅ Predicted: **{top_label}** ({top_prob * 100:.2f}%)")
            for label, prob in results[1:]:
                st.write(f"**{label}**: {prob * 100:.2f}%")
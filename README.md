# 🌍 EnvironNet - Trash Classification AI

EnvironNet is a deep learning-powered application that classifies everyday waste into **10 categories** using a MobileNetV2-based convolutional neural network with accuracy of **92.63%**. It aims to make waste management faster, smarter, and more environmentally friendly.
<img width="1920" height="1080" alt="IMG_2208" src="https://github.com/user-attachments/assets/1540d80e-fc14-43dd-9c3f-e4b1a7132ef7" />

---

## Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Plans](#future-plans)
- [License](#license)

---

## 🔹 Features
- Classifies images of waste into **10 categories**: Battery, Plastic, Shoe, Cardboard, Clothes, Metal, Organic, Glass, Paper, and Trash.
- Supports **image upload** from both phone and laptop.
- Supports **take photo** feature via device camera and laptop webcam.
- **Live webcam testing** (local only): available in `full.py` for experimentation, but not included in `app.py` since Streamlit Cloud does not support webcam streams.
- Provides **top-3 predictions** with probabilities.
- Smooth and interactive UI built with Streamlit.
- Designed for **fast inference**, even on mobile devices.

---

## 🎥 Demo

[EnvironNet Demo](https://environnet.streamlit.app/)

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yoon-thiri04/EnvironNet.git
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app locally:

```bash
streamlit run app.py
<or>
streamlit run full.py
```

> **Note:**
>
> * `app.py` → Deployment-ready (Streamlit Cloud), supports **Upload Photo** and **Take Photo**.
> * `full.py` → Local-only version with **live webcam support**. Run with:
>
>   ```bash
>   streamlit run full.py
>   ```

---

## 🚀 Usage

1. **Home Page:** Overview of EnvironNet and project goals.
2. **Upload Photo:** Upload an image from your device.
3. **Take Photo:** Capture a photo directly with your camera.
4. **Live Webcam (local only):** Real-time classification using webcam.
5. Result **top-3 predicted classes** with probabilities for each image.

---

## 📁 Project Structure

```
environnet/
│
├─ app.py                  # Deployment version (upload & take photo)
├─ full.py                 # Local version with webcam support
├─ environ_net.pt          # Trained PyTorch model
├─ requirements.txt        # Python dependencies
├─ demo/                   # Demo images for testing
│   ├─ environNet.jpg
│   ├─ metal_2389.jpg
│   ├─ glass381.jpg
│   └─ ... 
```

---

## 🛠 Technologies

* **Python**
* **Streamlit** – Frontend UI
* **PyTorch** – Deep learning framework
* **Torchvision** – Model architecture & image transforms
* **Pillow** – Image processing
* **OpenCV** – Image/video handling (for webcam in `full.py`)

---

## 🌟 Future Plans

* IoT integration: Connect EnvironNet with smart bins for automatic sorting.
* Expand classification to include recyclable vs non-recyclable categories.
* Optimize for larger datasets and faster inference.

---

## 📄 License

This project is licensed under the MIT License.


import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import gdown

# ----------------- Download Models if Not Exist -----------------

os.makedirs("models", exist_ok=True)

model_urls = {
    "GFPGANv1.4.pth": "https://drive.google.com/uc?id=1mHqzYV1X1Wn1AqGJf0sXg_J7t7hS3rB1",  # replace with actual link
    "RealESRGAN_x4.pth": "https://drive.google.com/uc?id=1bL4v0Xq2F2V1QJqG9aZcD3Yx0FvH2LkP"  # replace with actual link
}

for name, url in model_urls.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        st.write(f"Downloading {name}...")
        gdown.download(url, path, quiet=False)

# ----------------- Import AI Models -----------------

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Initialize models
gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean')
rrdbnet = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
realesrganer = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4.pth', model=rrdbnet, half=False)

# ----------------- Streamlit App -----------------

st.title("Ultra HQ Old Photo Restorer")

uploaded_file = st.file_uploader("Upload an old photo", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Original', use_column_width=True)

    if st.button("Restore Photo"):
        st.write("Restoring photo, please wait...")

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # GFPGAN face restoration
        restored_face, _ = gfpganer.enhance(img_cv, has_aligned=False)

        # Real-ESRGAN upscaling
        high_res, _ = realesrganer.enhance(restored_face, outscale=4)

        # Convert back to PIL
        restored_img = Image.fromarray(cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB))

        st.image(restored_img, caption='Restored', use_column_width=True)
        st.download_button("Download Restored Photo", data=np.array(restored_img), file_name="restored.png")


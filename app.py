# ============================================================
# 📌 STREAMLIT — Image Caption Generator (Model Selection)
# ============================================================

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import torch

# Transformers imports
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer
)

# ============================================================
# 1. APP TITLE
# ============================================================
st.title("📸 Image Caption Generator")
st.write("Upload an image → choose a model → get an AI-generated caption")

# ============================================================
# 2. MODEL SELECTION
# ============================================================
model_choice = st.selectbox(
    "Select a Captioning Model:",
    [
        "BLIP Base (Salesforce/blip-image-captioning-base)",
        "BLIP Large (Salesforce/blip-image-captioning-large)",
        "ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)"
    ]
)

# Map user-selected model
if "Base" in model_choice:
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    selected_model_type = "blip"
elif "Large" in model_choice:
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    selected_model_type = "blip"
else:
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    selected_model_type = "vit-gpt2"

# ============================================================
# 3. IMAGE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to RGB
    img_rgb = img.convert("RGB")

    # ============================================================
    # 4. CAPTION GENERATION
    # ============================================================
    st.subheader("🔍 Generating Caption...")

    if selected_model_type == "blip":
        inputs = processor(img_rgb, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    else:  # ViT-GPT2
        pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ============================================================
    # 5. DISPLAY FINAL CAPTION
    # ============================================================
    st.success("✅ Caption Generated Successfully!")
    st.markdown(f"""
        ### **📌 Final Caption for the Uploaded Image:**  
        ### 👉 **{caption}**
    """)


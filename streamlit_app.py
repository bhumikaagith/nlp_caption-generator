import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("📸 Image Caption Generator with Model Selection")

st.write("Upload an image and choose a model to generate a caption.")

# ------------------------------------------------------------
# SIDEBAR – MODEL SELECTION
# ------------------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Choose a captioning model:",
    [
        "BLIP Base (Salesforce/blip-image-captioning-base)",
        "BLIP Large (Salesforce/blip-image-captioning-large)",
        "ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)"
    ]
)

if "BLIP Base" in model_choice:
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

elif "BLIP Large" in model_choice:
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

else:
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

# ------------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_rgb = img.convert("RGB")

    st.subheader("🔍 Generating Caption...")

    # --------------------------------------------------------
    # BLIP MODELS
    # --------------------------------------------------------
    if "BLIP" in model_choice:
        inputs = processor(img_rgb, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    # --------------------------------------------------------
    # VIT-GPT2 MODEL
    # --------------------------------------------------------
    else:
        pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    st.success("✨ Caption Generated Successfully!")
    st.markdown(f"""
    ### 🟩 Final Caption:
    **{caption}**
    """)


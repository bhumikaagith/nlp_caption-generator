import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)

# -------------------------------
# Title
# -------------------------------
st.title("📸 Image Caption Generator with Model Selection")

st.write("Upload an image and choose a captioning model!")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Model Options
# -------------------------------
model_choice = st.selectbox(
    "Choose a model:",
    ("BLIP Base", "BLIP Large", "ViT-GPT2")
)

# -------------------------------
# Load Model
# -------------------------------
def load_model(choice):
    if choice == "BLIP Base":
        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        return model, processor, None

    elif choice == "BLIP Large":
        model_name = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        return model, processor, None

    else:  # ViT-GPT2
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, processor, tokenizer


# -------------------------------
# Generate Caption
# -------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.subheader("📌 Generating Caption...")

    # Load chosen model
    model, processor, tokenizer = load_model(model_choice)

    # BLIP MODELS
    if model_choice in ["BLIP Base", "BLIP Large"]:
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    # VIT-GPT2 MODEL
    else:
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # -------------------------------
    # Final Caption Display
    # -------------------------------
    st.success("✨ Caption Generated Successfully!")
    st.markdown(f"### 👉 **{caption}**")


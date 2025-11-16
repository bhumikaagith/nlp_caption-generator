# ============================================================
# 📌 Image Caption Generator - Model Selection + Caption Output
# ============================================================

print("Problem: Build a model that generates a meaningful caption for an image.\n")

# ------------------------------------------------------------
# 2. Data Collection (User Upload)
from google.colab import files
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

print("Step 2: Upload an image...")
uploaded = files.upload()

# ------------------------------------------------------------
# 3. Data Analysis
for filename in uploaded.keys():
    img = Image.open(io.BytesIO(uploaded[filename]))
    display(img)

    print("\nStep 3: Data Analysis")
    print(f"Image name: {filename}")
    print(f"Format: {img.format}")
    print(f"Mode (color type): {img.mode}")
    print(f"Size (width x height): {img.size}")

    img_array = np.array(img)

    if img.mode == "RGB":
        mean_colors = img_array.mean(axis=(0,1))
        print(f"Average RGB values: {mean_colors}")
    elif img.mode == "L":
        print(f"Average grayscale value: {img_array.mean()}")

# ------------------------------------------------------------
# 4. Data Visualization
print("\nStep 4: Data Visualization")
plt.hist(img_array.ravel(), bins=50)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()


# ------------------------------------------------------------
# ⭐ 5. MODEL SELECTION + CAPTION GENERATION
print("\nStep 5: Choose Model for Caption Generation")

print("""
Select a model:
1. BLIP Base  (Salesforce/blip-image-captioning-base)
2. BLIP Large (Salesforce/blip-image-captioning-large)
3. ViT-GPT2   (nlpconnect/vit-gpt2-image-captioning)
""")

choice = int(input("Enter model number (1/2/3): "))

# Install transformers
!pip install -q transformers
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

if choice == 1:
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
elif choice == 2:
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
elif choice == 3:
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    raise ValueError("Invalid choice!")

# Ensure image is RGB
img_rgb = img.convert("RGB")

# --------- DIFFERENT PREPROCESSING FOR DIFFERENT MODELS ---------
if choice in [1,2]:
    inputs = processor(img_rgb, return_tensors="pt")
    out = model.generate(**inputs)
    predicted_caption = processor.decode(out[0], skip_special_tokens=True)
else:  # ViT-GPT2
    pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    predicted_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n===============================================")
print("📌 FINAL CAPTION FOR THE IMAGE UPLOADED:")
print("👉", predicted_caption)
print("===============================================")

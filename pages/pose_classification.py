import streamlit as st
import os
import boto3
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline, AutoImageProcessor

bucket_name = "mlops-sentianni"
local_path = 'vit-human-pose-classification'
s3_prefix = 'ml-models/vit-human-pose-classification/'
s3 = boto3.client('s3')


def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(
                    local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)


st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
if st.button("⬇️ Download Pose Model to Classify Human Pose"):
    with st.spinner("Downloading model from S3... Please wait! This may take a few minutes."):
        try:
            download_dir(local_path, s3_prefix)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download pose model: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #27AE60;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #27AE60;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1B7F47;
        cursor: pointer;
    }
    .download-btn-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 100;
    }
    .footer {
        font-size: 0.9rem;
        color: #555;
        margin-top: 50px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Pose Classification (Human)</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or use webcam to classify human pose.</div>',
            unsafe_allow_html=True)

img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Optionally: Webcam support via st.camera_input (Streamlit >= 1.11)
if hasattr(st, 'camera_input'):
    st.markdown('Or capture from webcam:')
    webcam_img = st.camera_input("Take a picture")
else:
    webcam_img = None

image = None
if img_file is not None:
    image = Image.open(img_file)
elif webcam_img is not None:
    image = Image.open(webcam_img)

if image:
    st.image(image, caption="Selected Image", use_column_width=True)

st.markdown("---")

if st.button("Predict Pose"):
    if not os.path.exists(local_path):
        st.warning("⚠️ Model not found. Please download the model first.")
    elif image is None:
        st.warning("Please upload or capture an image.")
    else:
        try:
            device = 0 if torch.cuda.is_available() else -1
            model = "vit-human-pose-classification"
            pipe = pipeline(
                "image-classification",
                model=model,
                image_processor=AutoImageProcessor.from_pretrained(local_path),
                device=device
            )
            with st.spinner("Predicting sentiment..."):
                output = pipe(image)
            st.success("✅ Pose prediction complete!")
            label = output[0]['label']
            score = output[0]['score']

            st.markdown(f"### Pose: **{label}**")
            st.write(f"Confidence Score: {score:.3f}")
        except Exception as e:
            st.error(f"❌ Failed to run pose classification: {e}")


# Footer
st.markdown(
    """
    <div class="footer">
    Full Streamlit Code Repository: <a href="https://github.com/anupancholi/sentiment-analysis-Pose-Classification" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

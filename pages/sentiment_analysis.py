import streamlit as st
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import boto3

bucket_name = "mlops-sentianni"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'
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
if st.button("⬇️ Download Model to predict Sentiment"):
    with st.spinner("Downloading model from S3... Please wait!"):
        try:
            download_dir(local_path, s3_prefix)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2980B9;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1C5980;
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

st.markdown('<div class="title">A tinyBERT sentiment analysis model hosted with Streamlit and AWS S3</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Model Deployment at the Server</div>',
            unsafe_allow_html=True)

text = st.text_area("Enter Your Review", height=130,
                    placeholder="Type your review here...")

st.markdown("---")

if st.button("Predict Sentiment"):
    if not os.path.exists(local_path):
        st.warning("⚠️ Model not found. Please download the model first.")
    elif not text.strip():
        st.warning("Please enter some text to predict.")
    else:
        try:
            device = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                local_path)
            classifier = pipeline(
                "text-classification", model=model, tokenizer=tokenizer, device=device)
            with st.spinner("Predicting sentiment..."):
                output = classifier(text)
            st.success("✅ Prediction Complete!")
            label = output[0]['label']
            score = output[0]['score']
            st.markdown(f"### Sentiment: **{label}**")
            st.write(f"Confidence Score: {score:.3f}")
        except Exception as e:
            st.error(f"❌ Failed to run prediction: {e}")


# Footer
st.markdown(
    """
    <div class="footer">
    Full Streamlit Code Repository: <a href="https://github.com/anupancholi/08-Deploy_ML_Model_at_Streamlit_Server" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

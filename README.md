# 🤖 TinyBERT Sentiment Analysis and Pose Classification Web App

A dual-purpose machine learning web app built with **Streamlit**, showcasing:

- **Sentiment analysis** using a fine-tuned **TinyBERT** model
- **Pose classification** using a **Vision Transformer (ViT)** model

Both models are dynamically downloaded from **AWS S3**, run **fully locally**, and presented via an intuitive web interface.

---

## 🌟 Features

### 🧠 Sentiment Analysis (TinyBERT)
- 🔍 Classifies text as Positive, Negative, or Neutral
- 🤖 Powered by a fine-tuned TinyBERT model
- ✨ Fast and lightweight transformer inference

### 🧍‍♂️ Pose Classification (ViT)
- 📷 Upload or capture an image (JPG/PNG) to classify human pose
- 🖼️ Real-time image prediction using ViT
- 🔍 Top prediction label with confidence score

### 💡 Shared App Features
- ☁️ **AWS S3 Download**: Fetch models on demand
- ⚡ **Streamlit UI**: Instant feedback and predictions
- 🔐 Developer-only download button for model access
- 🌐 Deployed live on Streamlit Cloud

---

## 📷 Pose Classifier Interface

Screenshot 2025-06-20 at 17.40.41.png

> Upload or capture an image, then click "Predict Pose" to classify standing, sitting, yoga, etc.

---
## 🚀 Quick Start
[Link to the website](https://poseandsenti.streamlit.app)
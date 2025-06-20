import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="ML App Selector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

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
    .option-btn > button {
        background-color: #2980B9;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        font-size: 1.25rem;
        margin-bottom: 16px;
        margin-top: 8px;
    }
    .option-btn > button:hover {
        background-color: #1C5980;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Welcome to ML Model App</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">What would you like to do?</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Sentiment Analysis", key="sentiment", help="Analyze sentiment of text input"):
        st.switch_page(str(Path("pages/sentiment_analysis.py")))
with col2:
    if st.button("Pose Classification", key="pose", help="Classify human pose from image input"):
        st.switch_page(str(Path("pages/pose_classification.py")))

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
    Made with ❤️ by Anurodh
    </div>
    """,
    unsafe_allow_html=True,
)

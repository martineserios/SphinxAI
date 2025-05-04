import streamlit as st

st.header("🔴 Transcription")

input_video = st.file_uploader("Upload video", type=["mp4", "avi"])


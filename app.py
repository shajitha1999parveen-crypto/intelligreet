"""
IntelliGreet: Multimodal Facial and Speech Emotion Recognition for Adaptive Greetings
Streamlit entry point — deploy this on Streamlit Community Cloud.
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from emotion_utils import process_video, process_image

st.set_page_config(page_title="IntelliGreet", page_icon="👋", layout="centered")

st.title("👋 IntelliGreet")
st.caption("Multimodal facial + speech emotion recognition for adaptive greetings")

tab_video, tab_photo = st.tabs(["📹 Upload a video", "📷 Take a photo"])

# -------------------------------------------------------------------------
# Tab 1: Video upload (face + voice emotion, multi-person)
# -------------------------------------------------------------------------
with tab_video:
    st.write("Upload a short video. Faces are detected in the opening frames; "
             "audio (if present) is analyzed for voice emotion and combined "
             "with the primary speaker's facial emotion.")

    uploaded_video = st.file_uploader(
        "Choose a video file", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:
        if st.button("Analyze video", type="primary"):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_video.name)[1]
            ) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            with st.spinner("Analyzing faces and voice..."):
                try:
                    results = process_video(tmp_path)
                finally:
                    os.remove(tmp_path)

            if not results:
                st.warning("No faces detected in the opening frames.")
            else:
                for r in results:
                    with st.container(border=True):
                        st.subheader(f"Person {r['person']}")
                        st.write(f"**Detected emotion:** {r['face_emotion']}")
                        if r["note"]:
                            st.caption(r["note"])
                        st.success(r["greeting"])

# -------------------------------------------------------------------------
# Tab 2: Webcam photo (face emotion only — Streamlit Cloud has no live video)
# -------------------------------------------------------------------------
with tab_photo:
    st.write("Take a photo with your camera. Each detected face gets its own greeting.")

    camera_image = st.camera_input("Camera")

    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing faces..."):
            results = process_image(frame)

        if not results:
            st.warning("No faces detected.")
        else:
            for r in results:
                with st.container(border=True):
                    st.subheader(f"Person {r['person']}")
                    st.write(f"**Detected emotion:** {r['face_emotion']}")
                    st.success(r["greeting"])





                    
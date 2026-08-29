# IntelliGreet

Multimodal Facial and Speech Emotion Recognition for Adaptive Greetings.

Upload a video (face + voice emotion, multi-person) or take a photo
(face emotion) and get a personalized, emotion-aware greeting generated
on the fly.

## Project structure

```
intelligreet/
├── app.py             # Streamlit UI (entry point)
├── emotion_utils.py   # Core detection + greeting logic
├── requirements.txt   # Python dependencies
├── packages.txt       # System (apt) dependencies for Streamlit Cloud
└── .gitignore
```

## Run locally (VS Code)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# On Windows/Mac you may also need ffmpeg installed system-wide for moviepy.

streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io and click "New app".
3. Point it at your repo, branch, and set the main file to `app.py`.
4. Streamlit Cloud will automatically install `requirements.txt` and
   `packages.txt` before launching.
5. First launch will be slow (downloading model weights for DeepFace,
   flan-t5, and the speech-emotion model) — subsequent restarts are faster
   thanks to `st.cache_resource`.

## Notes

- Live webcam video isn't supported on Streamlit Cloud — `st.camera_input`
  captures a single photo per click instead of a live feed, which is what
  the "Take a photo" tab uses.
- DeepFace and the audio model download their weights on first use; make
  sure the deployment has enough memory (Streamlit Cloud's free tier is
  1 GB RAM, which can be tight for `torch` + `tensorflow` + `deepface`
  together — consider trimming to CPU-only wheels if you hit memory limits).
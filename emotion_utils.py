"""
emotion_utils.py
Core pipeline: face detection + emotion analysis, voice emotion, and
greeting generation. Kept separate from app.py so the Streamlit layer
stays thin and this logic is easy to test / reuse.

Heavy libraries (deepface/TensorFlow, moviepy, transformers/PyTorch) are
imported LAZILY inside the functions that need them, not at module load
time. This keeps app startup light — nothing heavy loads until the user
actually analyzes a video or photo, which matters a lot on memory-limited
hosts like Streamlit Community Cloud's free tier.
"""

import os
import gc
import cv2
import streamlit as st

# -------------------------------------------------------------------------
# Fallback greetings (used if the generative model fails)
# -------------------------------------------------------------------------
GREETINGS = {
    "happy":    "Hey there! Your smile is contagious - great to see you!",
    "sad":      "Hi... I can see things feel heavy right now. I'm here with you.",
    "angry":    "Hello. Let's take a breath together - I'm here to help, no rush.",
    "surprise": "Whoa, welcome! Something exciting going on?",
    "fear":     "Hi, it's okay - you're safe here. Let's take it one step at a time.",
    "disgust":  "Hello there - let's see how I can turn things around for you.",
    "neutral":  "Hi! Good to have you here."
}


# -------------------------------------------------------------------------
# Cached model loaders — Streamlit will only load these once per session,
# and only the first time they're actually needed.
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading greeting-generation model...")
def load_text_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    # flan-t5-small instead of flan-t5-base: ~80M params vs ~250M,
    # same feature (AI-generated greetings), much smaller footprint.
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small", low_cpu_mem_usage=True
    )
    return tokenizer, model


@st.cache_resource(show_spinner="Loading speech-emotion model...")
def load_audio_classifier():
    from transformers import pipeline
    # wav2vec2-BASE fine-tune instead of wav2vec2-large-xlsr-53: ~95M
    # params vs ~300M+ (roughly 3.5x smaller download/RAM footprint).
    return pipeline(
        "audio-classification",
        model="HaniaRuby/speech-emotion-recognition-wav2vec2"
    )


def generate_greeting(emotion):
    tokenizer, model = load_text_model()
    prompt = f"Write a short, warm, one-sentence welcome greeting for someone who looks {emotion}."
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=40, do_sample=True, temperature=0.9)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text if text else GREETINGS.get(emotion, "Hello! Welcome.")
    except Exception as e:
        print(f"Generation failed, using fallback: {e}")
        return GREETINGS.get(emotion, "Hello! Welcome.")


# -------------------------------------------------------------------------
# Frame extraction / face analysis
# -------------------------------------------------------------------------
def extract_frames(video_path, sample_rate=5, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


def analyze_all_faces(frame):
    from deepface import DeepFace
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='retinaface'
        )
        return results
    except Exception as e:
        print(f"Analysis failed: {e}")
        return []


def get_best_face_results(frames):
    """Scan opening frames, keep the one with the most successfully detected faces."""
    best_frame = None
    best_results = []
    for frame in frames:
        results = analyze_all_faces(frame)
        if len(results) > len(best_results):
            best_results = results
            best_frame = frame
    best_results_sorted = sorted(best_results, key=lambda r: r['region']['x'])
    return best_frame, best_results_sorted


# -------------------------------------------------------------------------
# Audio / voice emotion
# -------------------------------------------------------------------------
def extract_audio(video_path, audio_path="audio.wav"):
    from moviepy import VideoFileClip
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return None
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()
    return audio_path


def get_voice_emotion(video_path):
    try:
        audio_path = extract_audio(video_path)
        if audio_path is None:
            return None
        classifier = load_audio_classifier()
        audio_results = classifier(audio_path)
        os.remove(audio_path)
        return audio_results[0]['label']
    except Exception as e:
        print(f"Voice analysis failed or no audio track: {e}")
        return None


def combine_emotions(face_emotion, voice_emotion):
    if voice_emotion is None:
        return face_emotion, "voice unavailable - using face only"
    if face_emotion == voice_emotion:
        return face_emotion, "face and voice agree"
    return face_emotion, f"mixed signal (face: {face_emotion}, voice: {voice_emotion}) - face used as primary"


# -------------------------------------------------------------------------
# High-level pipelines used by app.py
# -------------------------------------------------------------------------
def process_video(video_path):
    """Returns a list of dicts: [{person, face_emotion, note, greeting}, ...]"""
    frames = extract_frames(video_path, sample_rate=5, max_frames=10)
    best_frame, people = get_best_face_results(frames)

    if not people:
        return []

    voice_emotion = get_voice_emotion(video_path)
    output = []

    for i, person in enumerate(people, start=1):
        face_emotion = person['dominant_emotion']
        if i == 1:
            final_emotion, note = combine_emotions(face_emotion, voice_emotion)
            greeting = generate_greeting(final_emotion)
        else:
            note = None
            greeting = generate_greeting(face_emotion)
        output.append({
            "person": i,
            "face_emotion": face_emotion,
            "note": note,
            "greeting": greeting,
        })

    frames.clear()
    gc.collect()
    return output


def process_image(frame):
    """Returns a list of dicts: [{person, face_emotion, greeting}, ...]"""
    people = analyze_all_faces(frame)
    people_sorted = sorted(people, key=lambda r: r['region']['x'])

    output = []
    for i, person in enumerate(people_sorted, start=1):
        emotion = person['dominant_emotion']
        greeting = generate_greeting(emotion)
        output.append({
            "person": i,
            "face_emotion": emotion,
            "greeting": greeting,
        })

    gc.collect()
    return output

# """
# emotion_utils.py
# Core pipeline: face detection + emotion analysis, voice emotion, and
# greeting generation. Kept separate from app.py so the Streamlit layer
# stays thin and this logic is easy to test / reuse.
# """

# import os
# import cv2
# import streamlit as st
# from deepface import DeepFace
# from moviepy import VideoFileClip
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# # -------------------------------------------------------------------------
# # Fallback greetings (used if the generative model fails)
# # -------------------------------------------------------------------------
# GREETINGS = {
#     "happy":    "Hey there! Your smile is contagious - great to see you!",
#     "sad":      "Hi... I can see things feel heavy right now. I'm here with you.",
#     "angry":    "Hello. Let's take a breath together - I'm here to help, no rush.",
#     "surprise": "Whoa, welcome! Something exciting going on?",
#     "fear":     "Hi, it's okay - you're safe here. Let's take it one step at a time.",
#     "disgust":  "Hello there - let's see how I can turn things around for you.",
#     "neutral":  "Hi! Good to have you here."
# }


# # -------------------------------------------------------------------------
# # Cached model loaders — Streamlit will only load these once per session
# # -------------------------------------------------------------------------
# @st.cache_resource(show_spinner="Loading greeting-generation model...")
# def load_text_model():
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#     return tokenizer, model


# @st.cache_resource(show_spinner="Loading speech-emotion model...")
# def load_audio_classifier():
#     return pipeline(
#         "audio-classification",
#         model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
#     )


# def generate_greeting(emotion):
#     tokenizer, model = load_text_model()
#     prompt = f"Write a short, warm, one-sentence welcome greeting for someone who looks {emotion}."
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt")
#         outputs = model.generate(**inputs, max_length=40, do_sample=True, temperature=0.9)
#         text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return text if text else GREETINGS.get(emotion, "Hello! Welcome.")
#     except Exception as e:
#         print(f"Generation failed, using fallback: {e}")
#         return GREETINGS.get(emotion, "Hello! Welcome.")


# # -------------------------------------------------------------------------
# # Frame extraction / face analysis
# # -------------------------------------------------------------------------
# def extract_frames(video_path, sample_rate=5, max_frames=10):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     count = 0
#     while len(frames) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % sample_rate == 0:
#             frames.append(frame)
#         count += 1
#     cap.release()
#     return frames


# def analyze_all_faces(frame):
#     try:
#         results = DeepFace.analyze(
#             frame,
#             actions=['emotion'],
#             enforce_detection=False,
#             detector_backend='retinaface'
#         )
#         return results
#     except Exception as e:
#         print(f"Analysis failed: {e}")
#         return []


# def get_best_face_results(frames):
#     """Scan opening frames, keep the one with the most successfully detected faces."""
#     best_frame = None
#     best_results = []
#     for frame in frames:
#         results = analyze_all_faces(frame)
#         if len(results) > len(best_results):
#             best_results = results
#             best_frame = frame
#     best_results_sorted = sorted(best_results, key=lambda r: r['region']['x'])
#     return best_frame, best_results_sorted


# # -------------------------------------------------------------------------
# # Audio / voice emotion
# # -------------------------------------------------------------------------
# def extract_audio(video_path, audio_path="audio.wav"):
#     clip = VideoFileClip(video_path)
#     if clip.audio is None:
#         clip.close()
#         return None
#     clip.audio.write_audiofile(audio_path, logger=None)
#     clip.close()
#     return audio_path


# def get_voice_emotion(video_path):
#     try:
#         audio_path = extract_audio(video_path)
#         if audio_path is None:
#             return None
#         classifier = load_audio_classifier()
#         audio_results = classifier(audio_path)
#         os.remove(audio_path)
#         return audio_results[0]['label']
#     except Exception as e:
#         print(f"Voice analysis failed or no audio track: {e}")
#         return None


# def combine_emotions(face_emotion, voice_emotion):
#     if voice_emotion is None:
#         return face_emotion, "voice unavailable - using face only"
#     if face_emotion == voice_emotion:
#         return face_emotion, "face and voice agree"
#     return face_emotion, f"mixed signal (face: {face_emotion}, voice: {voice_emotion}) - face used as primary"


# # -------------------------------------------------------------------------
# # High-level pipelines used by app.py
# # -------------------------------------------------------------------------
# def process_video(video_path):
#     """Returns a list of dicts: [{person, face_emotion, note, greeting}, ...]"""
#     frames = extract_frames(video_path, sample_rate=5, max_frames=10)
#     best_frame, people = get_best_face_results(frames)

#     if not people:
#         return []

#     voice_emotion = get_voice_emotion(video_path)
#     output = []

#     for i, person in enumerate(people, start=1):
#         face_emotion = person['dominant_emotion']
#         if i == 1:
#             final_emotion, note = combine_emotions(face_emotion, voice_emotion)
#             greeting = generate_greeting(final_emotion)
#         else:
#             note = None
#             greeting = generate_greeting(face_emotion)
#         output.append({
#             "person": i,
#             "face_emotion": face_emotion,
#             "note": note,
#             "greeting": greeting,
#         })
#     return output


# def process_image(frame):
#     """Returns a list of dicts: [{person, face_emotion, greeting}, ...]"""
#     people = analyze_all_faces(frame)
#     people_sorted = sorted(people, key=lambda r: r['region']['x'])

#     output = []
#     for i, person in enumerate(people_sorted, start=1):
#         emotion = person['dominant_emotion']
#         greeting = generate_greeting(emotion)
#         output.append({
#             "person": i,
#             "face_emotion": emotion,
#             "greeting": greeting,
#         })
#     return output


# # import os
# # import cv2
# # import streamlit as st
# # from deepface import DeepFace
# # from moviepy.editor import VideoFileClip
# # from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# # # -------------------------------------------------------------------------
# # # Fallback greetings (used if the generative model fails)
# # # -------------------------------------------------------------------------
# # GREETINGS = {
# #     "happy":    "Hey there! Your smile is contagious - great to see you!",
# #     "sad":      "Hi... I can see things feel heavy right now. I'm here with you.",
# #     "angry":    "Hello. Let's take a breath together - I'm here to help, no rush.",
# #     "surprise": "Whoa, welcome! Something exciting going on?",
# #     "fear":     "Hi, it's okay - you're safe here. Let's take it one step at a time.",
# #     "disgust":  "Hello there - let's see how I can turn things around for you.",
# #     "neutral":  "Hi! Good to have you here."
# # }


# # # -------------------------------------------------------------------------
# # # Cached model loaders — Streamlit will only load these once per session
# # # -------------------------------------------------------------------------
# # @st.cache_resource(show_spinner="Loading greeting-generation model...")
# # def load_text_model():
# #     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# #     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
# #     return tokenizer, model


# # @st.cache_resource(show_spinner="Loading speech-emotion model...")
# # def load_audio_classifier():
# #     return pipeline(
# #         "audio-classification",
# #         model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# #     )


# # def generate_greeting(emotion):
# #     tokenizer, model = load_text_model()
# #     prompt = f"Write a short, warm, one-sentence welcome greeting for someone who looks {emotion}."
# #     try:
# #         inputs = tokenizer(prompt, return_tensors="pt")
# #         outputs = model.generate(**inputs, max_length=40, do_sample=True, temperature=0.9)
# #         text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #         return text if text else GREETINGS.get(emotion, "Hello! Welcome.")
# #     except Exception as e:
# #         print(f"Generation failed, using fallback: {e}")
# #         return GREETINGS.get(emotion, "Hello! Welcome.")


# # # -------------------------------------------------------------------------
# # # Frame extraction / face analysis
# # # -------------------------------------------------------------------------
# # def extract_frames(video_path, sample_rate=5, max_frames=10):
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []
# #     count = 0
# #     while len(frames) < max_frames:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         if count % sample_rate == 0:
# #             frames.append(frame)
# #         count += 1
# #     cap.release()
# #     return frames


# # def analyze_all_faces(frame):
# #     try:
# #         results = DeepFace.analyze(
# #             frame,
# #             actions=['emotion'],
# #             enforce_detection=False,
# #             detector_backend='retinaface'
# #         )
# #         return results
# #     except Exception as e:
# #         print(f"Analysis failed: {e}")
# #         return []


# # def get_best_face_results(frames):
# #     """Scan opening frames, keep the one with the most successfully detected faces."""
# #     best_frame = None
# #     best_results = []
# #     for frame in frames:
# #         results = analyze_all_faces(frame)
# #         if len(results) > len(best_results):
# #             best_results = results
# #             best_frame = frame
# #     best_results_sorted = sorted(best_results, key=lambda r: r['region']['x'])
# #     return best_frame, best_results_sorted


# # # -------------------------------------------------------------------------
# # # Audio / voice emotion
# # # -------------------------------------------------------------------------
# # def extract_audio(video_path, audio_path="audio.wav"):
# #     clip = VideoFileClip(video_path)
# #     if clip.audio is None:
# #         clip.close()
# #         return None
# #     clip.audio.write_audiofile(audio_path, logger=None)
# #     clip.close()
# #     return audio_path


# # def get_voice_emotion(video_path):
# #     try:
# #         audio_path = extract_audio(video_path)
# #         if audio_path is None:
# #             return None
# #         classifier = load_audio_classifier()
# #         audio_results = classifier(audio_path)
# #         os.remove(audio_path)
# #         return audio_results[0]['label']
# #     except Exception as e:
# #         print(f"Voice analysis failed or no audio track: {e}")
# #         return None


# # def combine_emotions(face_emotion, voice_emotion):
# #     if voice_emotion is None:
# #         return face_emotion, "voice unavailable - using face only"
# #     if face_emotion == voice_emotion:
# #         return face_emotion, "face and voice agree"
# #     return face_emotion, f"mixed signal (face: {face_emotion}, voice: {voice_emotion}) - face used as primary"


# # # -------------------------------------------------------------------------
# # # High-level pipelines used by app.py
# # # -------------------------------------------------------------------------
# # def process_video(video_path):
# #     """Returns a list of dicts: [{person, face_emotion, note, greeting}, ...]"""
# #     frames = extract_frames(video_path, sample_rate=5, max_frames=10)
# #     best_frame, people = get_best_face_results(frames)

# #     if not people:
# #         return []

# #     voice_emotion = get_voice_emotion(video_path)
# #     output = []

# #     for i, person in enumerate(people, start=1):
# #         face_emotion = person['dominant_emotion']
# #         if i == 1:
# #             final_emotion, note = combine_emotions(face_emotion, voice_emotion)
# #             greeting = generate_greeting(final_emotion)
# #         else:
# #             note = None
# #             greeting = generate_greeting(face_emotion)
# #         output.append({
# #             "person": i,
# #             "face_emotion": face_emotion,
# #             "note": note,
# #             "greeting": greeting,
# #         })
# #     return output


# # def process_image(frame):
# #     """Returns a list of dicts: [{person, face_emotion, greeting}, ...]"""
# #     people = analyze_all_faces(frame)
# #     people_sorted = sorted(people, key=lambda r: r['region']['x'])

# #     output = []
# #     for i, person in enumerate(people_sorted, start=1):
# #         emotion = person['dominant_emotion']
# #         greeting = generate_greeting(emotion)
# #         output.append({
# #             "person": i,
# #             "face_emotion": emotion,
# #             "greeting": greeting,
# #         })
# #     return output
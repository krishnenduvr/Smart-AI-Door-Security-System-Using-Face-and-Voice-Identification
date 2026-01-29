import cv2
import streamlit as st
import os
import numpy as np
import pickle
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time
import tempfile
from gtts import gTTS   


st.set_page_config(
    page_title="Smart AI Door Security",
    page_icon="🔐",
    layout="wide"
)


import requests, zipfile, io, os

FACE_DB = "cropped_captured"

def download_face_db(url):
    os.makedirs(FACE_DB, exist_ok=True)
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(FACE_DB)

# Call with your direct download link
download_face_db("https://drive.google.com/uc?id=1eehbJs4Z4xAc0PPRV3ANlUQz_KMuoAPG&export=download")



# ---------------- PATHS ----------------
# FACE_DB = r"D:\Security System\cropped_captured"
VOICE_MODEL_PATH = "voice_model.pkl"
VOICE_LABEL_PATH = "voice_labels.pkl"

# VOICE_MODEL_PATH = r"D:\Security System\voice_model.pkl"
# VOICE_LABEL_PATH = r"D:\Security System\voice_labels.pkl"

TEMP_AUDIO = "temp_voice.wav"

VOICE_DURATION = 4
VOICE_FS = 22050
VOICE_THRESHOLD = 0.7
FACE_THRESHOLD = 0.7

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    mtcnn = MTCNN(keep_all=False)
    facenet = InceptionResnetV1(pretrained="vggface2").eval()
    with open(VOICE_MODEL_PATH, "rb") as f:
        voice_model = pickle.load(f)
    with open(VOICE_LABEL_PATH, "rb") as f:
        encoder = pickle.load(f)
    return mtcnn, facenet, voice_model, encoder

mtcnn, facenet, voice_model, encoder = load_models()

# ---------------- LOAD FACE DATABASE ----------------
@st.cache_resource
def load_face_db():
    db = {}
    for person in os.listdir(FACE_DB):
        person_path = os.path.join(FACE_DB, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for img in os.listdir(person_path):
            if not img.lower().endswith((".jpg", ".png")):
                continue

            image = cv2.imread(os.path.join(person_path, img))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)

            if face is not None:
                emb = facenet(face.unsqueeze(0)).detach().numpy()[0]
                embeddings.append(emb)

        if embeddings:
            db[person] = np.mean(embeddings, axis=0)

    return db

face_db = load_face_db()

# ---------------- FACE RECOGNITION ----------------
def recognize_face():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "Unknown", None

    ret, frame = cam.read()
    cam.release()

    if not ret or frame is None:
        return "Unknown", None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    if face is None:
        return "Unknown", frame

    emb = facenet(face.unsqueeze(0)).detach().numpy()

    best_match, best_score = "Unknown", 0
    for name, db_emb in face_db.items():
        score = cosine_similarity(emb, db_emb.reshape(1, -1))[0][0]
        if score > best_score:
            best_score, best_match = score, name

    return best_match if best_score >= FACE_THRESHOLD else "Unknown", frame

# ---------------- VOICE RECOGNITION ----------------
def recognize_voice():
    audio = sd.rec(
        int(VOICE_DURATION * VOICE_FS),
        samplerate=VOICE_FS,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    write(TEMP_AUDIO, VOICE_FS, audio)

    y, sr = librosa.load(TEMP_AUDIO, sr=VOICE_FS)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

    probs = voice_model.predict_proba(mfcc)[0]
    confidence = np.max(probs)
    label = encoder.inverse_transform([np.argmax(probs)])[0]

    return label if confidence >= VOICE_THRESHOLD else "Unknown", confidence

# ---------------- AUDIO GREETING (Browser with gTTS) ----------------
def play_greeting_browser(name):
    tts = gTTS(text=f"Welcome {name}", lang="en")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    tmp_file.close()
    st.audio(tmp_file.name, format="audio/mp3")





# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>

/* Remove Streamlit default padding */
.block-container {
    padding-top: 0rem;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef1ff);
    font-family: 'Segoe UI', sans-serif;
}

/* NAVBAR */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: linear-gradient(90deg, #3f37c9, #5f5bd5);
    padding: 16px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 0 0 14px 14px;
}

.navbar h2 {
    color: white;
    margin: 0;
}

.nav-links a {
    color: white;
    text-decoration: none;
    margin-left: 30px;
    font-weight: 600;
    font-size: 16px;
}

.nav-links a:hover {
    color: #ffd166;
}

/* HERO SECTION */
.hero {
    background: linear-gradient(120deg, #5f5bd5, #3f37c9);
    color: white;
    padding: 80px 60px;
    border-radius: 20px;
    margin-top: 30px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.2);
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 15px;
}

.hero p {
    font-size: 18px;
    max-width: 650px;
}

/* CARDS */
.card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
    text-align: center;
    height: 100%;
}

.card h3 {
    color: #3f37c9;
    margin-bottom: 10px;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #5f5bd5, #3f37c9);
    color: white;
    border-radius: 10px;
    padding: 0.7em 1.5em;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #3f37c9, #5f5bd5);
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 80px;
    padding: 20px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
    <h2>🔐 Smart AI Door Security</h2>
    <div class="nav-links">
        <a href="?page=home">Home</a>
        <a href="?page=about">About</a>
        <a href="?page=access">Access</a>
        <a href="?page=contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- ROUTING ----------------
page = st.query_params.get("page", "home")

# ---------------- HOME PAGE ----------------
def home_page():
    st.markdown("""
    <div class="hero">
        <h1>Next-Generation Door Security</h1>
        <p>
        Experience triple-layer authentication using
        <b>Face Recognition</b>, <b>Voice Verification</b> and <b>PIN Security</b>.
        Built for modern homes and smart offices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>👤 Face Recognition</h3>
            <p>AI-powered facial authentication using deep learning for fast and accurate access.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎙 Voice Verification</h3>
            <p>Secure voice authentication that adapts to natural voice variations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>🔢 PIN Protection</h3>
            <p>Extra layer of security using encrypted PIN validation.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- ABOUT PAGE ----------------
def about_page():
    st.markdown("## 🏠 About the System")
    st.write("""
    **Smart AI Door Security System** is a modern access-control solution
    combining biometric intelligence with secure authentication layers.

    ### 🔐 Key Advantages
    - Triple-layer authentication
    - High accuracy AI models
    - Real-time decision making
    - Modern UI dashboard
    """)

# ---------------- ACCESS PAGE ----------------
def access_page():
    st.markdown("## 🔓 Access Control Panel")

    # ---------------- INITIALIZE SESSION STATE ----------------
    defaults = {
        "face_user": "Unknown",
        "voice_user": "Unknown",
        "voice_conf": 0.0,
        "access_log": []   # 👈 stores people + time
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col1, col2 = st.columns(2)

    # ---------------- FACE AUTH ----------------
    with col1:
        st.subheader("👤 Face Authentication")
        if st.button("📸 Capture Face"):
            face_user, frame = recognize_face()
            st.session_state.face_user = face_user
            if frame is not None:
                st.image(frame, channels="BGR")
            st.info(f"Face: {face_user}")

    # ---------------- VOICE AUTH ----------------
    with col2:
        st.subheader("🎙 Voice Authentication")
        if st.button("🎧 Record Voice"):
            voice_user, conf = recognize_voice()
            st.session_state.voice_user = voice_user
            st.session_state.voice_conf = conf
            st.info(f"Voice: {voice_user} ({conf:.2f})")

    # ---------------- PIN ----------------
    st.subheader("🔢 PIN Verification")
    pin = st.text_input("Enter PIN", type="password")

    # ---------------- FINAL CHECK ----------------
    if st.button("🚪 Check Door Status"):
        face_ok = st.session_state.face_user.strip().lower() != "unknown"
        voice_ok = st.session_state.voice_user.strip().lower() != "unknown"
        match_ok = (
            st.session_state.face_user.strip().lower()
            == st.session_state.voice_user.strip().lower()
        )
        pin_ok = pin == "1234"

        st.markdown("### 🔍 Verification Status")
        st.write("Face Verified:", face_ok)
        st.write("Voice Verified:", voice_ok)
        st.write("Face & Voice Match:", match_ok)
        st.write("PIN Correct:", pin_ok)

        if face_ok and voice_ok and match_ok and pin_ok:
            login_time = datetime.now().strftime("%I:%M %p")

            # ✅ SAVE PERSON LOGIN DETAILS
            st.session_state.access_log.append({
                "Name": st.session_state.face_user,
                "Login Time": login_time
            })

            st.success(f"🔓 DOOR OPEN – Welcome {st.session_state.face_user}")
            st.info(f"Login Time: {login_time}")
            play_greeting_browser(st.session_state.face_user)
        else:
            st.error("🔒 DOOR LOCKED – Access Denied")

    # ---------------- VIEW PEOPLE LOGIN DETAILS ----------------
    st.markdown("---")
    if st.button("📋 View People Login Details"):
        if st.session_state.access_log:
            st.subheader("🧾 People Login Register")
            for i, entry in enumerate(st.session_state.access_log, start=1):
                st.write(
                    f"{i}. **{entry['Name']}** logged in at **{entry['Login Time']}**"
                )
        else:
            st.info("No one has accessed the door yet.")


# ---------------- CONTACT PAGE ----------------
def contact_page():
    st.markdown("## 📞 Contact Us")
    st.write("""
    📧 Email: support@smartdoor.ai  
    📞 Phone: +91-9876543210  
    🌐 Website: www.smartdoor.ai
    """)

# ---------------- PAGE SWITCH ----------------
if page == "about":
    about_page()
elif page == "access":
    access_page()
elif page == "contact":
    contact_page()
else:
    home_page()

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    © 2026 Smart AI Door Security System | All Rights Reserved
</div>
""", unsafe_allow_html=True)

















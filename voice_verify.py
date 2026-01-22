import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import pickle

MODEL_PATH = r"D:\Security System\voice_model.pkl"
LABEL_PATH = r"D:\Security System\voice_labels.pkl"

THRESHOLD = 0.4
SILENCE_THRESHOLD = 0.01
SILENCE_RATIO = 0.9

# Load scikit-learn model + encoder
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(LABEL_PATH, "rb") as f:
    encoder = pickle.load(f)

def record_voice(output_path="recorded_voice.wav", duration=5, sr=22050):
    input("Press ENTER to start recording...")
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    wav.write(output_path, sr, (audio * 32767).astype(np.int16))
    print(f"Saved recording to {output_path}")
    return output_path

def verify_voice(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)

    # Silence check
    rms = librosa.feature.rms(y=audio)[0]
    silence_ratio = np.sum(rms < SILENCE_THRESHOLD) / len(rms)
    if silence_ratio > SILENCE_RATIO:
        return "No speech detected", 0.0

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Predict
    preds = model.predict_proba(mfcc)[0]
    confidence = float(np.max(preds))
    label = encoder.inverse_transform([np.argmax(preds)])[0]

    if confidence >= THRESHOLD:
        return label, confidence
    else:
        return "Unknown", confidence

# Loop
while True:
    voice_file = record_voice(duration=3)
    name, conf = verify_voice(voice_file)
    print("Speaker:", name, "Confidence:", conf)

    choice = input("Press ENTER to try again or type 'q' to quit: ")
    if choice.lower() == 'q':
        break
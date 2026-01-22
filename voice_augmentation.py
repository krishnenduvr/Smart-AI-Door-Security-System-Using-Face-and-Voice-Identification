import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

INPUT_DIR = r"D:\Security System\captured_voices"
OUTPUT_DIR = r"D:\Security System\augmented_voices"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Augmentation Functions
# -----------------------------

def add_noise(audio, noise_factor=0.003):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate=rate)

def volume_scaling(audio, factor=1.2):
    return audio * factor

def background_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, 1, len(audio))
    return audio + noise_level * noise

def random_crop(audio, sr, crop_len=2.0):
    target_len = int(sr * crop_len)
    if len(audio) > target_len:
        start = np.random.randint(0, len(audio) - target_len)
        return audio[start:start+target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)))

def reverb(audio, sr, delay=0.02, decay=0.4):
    delay_samples = int(sr * delay)
    echo = np.zeros(len(audio) + delay_samples)
    echo[:len(audio)] = audio
    echo[delay_samples:] += decay * audio
    return echo[:len(audio)]

def speed_perturb(audio, factor=1.05):
    idx = np.round(np.arange(0, len(audio), factor))
    idx = idx[idx < len(audio)].astype(int)
    return audio[idx]

def bandpass_filter(audio, sr, lowcut=300, highcut=3400, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, audio)

# -----------------------------
# Augmentation Pipeline
# -----------------------------

print("Starting voice augmentation...\n")

for person in os.listdir(INPUT_DIR):
    person_path = os.path.join(INPUT_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"Processing speaker: {person}")

    out_person_dir = os.path.join(OUTPUT_DIR, person)
    os.makedirs(out_person_dir, exist_ok=True)

    for file in os.listdir(person_path):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(person_path, file)

        try:
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
            continue

        base = os.path.splitext(file)[0]

        # Original
        sf.write(os.path.join(out_person_dir, base + "_orig.wav"), audio, sr)

        # Noise
        sf.write(os.path.join(out_person_dir, base + "_noise.wav"), add_noise(audio), sr)

        # Pitch shift
        sf.write(os.path.join(out_person_dir, base + "_pitch.wav"), pitch_shift(audio, sr), sr)

        # Slow
        sf.write(os.path.join(out_person_dir, base + "_slow.wav"), time_stretch(audio, 0.9), sr)

        # Fast
        sf.write(os.path.join(out_person_dir, base + "_fast.wav"), time_stretch(audio, 1.1), sr)

        # Loud
        sf.write(os.path.join(out_person_dir, base + "_loud.wav"), volume_scaling(audio, 1.2), sr)

        # Quiet
        sf.write(os.path.join(out_person_dir, base + "_quiet.wav"), volume_scaling(audio, 0.8), sr)

        # Background noise
        sf.write(os.path.join(out_person_dir, base + "_bgnoise.wav"), background_noise(audio), sr)

        # Random crop
        sf.write(os.path.join(out_person_dir, base + "_crop.wav"), random_crop(audio, sr, 2.0), sr)

        # Reverb
        sf.write(os.path.join(out_person_dir, base + "_reverb.wav"), reverb(audio, sr), sr)

        # Speed perturbation
        sf.write(os.path.join(out_person_dir, base + "_speed.wav"), speed_perturb(audio, 1.05), sr)

        # Bandpass filter
        sf.write(os.path.join(out_person_dir, base + "_bandpass.wav"), bandpass_filter(audio, sr), sr)

        print(f"  ✅ Augmented {file}")

print("\n🎉 Voice augmentation completed successfully!")
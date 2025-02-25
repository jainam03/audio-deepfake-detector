import io
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import json
import os

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import librosa

from firebase_admin import credentials, auth, initialize_app

# =========================
# Firebase Initialization (if needed)
# =========================
with open("./firebaseadmin.json", "r") as config_file:
    firebase_config = json.load(config_file)

cred = credentials.Certificate(firebase_config)
initialize_app(cred)

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
# Set secret key as a hex string for proper session management
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())


# =========================
# Define the CNN Classifier Model (same as during training)
# =========================
class AudioClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, 1, 80, T_fixed]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)  # shape: [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # shape: [B, 64]
        x = self.fc(x)  # shape: [B, n_classes]
        return x


# =========================
# Preprocessing Function
# =========================
def preprocess_audio_file(
    audiopath, target_sample_rate=10000, max_duration=15.0, T_fixed=300
):
    """
    Loads an audio file (mp3 or wav), converts it to mono,
    resamples to target_sample_rate, trims to max_duration,
    computes an 80-bin Mel spectrogram with log transformation,
    normalizes it, and forces the time dimension to T_fixed via interpolation.
    """
    try:
        if audiopath.filename.endswith(".mp3"):
            audio, sr = librosa.load(
                io.BytesIO(audiopath.read()), sr=target_sample_rate
            )
            audiopath.seek(0)
            waveform = torch.FloatTensor(audio).unsqueeze(0)  # shape: [1, L]
        elif audiopath.filename.endswith(".wav"):
            waveform, sr = torchaudio.load(audiopath)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            return None

        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        max_samples = int(target_sample_rate * max_duration)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=80,
            n_fft=2048,
            hop_length=128,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        mel_min = mel_spec.min()
        mel_max = mel_spec.max()
        mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + 1e-9)

        # Force time dimension to T_fixed using interpolation:
        mel_spec = mel_spec.unsqueeze(0)  # [1, 1, 80, T]
        mel_spec = F.interpolate(
            mel_spec, size=(80, T_fixed), mode="bilinear", align_corners=False
        )
        mel_spec = mel_spec.squeeze(0)  # [1, 80, T_fixed]

        return mel_spec
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


# =========================
# Classification Function
# =========================
def classify_audio_clip(audiopath):
    mel_spec = preprocess_audio_file(
        audiopath, target_sample_rate=10000, max_duration=15.0, T_fixed=300
    )
    if mel_spec is None:
        return None
    mel_spec = mel_spec.unsqueeze(0)  # shape: [1, 1, 80, 300]

    model_path = os.environ.get("MODEL_PATH", "./audio_classifier.pth")
    classifier = AudioClassifier(n_classes=2)
    classifier.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    classifier.eval()

    with torch.no_grad():
        outputs = classifier(mel_spec)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    # Assuming label 0 = fake, label 1 = real
    fake_prob = probs[0] * 100
    real_prob = probs[1] * 100
    final_label = "FAKE (deepfake)" if fake_prob > real_prob else "REAL (human voice)"

    return fake_prob, real_prob, final_label


# =========================
# Flask Routes
# =========================
@app.route("/landingpage", methods=["GET", "POST"])
def landingPage():
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user_id = sign_up(email, password)
        if user_id:
            return redirect(url_for("index"))
        else:
            return redirect(url_for("signup"))
    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user_id = login(email, password)
        if user_id:
            return redirect(url_for("index"))
        else:
            return redirect(url_for("signin"))
    return render_template("signin.html")


@app.route("/signout")
def signout():
    session.clear()
    return redirect(url_for("signin"))


@app.route("/")
def index():
    return render_template("signup.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    print("upload route triggered")
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            result = classify_audio_clip(file)
            if result is not None:
                fake_prob, real_prob, final_label = result
                print(final_label)
                session["result_label"] = final_label
                session["fake_prob"] = float(fake_prob)  # Convert to Python float
                session["filename"] = file.filename  # Store filename in session
                return redirect(url_for("results"))
            else:
                return jsonify(error="Audio processing error")
        return jsonify(error="Invalid audio format")
    else:
        return render_template("index.html", error="Invalid audio file format")


@app.route("/results")
def results():
    result_label = session.get("result_label", "Unknown")
    fake_prob = session.get("fake_prob", "0")
    filename = session.get("filename", "Unknown")  # Get filename from session
    print("Results route variables:", result_label, fake_prob, filename)
    return render_template(
        "result.html", result_label=result_label, result_probability=fake_prob, filename=filename
    )


@app.route("/download_result", methods=["POST"])
def download_result():
    result_label = session.get("result_label", "Unknown")
    fake_prob = session.get("fake_prob", "0")
    result_text = f"Result: {result_label}\nFake Probability: {fake_prob}%"
    return (
        result_text,
        200,
        {
            "Content-Type": "text/plain",
            "Content-Disposition": 'attachment; filename="result.txt"',
        },
    )


@app.route("/fatal_error")
def fatal_error():
    return "<h1>OH NO! A FATAL ERROR OCCURED. PLEASE RELAUNCH EVERYTHING. :-(</h1>"


@app.route("/knowmore")
def knowmore():
    return render_template("knowmore.html")


@app.route("/aboutus")
def aboutus():
    return "<h1>Coming soon...!</h1>"


# Firebase utility functions
def sign_up(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user.uid
    except Exception as e:
        print(f"Error creating user: {e}")
        return None


def login(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except Exception as e:
        print(f"Error signing in user: {e}")
        return None


def get_user_id(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return None


if __name__ == "__main__":
    app.run(debug=True)

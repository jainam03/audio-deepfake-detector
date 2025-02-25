# from flask import Flask, render_template, request, redirect, url_for
# import io
# import base64
# import librosa
# import torch.nn.functional as F
# import torch.nn as nn
# import torchaudio
# import torch
# from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
# import os
# from firebase_admin import *
# import json
# from firebase_admin import auth
# from flask import jsonify

# # Load Firebase config from the JSON file
# with open("./firebaseadmin.json", "r") as config_file:
#     firebase_config = json.load(config_file)

# # Initialize Firebase
# cred = credentials.Certificate(firebase_config)
# initialize_app(cred)

# MEL_NORMS_PATH_ENV_VAR = "MEL_NORMS_PATH"
# MODEL_PATH_ENV_VAR = "MODEL_PATH"


# def get_file_path(env_var, default_path):
#     return os.environ.get(env_var, default_path)


# mel_norms_path = os.environ.get(MEL_NORMS_PATH_ENV_VAR, "./mel_norms.pth")
# model_path = os.environ.get(MODEL_PATH_ENV_VAR, "./model_new2.pth")

# app = Flask(__name__)


# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRNN, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.rnn(x)
#         out = self.fc(out[:, -1, :])
#         return out


# MAX_ALLOWED_DURATION = 6000


# def trim_audio(audio, max_duration):
#     if len(audio) > max_duration:
#         audio = audio[:max_duration]
#     return audio


# def load_audio(audiopath, sampling_rate=10000):
#     if audiopath is None:
#         return None

#     try:
#         if audiopath.filename.endswith(".mp3"):
#             audio, lsr = librosa.load(io.BytesIO(audiopath.read()), sr=sampling_rate)
#             audio = torch.FloatTensor(audio)
#         elif audiopath.filename.endswith(".wav"):
#             audio, lsr = torchaudio.load(audiopath)
#             audio = audio[0]
#         else:
#             return None

#         if lsr != sampling_rate:
#             audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

#         if torch.any(audio > 2) or not torch.any(audio < 0):
#             return None

#         audio.clip_(-1, 1)

#         return audio.unsqueeze(0)

#     except Exception as e:
#         print(f"Error loading the audio: {e}")
#         return None


# def classify_audio_clip(clip):
#     model1 = AudioMiniEncoderWithClassifierHead(
#         2,
#         spec_dim=1,
#         embedding_dim=512,
#         depth=5,
#         downsample_factor=2,
#         attn_blocks=4,
#         num_attn_heads=4,
#         base_channels=32,
#         dropout=0,
#         kernel_size=5,
#         distribute_zero_label=False,
#     )

#     model2 = AudioMiniEncoderWithClassifierHead(
#         2,
#         spec_dim=1,
#         embedding_dim=512,
#         depth=5,
#         downsample_factor=2,
#         attn_blocks=4,
#         num_attn_heads=4,
#         base_channels=32,
#         dropout=0,
#         kernel_size=5,
#         distribute_zero_label=False,
#     )

#     state_dict1 = torch.load(mel_norms_path, map_location=torch.device("cpu"))
#     state_dict2 = torch.load(model_path, map_location=torch.device("cpu"))

#     model1.load_state_dict(state_dict1, strict=False)
#     model2.load_state_dict(state_dict2, strict=False)

#     model1.eval()
#     model2.eval()

#     clip = clip.cpu().unsqueeze(0)
#     clip = clip.permute(0, 2, 1)
#     clip = clip.view(clip.size(0), 1, -1)

#     with torch.no_grad():
#         output1 = model1(clip)
#         output2 = model2(clip)

#     model1_weight = 0.5
#     model2_weight = 0.5

#     ensembled_output = (output1 * model1_weight) + ((output2 * model2_weight) / 2)
#     result = F.softmax(ensembled_output, dim=-1)
#     return result[0][0]


# result_probability_1 = 0


# def sign_up(email, password):
#     try:
#         user = auth.create_user(email=email, password=password)
#         return user.uid
#     except auth.AuthError as e:
#         print(f"Error creating user: {e}")
#         return None


# def login(email, password):
#     try:
#         user = auth.get_user_by_email(email, password)
#         return user.uid
#     except auth.AuthError as e:
#         print(f"Error signing in user: {e}")
#         return None


# def get_user_id(email, password):
#     try:
#         user = auth.get_user_by_email(email, password)
#         return user.uid
#     except Exception as e:
#         print(f"Error getting user ID: {e}")
#         return None


# @app.route("/landingpage", methods=["GET", "POST"])
# def landingPage():
#     return render_template("landing.html")


# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         email = request.form.get("email")
#         password = request.form.get("password")

#         user_id = sign_up(email, password)

#         if user_id:
#             return redirect(url_for("index"))
#         else:
#             return redirect(url_for("signup"))

#     return render_template("signup.html")


# @app.route("/signin", methods=["GET", "POST"])
# def signin():
#     if request.method == "POST":
#         email = request.form.get("email")
#         password = request.form.get("password")

#         user_id = login(email, password)

#         if user_id:
#             return redirect(url_for("index"))
#         else:
#             return redirect(url_for("signin"))

#     return render_template("signin.html")


# @app.route("/signout")
# def signout():
#     return "<h1>Ruko jara, kaam chaalu hai</h1>"


# @app.route("/")
# def index():
#     return render_template("signup.html")


# @app.route("/upload", methods=["GET", "POST"])
# def upload():
#     global result_probability_1
#     print("upload route triggered")

#     if request.method == "POST":
#         if "file" in request.files:

#             file = request.files["file"]
#             audio_clip = load_audio(file)

#             if audio_clip is not None:
#                 audio_duration = len(audio_clip)
#                 if audio_duration > MAX_ALLOWED_DURATION:
#                     audio_clip = trim_audio(audio_clip, MAX_ALLOWED_DURATION)

#                 audio_base64 = base64.b64encode(audio_clip.numpy().tobytes()).decode(
#                     "utf-8"
#                 )

#                 result_probability_1 = classify_audio_clip(audio_clip).item() * 100
#                 print(result_probability_1)

#             return jsonify(result_probability=result_probability_1)

#         return jsonify(error="Invalid audio format")
#     else:

#         return render_template("index.html", error="Invalid audio file format")


# @app.route("/results")
# def results():
#     global result_probability_1
#     return render_template(
#         "result.html", audio_base64="", result_probability=result_probability_1
#     )


# @app.route("/download_result", methods=["POST"])
# def download_result():
#     return "<h2>Feature under development</h2>"


# @app.route("/fatal_error")
# def fatal_error():
#     return "<h1>OH NO! A FATAL ERROR OCCURED. PLEASE RELAUNCH EVERYTHING. :-(</h1>"


# @app.route("/knowmore")
# def knowmore():
#     return render_template("knowmore.html")


# @app.route("/aboutus")
# def aboutus():
#     return "<h1>Coming soon...!</h1>"


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import io
import base64
import json
import os
import random

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import librosa
import IPython.display as ipd

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
# Set secret key (not really needed now if we're not using session, but we'll keep it)
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
    Loads an audio file (mp3 or wav), converts to mono,
    resamples to target_sample_rate, trims to max_duration,
    computes an 80-bin Mel spectrogram with log transformation,
    normalizes it, and forces the time dimension to T_fixed via interpolation.
    """
    try:
        if audiopath.filename.endswith(".mp3"):
            audio, sr = librosa.load(
                io.BytesIO(audiopath.read()), sr=target_sample_rate
            )
            waveform = torch.FloatTensor(audio).unsqueeze(0)  # [1, L]
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
        mel_spec = mel_spec.unsqueeze(0)  # shape: [1, 1, 80, T]
        mel_spec = F.interpolate(
            mel_spec, size=(80, T_fixed), mode="bilinear", align_corners=False
        )
        mel_spec = mel_spec.squeeze(0)  # now [1, 80, T_fixed]

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
    return render_template("signout.html")


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
                print("Classification result:", final_label, fake_prob)
                return render_template(
                    "result.html",
                    result_label=final_label,
                    result_probability=fake_prob,
                )
            else:
                return jsonify(error="Audio processing error")
        return jsonify(error="Invalid audio format")
    else:
        return render_template("index.html", error="Invalid audio file format")


# Remove the /results route since we're directly rendering from /upload


@app.route("/download_result", methods=["POST"])
def download_result():
    return "<h2>Feature under development</h2>"


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

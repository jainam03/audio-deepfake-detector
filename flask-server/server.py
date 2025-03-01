from flask import Flask, request, jsonify
import io
import os
import torch  # type: ignore
import torchaudio  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import librosa  # type: ignore
from flask_cors import CORS
import soundfile as sf
import numpy as np  # Ensure proper float conversions

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())


# =========================
# Audio Classifier Model
# =========================
class AudioClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout_prob=0.3):
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
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================
# Preprocessing Function
# =========================
def preprocess_audio_file(
    audio_data, sr, target_sample_rate=10000, max_duration=10.0, T_fixed=300
):
    try:
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        waveform = torch.FloatTensor(audio_data).unsqueeze(0)  # [1, L]

        # Resample if necessary
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        max_samples = int(target_sample_rate * max_duration)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=80,
            n_fft=2048,
            hop_length=56,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        mel_min = mel_spec.min()
        mel_max = mel_spec.max()
        mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + 1e-9)

        # Interpolate time dimension to T_fixed
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
def classify_audio_clip(file):
    try:
        # Read the file using soundfile instead of audioread
        audio_data, sr = sf.read(file)

        # Preprocess the audio
        mel_spec = preprocess_audio_file(audio_data, sr)
        if mel_spec is None:
            return None

        mel_spec = mel_spec.unsqueeze(0)  # shape: [1, 1, 80, 300]

        # Load the model
        model_path = os.environ.get("MODEL_PATH", "./audio_classifier.pth")
        classifier = AudioClassifier(n_classes=2)
        classifier.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        classifier.eval()

        # Get predictions
        with torch.no_grad():
            outputs = classifier(mel_spec)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        # Convert probabilities to standard Python float
        fake_prob = float(probs[0] * 100)
        real_prob = float(probs[1] * 100)
        final_label = (
            "FAKE (deepfake)" if fake_prob > real_prob else "REAL (human voice)"
        )

        return fake_prob, real_prob, final_label

    except Exception as e:
        print(f"Error in classification: {e}")
        return None


# =========================
# API Endpoint
# =========================
@app.route("/api/upload", methods=["POST"])
@app.route("/upload", methods=["POST"])
def upload():
    print("🔹 Upload route triggered")  # Debug log

    if "file" not in request.files:
        print("⚠️ No file part in request")
        return jsonify(error="No file part in request"), 400

    file = request.files["file"]
    if file.filename == "":
        print("⚠️ No selected file")
        return jsonify(error="No file selected"), 400

    print(f"✅ Received file: {file.filename}")

    # Save file temporarily to read it properly
    temp_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)  # Ensure directory exists
    file.save(temp_path)

    result = classify_audio_clip(temp_path)  # Call classification function

    # Remove temp file after processing
    os.remove(temp_path)

    if result is not None:
        fake_prob, real_prob, final_label = result
        print(f"✅ Classification result: {final_label} (Fake: {fake_prob:.2f}%)")

        return jsonify(
            {
                "label": final_label,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
            }
        )
    else:
        print("❌ Audio processing error")
        return jsonify(error="Audio processing error"), 500


if __name__ == "__main__":
    app.run(debug=True)

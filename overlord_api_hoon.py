# ==============================
# overlord api deamon
# ==============================

# FastAPI 프레임워크 임포트
import io
import json
import tempfile
import warnings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
import base64
import librosa
import librosa.display
import datetime
import os
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traceback
from fastapi.staticfiles import StaticFiles
import firebase_admin
from firebase_admin import credentials, messaging
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
# ----------------------------------------
# FastAPI 앱 인스턴스 생성
# ----------------------------------------
# app 객체는 서버의 중심이며, 모든 API 엔드포인트를 여기에 등록함
app = FastAPI(
    title="overlord API",
    description="말벌 탐색 서비스 restful api",
    version="1.0.0"
)

# ----------------------------------------
# MongoDB 클라이언트 설정
# ----------------------------------------

# MongoDB 연결 설정
client = MongoClient("mongodb://localhost:27017/")
db = client["overlord_db"]  
predictions_collection = db["predictions"]
devices_collection = db["devices"]


# 한국 표준시 (KST) 타임존 설정
KST = datetime.timezone(datetime.timedelta(hours=9))  

# 모델 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "hornet_report")
SERVER_URL = "http://34.81.221.132:8000"

# =========================
# Khivemind 모델 설정
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "Khivemind_cnn_model.keras")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")


IMG_HEIGHT = 224
IMG_WIDTH  = 224

SR         = 32768
DURATION   = 2.0
N_FFT      = 4096
HOP_LENGTH = 128
N_MELS     = 128

SEGMENT_DURATION     = 2.0
SEGMENT_HOP_DURATION = 2.0
MIN_SEGMENT_SECONDS  = 0.5

DEFAULT_CLASS_INDEX_TO_NAME = {
    0: "Non_Hornet",
    1: "Hornet",
}

BEST_PARAMS = {
    "feature_global": {
        "n_fft":                   4096,
        "hop_length":               128,   # ← 노트북 값
        "n_mels":                   128,   # ← 노트북 값
        "peak_prominence_ratio":    0.025842991585710955,
        "non_target_attenuation":   0.6605727477246268,
    },
    "Non_Hornet": {
        "apply_highpass":       True,
        "cutoff_freq":          140,
        "emphasize_ranges":     [(150, 1200, 1.0482135436851856)],
        "band_weights": [
            (150, 300,  1.2916165560219532),
            (300, 600,  1.3239227373021167),
            (600, 1200, 1.27617867427226),
            (1200, 4000, 0.9370984346464801),
        ],
        "peak_boost":           True,
        "peak_boost_strength":  1.1539567805877669,
    },
    "Hornet": {
        "apply_highpass":       True,
        "cutoff_freq":          170,
        "emphasize_ranges":     [(90, 2000, 1.1869261451531752)],
        "band_weights": [
            (90,  150,  1.3399483381874375),
            (150, 300,  1.3262993018751943),
            (300, 600,  1.3262049853268651),
            (600, 1200, 1.0244949375800423),
            (1200, 2000, 0.9196616957918887),
        ],
        "peak_boost":           True,
        "peak_boost_strength":  1.053842936750483,
    }
}

loaded_model = tf.keras.models.load_model(MODEL_PATH)

# Firebase 초기화
if not firebase_admin._apps:
    cred = credentials.Certificate(os.path.join(BASE_DIR, "firebase_credentials.json"))
    firebase_admin.initialize_app(cred)


# 이미지 디렉토리 static으로 마운트   42342
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# class 선언
# register-app-token 엔드포인트 수정
class DeviceRegisterItem(BaseModel):
    device_id:   str
    user_id:     str
    app_token:   str
    device_name: str = ""
    group:       str = ""
    is_enabled:  bool = True
# ----------------------------------------
# predict 데이터 모델 정의 (POST 요청 시)
# ----------------------------------------
class PredictItem(BaseModel):
    device_id: str                
    event_time: str             
    wav_base64: str 

class DeviceUnregisterItem(BaseModel):
    device_id: str
    
class DeviceUpdateItem(BaseModel):
    device_id:   str
    device_name: str = ""
    group:       str = ""

def load_class_index_mapping():
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        return {int(v): k for k, v in class_indices.items()}
    return DEFAULT_CLASS_INDEX_TO_NAME.copy()

CLASS_INDEX_TO_NAME = load_class_index_mapping()

# 헬퍼 함수
def highpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    if len(data) <= max(len(a), len(b)) * 3:
        return data
    return filtfilt(b, a, data)


def load_audio_from_bytes(audio_bytes: bytes, sr=SR):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, loaded_sr = librosa.load(tmp_path, sr=sr, mono=True)

        target_length = int(sr * DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]

        y = y - np.mean(y)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        return y, loaded_sr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def split_waveform_into_segments(
    y: np.ndarray,
    sr: int,
    segment_duration: float = SEGMENT_DURATION,
    hop_duration: float = SEGMENT_HOP_DURATION,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
):
    segment_length = int(sr * segment_duration)
    hop_length = int(sr * hop_duration)
    min_segment_length = int(sr * min_segment_seconds)

    segments = []

    if len(y) == 0:
        return segments

    if len(y) < segment_length:
        padded = np.pad(y, (0, segment_length - len(y)), mode="constant")
        segments.append({
            "start_sec": 0.0,
            "end_sec": len(y) / sr,
            "waveform": padded
        })
        return segments

    for start in range(0, len(y) - segment_length + 1, hop_length):
        end = start + segment_length
        seg = y[start:end]
        segments.append({
            "start_sec": start / sr,
            "end_sec": end / sr,
            "waveform": seg
        })

    last_start = ((len(y) - segment_length) // hop_length) * hop_length if len(y) >= segment_length else 0
    covered_until = last_start + segment_length

    if covered_until < len(y):
        tail = y[covered_until:]
        if len(tail) >= min_segment_length:
            padded_tail = np.pad(tail, (0, segment_length - len(tail)), mode="constant")
            segments.append({
                "start_sec": covered_until / sr,
                "end_sec": len(y) / sr,
                "waveform": padded_tail
            })

    return segments

def get_feature_params_by_class(class_name: str):
    if class_name not in BEST_PARAMS:
        raise ValueError(f"{class_name}에 대한 파라미터가 없습니다.")

    global_params = BEST_PARAMS["feature_global"]
    cls_params = BEST_PARAMS[class_name]

    return {
        "apply_highpass": cls_params["apply_highpass"],
        "cutoff_freq": cls_params["cutoff_freq"],
        "emphasize_ranges": cls_params["emphasize_ranges"],
        "band_weights": cls_params["band_weights"],
        "peak_boost": cls_params["peak_boost"],
        "peak_boost_strength": cls_params["peak_boost_strength"],
        "peak_prominence_ratio": global_params["peak_prominence_ratio"],
        "non_target_attenuation": global_params["non_target_attenuation"],
        "n_fft": global_params["n_fft"],
        "hop_length": global_params["hop_length"],
        "n_mels": global_params["n_mels"],
    }
    
def make_enhanced_spectrogram_array_from_waveform(
    y,
    sr,
    emphasize_ranges=None,
    band_weights=None,
    peak_boost=True,
    peak_boost_strength=1.25,
    peak_prominence_ratio=0.05,
    non_target_attenuation=0.85,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
):
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)
    enhanced_spec = mel_spec.copy()

    if emphasize_ranges is not None:
        enhanced_spec *= non_target_attenuation
        for fmin, fmax, gain in emphasize_ranges:
            idx = np.where((mel_freqs >= fmin) & (mel_freqs < fmax))[0]
            if len(idx) > 0:
                enhanced_spec[idx, :] = mel_spec[idx, :] * gain

    if band_weights is not None:
        for fmin, fmax, weight in band_weights:
            idx = np.where((mel_freqs >= fmin) & (mel_freqs < fmax))[0]
            if len(idx) > 0:
                enhanced_spec[idx, :] *= weight

    if peak_boost:
        freq_profile = np.mean(enhanced_spec, axis=1)
        if np.max(freq_profile) > 0:
            peaks, _ = find_peaks(
                freq_profile,
                prominence=np.max(freq_profile) * peak_prominence_ratio
            )
            for p in peaks:
                left = max(0, p - 1)
                right = min(len(freq_profile), p + 2)
                enhanced_spec[left:right, :] *= peak_boost_strength

    enhanced_spec = np.maximum(enhanced_spec, 1e-10)
    mel_spec_db = librosa.power_to_db(enhanced_spec, ref=np.max)
    
    mel_spec_db = np.clip(mel_spec_db, a_min=-80, a_max=None)

    if N_MELS == 128:
        mel_spec_db = mel_spec_db[12:96, :]
    elif N_MELS == 64:
        mel_spec_db = mel_spec_db[6:48, :]
    else:
        start_bin = int(N_MELS * 0.10)
        end_bin   = int(N_MELS * 0.75)
        mel_spec_db = mel_spec_db[start_bin:end_bin, :]

    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis=None,
        y_axis=None,
        cmap="viridis",
        ax=ax
    )
    plt.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, mel_spec_db


def make_fft_buf(y, sr):
    from scipy.ndimage import gaussian_filter1d

    fft_result   = np.fft.rfft(y)
    freqs        = np.fft.rfftfreq(len(y), d=1/sr)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(fft_result), 1e-10))

    log_freqs  = np.logspace(np.log10(50), np.log10(sr / 2), 1024)
    log_mag    = np.interp(log_freqs, freqs, magnitude_db)


    # 가우시안 블러 스무딩
    mag_smooth = gaussian_filter1d(log_mag, sigma=15)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.fill_between(log_freqs, mag_smooth, mag_smooth.min(), alpha=0.2, color="#2196F3")
    ax.plot(log_freqs, mag_smooth, color="#1565C0", linewidth=1.2)

    for fmin, fmax, label, color in [
        (90,  300,  "90-300Hz",   "#FF6B6B"),
        (300, 1200, "300-1200Hz", "#FFA726"),
    ]:
        ax.axvspan(fmin, fmax, alpha=0.1, color=color, label=label)

    ax.set_xscale("log")
    ax.set_xlim(50, sr / 2)
    ax.set_xlabel("Frequency (Hz)", fontsize=10, color="black")
    ax.set_ylabel("Magnitude (dB)", fontsize=10, color="black")
    ax.tick_params(colors="black", labelsize=8)
    ax.spines[:].set_color("#cccccc")
    ax.grid(True, alpha=0.3, color="#cccccc", linestyle="--", which="both")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000, 16000])
    ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k", "16k"], fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def make_display_spectrogram_buf(mel_spec_db, sr, hop_length=HOP_LENGTH):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    img_display = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis=None,        # ← "mel" → None (크롭 후 mel 축 정보 없음)
        cmap="magma",
        vmin=-60,
        vmax=0,
        ax=ax
    )
    fig.colorbar(img_display, ax=ax, format="%+2.0f dB", label="dB")
    ax.set_xlabel("Time (s)", fontsize=10, color="black")
    ax.set_ylabel("Mel band (cropped)", fontsize=10, color="black")  # ← 레이블도 수정
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def make_rms_buf(y, sr):
    from scipy.signal import welch

    # Welch method로 PSD 계산 (주파수별 RMS 에너지)
    freqs, psd = welch(y, fs=sr, nperseg=4096)
    rms_per_freq = np.sqrt(psd)  # PSD → RMS 스케일

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.fill_between(freqs, rms_per_freq, 0, alpha=0.2, color="#4CAF50")
    ax.plot(freqs, rms_per_freq, color="#2E7D32", linewidth=0.8)

    # 주요 주파수 구간 표시
    for fmin, fmax, label, color in [
        (90,  300,  "90-300Hz",   "#FF6B6B"),
        (300, 1200, "300-1200Hz", "#FFA726"),
    ]:
        ax.axvspan(fmin, fmax, alpha=0.1, color=color, label=label)

    ax.set_xscale("log")
    ax.set_xlim(50, sr / 2)
    ax.set_xlabel("Frequency (Hz)", fontsize=10, color="black")
    ax.set_ylabel("RMS Amplitude", fontsize=10, color="black")
    ax.tick_params(colors="black", labelsize=8)
    ax.spines[:].set_color("#cccccc")
    ax.grid(True, alpha=0.3, color="#cccccc", linestyle="--", which="both")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000, 16000])
    ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k", "16k"], fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_input_tensor_from_waveform(y, sr, class_name: str):
    params = get_feature_params_by_class(class_name)

    y_proc = y.copy()
    if params["apply_highpass"] and params["cutoff_freq"] is not None:
        y_proc = highpass_filter(y_proc, cutoff=params["cutoff_freq"], fs=sr)

    x, mel_spec_db = make_enhanced_spectrogram_array_from_waveform(
        y=y_proc,
        sr=sr,
        emphasize_ranges=params["emphasize_ranges"],
        band_weights=params["band_weights"],
        peak_boost=params["peak_boost"],
        peak_boost_strength=params["peak_boost_strength"],
        peak_prominence_ratio=params["peak_prominence_ratio"],
        non_target_attenuation=params["non_target_attenuation"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
    )
    return x, mel_spec_db


######

def save_mobile_image(spec_buf, fft_buf, rms_buf, file_path):
    MOBILE_W = 1080
    PADDING  = 40
    LABEL_H  = 70
    img_h = 900
    img_w = MOBILE_W - PADDING * 2
    MOBILE_H = PADDING * 4 + LABEL_H* 3 + img_h * 3

    canvas = Image.new("RGB", (MOBILE_W, MOBILE_H), color=(245, 245, 245))

    # ← 3개 기준으로 변경


    spec_buf.seek(0)
    fft_buf.seek(0)
    rms_buf.seek(0)
    spec_img = Image.open(spec_buf).convert("RGB").resize((img_w, img_h), Image.LANCZOS)
    fft_img  = Image.open(fft_buf).convert("RGB").resize((img_w, img_h), Image.LANCZOS)
    rms_img  = Image.open(rms_buf).convert("RGB").resize((img_w, img_h), Image.LANCZOS)

    draw = ImageDraw.Draw(canvas)
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        font_large = ImageFont.load_default()
        font_small = font_large

    # 스펙트로그램
    spec_label_y = PADDING
    spec_img_y   = spec_label_y + LABEL_H
    draw.text((PADDING, spec_label_y), "Mel Spectrogram", fill=(30, 30, 30), font=font_large)
    draw.text((PADDING, spec_label_y + 44), "Frequency vs Time  |  dB scale", fill=(100, 100, 100), font=font_small)
    canvas.paste(spec_img, (PADDING, spec_img_y))

    # 구분선 1
    div1_y = spec_img_y + img_h + PADDING // 2
    draw.line([(PADDING, div1_y), (MOBILE_W - PADDING, div1_y)], fill=(200, 200, 200), width=2)

    # FFT
    fft_label_y = spec_img_y + img_h + PADDING
    fft_img_y   = fft_label_y + LABEL_H
    draw.text((PADDING, fft_label_y), "FFT Spectrum", fill=(30, 30, 30), font=font_large)
    draw.text((PADDING, fft_label_y + 44), "Frequency (Hz) vs Magnitude (dB)", fill=(100, 100, 100), font=font_small)
    canvas.paste(fft_img, (PADDING, fft_img_y))

    # 구분선 2
    div2_y = fft_img_y + img_h + PADDING // 2
    draw.line([(PADDING, div2_y), (MOBILE_W - PADDING, div2_y)], fill=(200, 200, 200), width=2)

    # RMS
    rms_label_y = fft_img_y + img_h + PADDING
    rms_img_y   = rms_label_y + LABEL_H
    draw.text((PADDING, rms_label_y), "RMS Energy", fill=(30, 30, 30), font=font_large)
    draw.text((PADDING, rms_label_y + 44), "Time (s) vs Energy (dB)", fill=(100, 100, 100), font=font_small)
    canvas.paste(rms_img, (PADDING, rms_img_y))

    canvas.save(file_path, format="PNG")
    print(f"[INFO] 이미지 저장 완료: {file_path} ({os.path.getsize(file_path)} bytes)")
    
def predict_audio_bytes_with_khivemind(audio_bytes: bytes, model, hornet_alert_threshold: float = 0.35):
    y, sr = load_audio_from_bytes(audio_bytes, sr=SR)

    segments = split_waveform_into_segments(
        y=y,
        sr=sr,
        segment_duration=SEGMENT_DURATION,
        hop_duration=SEGMENT_HOP_DURATION,
    )

    if not segments:
        fft_buf = make_fft_buf(np.zeros(int(SR * DURATION), dtype=np.float32), SR)
        rms_buf = make_rms_buf(np.zeros(int(SR * DURATION), dtype=np.float32), SR)
        return {
            "predicted_class": "Non_Hornet",
            "confidence": 0.0,
            "hornet_score": 0.0,
            "alert": False,
            "candidate_scores": {"Non_Hornet": 0.0, "Hornet": 0.0},
            "segment_count": 0,
            "segment_results": [],
            "spec_buf": None,
            "fft_buf": fft_buf,
            "rms_buf":  rms_buf
        }

    segment_results = []
    all_segment_mean_probs = []
    best_segment_spec_buf = None
    best_segment_hornet_score = -1.0
    rms_buf = None

    for idx, segment in enumerate(segments):
        seg_y = segment["waveform"]

        probs_list = []
        per_transform_predictions = {}

        for class_name in ["Non_Hornet", "Hornet"]:
            x, mel_spec_db = build_input_tensor_from_waveform(seg_y, sr, class_name)
            probs = model.predict(x, verbose=0)[0]

            probs_dict = {
                CLASS_INDEX_TO_NAME[i]: float(probs[i])
                for i in range(len(probs))
            }

            probs_list.append(probs)
            per_transform_predictions[class_name] = probs_dict

            if class_name == "Hornet":
                hornet_spec_buf = make_display_spectrogram_buf(mel_spec_db, sr)

        mean_probs = np.mean(np.vstack(probs_list), axis=0)
        all_segment_mean_probs.append(mean_probs)

        averaged_scores = {
            CLASS_INDEX_TO_NAME[i]: float(mean_probs[i])
            for i in range(len(mean_probs))
        }

        hornet_score = averaged_scores.get("Hornet", 0.0)
        predicted_index = int(np.argmax(mean_probs))
        predicted_class = CLASS_INDEX_TO_NAME[predicted_index]

        alert = False
        if hornet_score >= hornet_alert_threshold:
            predicted_class = "Hornet"
            alert = True
        elif predicted_class == "Hornet":
            alert = True

        confidence = averaged_scores[predicted_class]

        if hornet_score > best_segment_hornet_score:
            best_segment_hornet_score = hornet_score
            best_segment_spec_buf = hornet_spec_buf

        segment_results.append({
            "segment_index": idx,
            "start_sec": round(segment["start_sec"], 3),
            "end_sec": round(segment["end_sec"], 3),
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "hornet_score": float(hornet_score),
            "alert": alert,
            "candidate_scores": averaged_scores,
            "per_transform_predictions": per_transform_predictions,
        })

    file_mean_probs = np.mean(np.vstack(all_segment_mean_probs), axis=0)
    averaged_scores = {
        CLASS_INDEX_TO_NAME[i]: float(file_mean_probs[i])
        for i in range(len(file_mean_probs))
    }

    hornet_score = averaged_scores.get("Hornet", 0.0)
    predicted_index = int(np.argmax(file_mean_probs))
    predicted_class = CLASS_INDEX_TO_NAME[predicted_index]

    any_segment_alert = any(seg["alert"] for seg in segment_results)

    if hornet_score >= 0.7:
        predicted_class = "Hornet"
        alert = True
    elif hornet_score >= 0.5:
        alert = True
    elif any_segment_alert:
        alert = True
    else:
        alert = False

    confidence = averaged_scores[predicted_class]
    fft_buf = make_fft_buf(y, sr)
    rms_buf = make_rms_buf(y, sr)
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "hornet_score": float(hornet_score),
        "alert": alert,
        "candidate_scores": averaged_scores,
        "segment_count": len(segment_results),
        "segment_results": segment_results,
        "spec_buf": best_segment_spec_buf,
        "fft_buf": fft_buf,
        "rms_buf": rms_buf,
    }

# send_fcm_notification 
def send_fcm_notification(device_id: str, event_time: str, confidence: float, image_url: str):
    device_doc = devices_collection.find_one(
        {"device_id": device_id},
        {"app_token": 1, "device_name": 1}
    )
    if not device_doc:
        print(f"[FCM] 디바이스 없음: device_id={device_id}")
        return False

    app_token = device_doc.get("app_token")
    if not app_token:
        print(f"[FCM] app_token 없음: device_id={device_id}")
        return False

    device_name = device_doc.get("device_name", device_id)

    # ← 전송 전 상세 로그
    print(f"[FCM] 전송 시도:")
    print(f"  device_id:   {device_id}")
    print(f"  device_name: {device_name}")
    print(f"  token 앞 20자: {app_token[:20]}...")
    print(f"  token 길이:  {len(app_token)}")
    print(f"  event_time:  {event_time}")
    print(f"  confidence:  {confidence}")
    print(f"  image_url:   {image_url}")

    message = messaging.Message(
        notification=messaging.Notification(
            title="말벌 감지!",
            body=f"{device_name} | {event_time} | 신뢰도 {confidence*100:.1f}%",
        ),
        data={
            "device_id":  device_id,
            "event_time": event_time,
            "confidence": str(round(confidence, 4)),
            "image_url":  image_url,
        },
        token=app_token,
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                sound="default",
                channel_id="hornet_alert",
            ),
        ),
        apns=messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(sound="default", badge=1)
            )
        ),
    )

    try:
        response = messaging.send(message)
        print(f"[FCM] 전송 성공: {response}")
        print(f"[FCM] message_id: {response}")  # ← message_id 확인
        return True
    except firebase_admin.exceptions.FirebaseError as e:
        print(f"[FCM] Firebase 에러: code={e.code}, message={e.cause}")
        print(f"[FCM] HTTP 상태: {e.http_response}")
        return False
    except Exception as e:
        print(f"[FCM] 알 수 없는 에러: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return False

# ----------------------------------------
# 기본 엔드포인트 (GET 요청)
# ----------------------------------------
@app.get("/")
def read_root():
    """
    서버 상태 확인용 기본 엔드포인트.
    브라우저나 curl로 GET 요청 시 메시지 반환.
    """
    return {"message": "overlord api 서버가 정상적으로 동작 중입니다!"}

# ----------------------------------------
# App Token 
# ----------------------------------------

# 디바이스 목록 조회 엔드포인트
@app.get("/v1/devices")
def get_devices(user_id: str):
    cursor = devices_collection.find({"user_id": user_id}, {"_id": 0})
    devices = list(cursor)
    return {"status": 200, "devices": devices}

# ----------------------------------------
#  POST 요청 (Body 데이터 받기)
# ----------------------------------------

@app.post("/v1/register-device")
async def register_device(item: DeviceRegisterItem):
    now = datetime.datetime.now(KST)
    devices_collection.update_one(
        {"device_id": item.device_id},
        {"$set": {
            "device_id":    item.device_id,
            "user_id":      item.user_id,
            "app_token":    item.app_token,
            "device_name":  item.device_name,
            "group":        item.group,
            "updated_at":   now.isoformat(),
            "is_enabled":   item.is_enabled,
        },
        "$setOnInsert": {
            "registered_at": now.isoformat(),  # 최초 등록 시각은 업데이트 안 함
        }},
        upsert=True
    )
    return {"status": "success", "message": f"디바이스 {item.device_id} 등록 완료"}

@app.post("/v1/unregister-device")
async def unregister_device(item: DeviceUnregisterItem):
    result = devices_collection.delete_one({"device_id": item.device_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"디바이스 {item.device_id}를 찾을 수 없습니다.")
    return {"status": "success", "message": f"디바이스 {item.device_id} 삭제 완료"}

@app.post("/v1/update-device")
async def update_device(item: DeviceUpdateItem):
    result = devices_collection.update_one(
        {"device_id": item.device_id},
        {"$set": {
            "device_name": item.device_name,
            "group":        item.group,
            "updated_at":   datetime.datetime.now(KST).isoformat(),
        }}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"디바이스 {item.device_id}를 찾을 수 없습니다.")
    return {"status": "success", "message": f"디바이스 {item.device_id} 정보 업데이트 완료"}

@app.post("/v1/predict")
def predict_sound(item: PredictItem):
    try:
        wav_base64 = item.wav_base64
        wav_base64 += "=" * (4 - len(wav_base64) % 4)
        audio_bytes = base64.b64decode(wav_base64)

        result = predict_audio_bytes_with_khivemind(
            audio_bytes=audio_bytes,
            model=loaded_model,
            hornet_alert_threshold=0.35,
        )

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        candidate_scores = result["candidate_scores"]
        segment_results = result["segment_results"]
        segment_count = result["segment_count"]
        hornet_score = result["hornet_score"]
        spec_buf = result["spec_buf"]
        fft_buf = result["fft_buf"]
        rms_buf = result["rms_buf"]

        is_hornet = bool(result["alert"])
        label = predicted_class

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e)) from e

    now = datetime.datetime.now(KST)
    time_str = now.strftime("%Y%m%d%H%M%S")
    prediction_seq = time_str + str(item.device_id).zfill(2)
    predict_time_str = now.strftime("%Y-%m-%dT%H:%M:%S+09:00")

    prediction_document = {
        "prediction_seq": prediction_seq,
        "device_id": item.device_id,
        "event_time": item.event_time,
        "is_hornet": is_hornet,
        "label": label,
        "confidence": confidence,
        "hornet_score": hornet_score,
        "candidate_scores": candidate_scores,
        "segment_count": segment_count,
        "segment_results": segment_results,
        "predict_time": predict_time_str,
        "duration": DURATION,
        "created_at": now.isoformat(),
    }

    image_url = None
    if is_hornet and spec_buf is not None and rms_buf is not None and fft_buf is not None:
        safe_event_time = item.event_time.replace(":", "-")
        file_name = f"{item.device_id}_{safe_event_time}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)

        save_mobile_image(spec_buf, fft_buf, rms_buf, file_path)

        image_url = f"{SERVER_URL}/images/{file_name}"
        send_fcm_notification(item.device_id, item.event_time, confidence, image_url)

        prediction_document["spectrogram_path"] = file_path
        prediction_document["image_url"] = image_url

    predictions_collection.insert_one(prediction_document)

    return {
        "status": 200,
        "prediction": {
            "prediction_seq": prediction_seq,
            "is_hornet": is_hornet,
            "label": label,
            "confidence": confidence,
            "hornet_score": hornet_score,
            "candidate_scores": candidate_scores,
            "segment_count": segment_count,
            "segment_results": segment_results,
            "event_time": item.event_time,
            "image_url": image_url,
        },
        "meta": {
            "device_id": item.device_id,
            "predict_time": predict_time_str,
            "duration": DURATION,
            "sampling": SR,
            "fft": N_FFT,
        },
    }


# ----------------------------------------
# GET /v1/predictions  (목록 조회)
# ----------------------------------------
@app.get("/v1/predictions")
def get_predictions(device_id: str, from_time: str, to_time: str):
    """
    ?device_id=1&from_time=2026-04-09T00:00:00&to_time=2026-04-10T23:59:59
    """
    cursor = predictions_collection.find(
        {
            "device_id":  device_id,
            "event_time": {"$gte": from_time, "$lte": to_time},
        },
        {"_id": 0},     # ObjectId는 직렬화 불가 → 제외
    ).sort("predict_time", 1)

    predictions = [
        {
            "prediction_seq": doc["prediction_seq"],
            "is_hornet":      doc["is_hornet"],
            "label":          doc["label"],
            "confidence":     doc["confidence"],
            "event_time":     doc["event_time"],
        }
        for doc in cursor
    ]

    return {"status": 200, "predictions": predictions}


# ----------------------------------------
#  Get 요청 (Body 데이터 받기)
# ----------------------------------------
@app.get("/v1/prediction")
def get_prediction(device_id: str, event_time: str, prediction_seq: str):
    doc = predictions_collection.find_one(
        {
            "device_id": device_id,
            "event_time": event_time,
            "prediction_seq": prediction_seq,
        },
        {"_id": 0}
    )

    if not doc:
        raise HTTPException(status_code=404, detail="예측 결과를 찾을 수 없습니다.")

    return {
        "status": 200,
        "prediction": {
            "prediction_seq": doc["prediction_seq"],
            "is_hornet": doc["is_hornet"],
            "label": doc["label"],
            "confidence": doc["confidence"],
            "hornet_score": doc.get("hornet_score"),
            "candidate_scores": doc.get("candidate_scores", {}),
            "segment_count": doc.get("segment_count", 0),
            "segment_results": doc.get("segment_results", []),
            "event_time": doc["event_time"],
            "image_url": doc.get("image_url"),
        },
        "meta": {
            "device_id": doc["device_id"],
            "predict_time": doc.get("predict_time"),
            "duration": doc.get("duration", DURATION),
            "sampling": SR,
            "fft": N_FFT,
        }
    }

@app.post("/v1/door/open")
def open_door(device_id: str, event_time: str):
    """
    POST 요청으로 문 열기 명령을 받는 엔드포인트.
    실제로는 하드웨어 제어 로직이 필요하지만, 여기서는 단순히 메시지만 반환.
    """
    print("/v1/door/open")
    return {
        "status": 200,
        "message": "소문 열기 완료",
        "device_id": device_id,
        "event_time": event_time
    }

@app.post("/v1/door/close")
def close_door(device_id: str, event_time: str):
    """
    POST 요청으로 문 닫기 명령을 받는 엔드포인트.
    실제로는 하드웨어 제어 로직이 필요하지만, 여기서는 단순히 메시지만 반환.
    """
    print("/v1/door/close")
    return {
        "status": 200,
        "message": "소문 닫기 완료",
        "device_id": device_id,
        "event_time": event_time
    }

@app.patch("/v1/devices/{device_id}/status")
def update_device_status(device_id: str, is_enabled: bool):
    """
    말벌이 활동하는 시간대에 맞춰 벌통 감지 시스템을 ON/OFF 할 수 있는 기능
    """
    devices_collection.update_one(
        {"device_id": device_id},
        {"$set": {
            "is_enabled": is_enabled,
            "updated_at": datetime.datetime.now(KST).isoformat(),
        }}
    )
    return {
        "status": 200,
        "message": f"디바이스 {device_id} 상태가 '{is_enabled}'로 업데이트되었습니다.",
        "is_enabled": is_enabled
    }

# ----------------------------------------
# 8. FastAPI 실행 (uvicorn)
# ----------------------------------------
# PyCharm에서 직접 실행하려면 아래 코드 블록을 그대로 두세요.
# 터미널에서 실행 시: uvicorn fastapi_basic:app --reload
if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="10.140.0.2", port=8000)

# ==============================
# overlord api deamon
# ==============================

# FastAPI 프레임워크 임포트
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # 입력 데이터 유효성 검사용
from typing import Optional
from pymongo import MongoClient
import base64
import json
import librosa
import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import traceback
from fastapi.staticfiles import StaticFiles
import matplotlib
from PIL import ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, messaging
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
loaded_model = tf.keras.models.load_model("./best_model.keras")

LABEL_MAP = {"hornet": 0, "non_hornet": 1, "normal": 2}
SR = 32768
DURATION = 1.5
N_FFT = 4096
HOP_LENGTH = 128
N_MELS = 126
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "hornet_spectrograms")
SERVER_URL = "http://34.81.221.132:8000"

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

# 헬퍼 함수
def highpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    if len(data) <= max(len(a), len(b)) * 3:
        return data
    return filtfilt(b, a, data)

def load_audio_from_bytes(audio_bytes: bytes, sr=SR, duration=DURATION):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, loaded_sr = librosa.load(tmp_path, sr=sr, mono=True)
        target_length = int(loaded_sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]
        return y, loaded_sr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def make_spectrogram_array(y, sr):
    from scipy.signal import find_peaks
    emphasize_ranges = [(90, 300, 1.3), (300, 1200, 1.15)]
    band_weights     = [(90,150,1.4),(150,300,1.3),(300,600,1.2),(600,1200,1.1),(1200,4000,0.9)]

    mel_spec  = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr / 2)
    enhanced  = mel_spec.copy() * 0.85

    for fmin, fmax, gain in emphasize_ranges:
        idx = np.where((mel_freqs >= fmin) & (mel_freqs < fmax))[0]
        if len(idx) > 0:
            enhanced[idx, :] = mel_spec[idx, :] * gain

    for fmin, fmax, weight in band_weights:
        idx = np.where((mel_freqs >= fmin) & (mel_freqs < fmax))[0]
        if len(idx) > 0:
            enhanced[idx, :] *= weight

    freq_profile = np.mean(enhanced, axis=1)
    if np.max(freq_profile) > 0:
        peaks, _ = find_peaks(freq_profile, prominence=np.max(freq_profile) * 0.05)
        for p in peaks:
            enhanced[max(0, p-1):min(len(freq_profile), p+2), :] *= 1.25

    mel_db = librosa.power_to_db(np.maximum(enhanced, 1e-10), ref=np.max)

    # ① 모델 입력용: viridis 유지 (학습 당시와 동일하게)
    fig_model, ax_model = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH,
                             cmap="viridis", ax=ax_model)  # ← viridis 유지
    ax_model.axis("off")
    plt.tight_layout(pad=0)
    buf_model = io.BytesIO()
    plt.savefig(buf_model, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig_model)
    buf_model.seek(0)
    img_arr = Image.open(buf_model).convert("RGB").resize((224, 224))
    model_input = np.expand_dims(np.array(img_arr).astype("float32") / 255.0, axis=0)

    # ② 저장용: magma + 축 레이블 (보기 좋게)
    fig_display, ax_display = plt.subplots(figsize=(6, 4), facecolor="white")
    ax_display.set_facecolor("white")
    img_display = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel",
        cmap="magma", vmin=-60, vmax=0,      # ← 표시용만 magma
        ax=ax_display
    )
    fig_display.colorbar(img_display, ax=ax_display, format="%+2.0f dB", label="dB")
    ax_display.set_xlabel("Time (s)", fontsize=10, color="black")
    ax_display.set_ylabel("Frequency (Hz)", fontsize=10, color="black")
    ax_display.tick_params(colors="black", labelsize=8)
    ax_display.spines[:].set_color("#cccccc")
    plt.tight_layout()
    buf_display = io.BytesIO()
    plt.savefig(buf_display, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig_display)
    buf_display.seek(0)

    return model_input, buf_display  # 모델입력(viridis), 저장용(magma) 분리

def make_fft_buf(y, sr):
    fft_result   = np.fft.rfft(y)
    freqs        = np.fft.rfftfreq(len(y), d=1/sr)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(fft_result), 1e-10))

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.fill_between(freqs, magnitude_db, magnitude_db.min(), alpha=0.2, color="#2196F3")
    ax.plot(freqs, magnitude_db, color="#1565C0", linewidth=0.8)

    # 주요 주파수 구간 표시
    for fmin, fmax, label, color in [
        (90,  300,  "90-300Hz",   "#FF6B6B"),
        (300, 1200, "300-1200Hz", "#FFA726"),
    ]:
        ax.axvspan(fmin, fmax, alpha=0.1, color=color, label=label)

    ax.set_xlim(0, sr / 2)
    ax.set_xlabel("Frequency (Hz)", fontsize=10, color="black")
    ax.set_ylabel("Magnitude (dB)", fontsize=10, color="black")
    ax.tick_params(colors="black", labelsize=8)
    ax.spines[:].set_color("#cccccc")
    ax.grid(True, alpha=0.3, color="#cccccc", linestyle="--")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # X축 주요 주파수 눈금
    ax.set_xticks([0, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["0", "500", "1k", "2k", "4k", "8k", "16k"], fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

######

def save_mobile_image(spec_buf, fft_buf, file_path):
    """스펙트로그램 + FFT를 9:16 모바일 비율로 합성 저장 (라이트모드)"""
    MOBILE_W  = 1080
    MOBILE_H  = 1920
    PADDING   = 40
    LABEL_H   = 70

    canvas = Image.new("RGB", (MOBILE_W, MOBILE_H), color=(245, 245, 245))  # 밝은 배경
    img_h  = (MOBILE_H - PADDING * 3 - LABEL_H * 2) // 2
    img_w  = MOBILE_W - PADDING * 2

    spec_buf.seek(0)
    fft_buf.seek(0)
    spec_img = Image.open(spec_buf).convert("RGB").resize((img_w, img_h), Image.LANCZOS)
    fft_img  = Image.open(fft_buf).convert("RGB").resize((img_w, img_h), Image.LANCZOS)

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

    # 구분선
    div_y = spec_img_y + img_h + PADDING // 2
    draw.line([(PADDING, div_y), (MOBILE_W - PADDING, div_y)], fill=(200, 200, 200), width=2)

    # FFT
    fft_label_y = spec_img_y + img_h + PADDING
    fft_img_y   = fft_label_y + LABEL_H
    draw.text((PADDING, fft_label_y), "FFT Spectrum", fill=(30, 30, 30), font=font_large)
    draw.text((PADDING, fft_label_y + 44), "Frequency (Hz) vs Magnitude (dB)", fill=(100, 100, 100), font=font_small)
    canvas.paste(fft_img, (PADDING, fft_img_y))

    canvas.save(file_path, format="PNG")
    print(f"[INFO] 이미지 저장 완료: {file_path} ({os.path.getsize(file_path)} bytes)")
    
def predict_sound_logic(audio_bytes: bytes, model):
    y, sr = load_audio_from_bytes(audio_bytes)

    y_normal = y.copy()
    y_hornet = highpass_filter(y.copy(), cutoff=90,  fs=sr)
    y_bee    = highpass_filter(y.copy(), cutoff=150, fs=sr)

    x_normal, spec_buf_n = make_spectrogram_array(y_normal, sr)
    x_hornet, spec_buf_h = make_spectrogram_array(y_hornet, sr)
    x_bee,    spec_buf_b = make_spectrogram_array(y_bee,    sr)

    p_normal = model.predict(x_normal, verbose=0)[0]
    p_hornet = model.predict(x_hornet, verbose=0)[0]
    p_bee    = model.predict(x_bee,    verbose=0)[0]

    candidate_scores = {
        "Bee":    float(p_bee[0]),
        "Hornet": float(p_hornet[1]),
        "Normal": float(p_normal[2]),
    }

    predicted_class = max(candidate_scores, key=candidate_scores.get)
    confidence      = candidate_scores[predicted_class]

    # 예측된 클래스에 맞는 스펙트로그램 buf 선택
    spec_buf_map = {"Bee": spec_buf_b, "Hornet": spec_buf_h, "Normal": spec_buf_n}
    spec_buf     = spec_buf_map[predicted_class]

    fft_buf = make_fft_buf(y, sr)  # 원본 오디오 기준 FFT

    return predicted_class, confidence, candidate_scores, spec_buf, fft_buf

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

    message = messaging.Message(
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
        return True
    except firebase_admin.exceptions.FirebaseError as e:
        print(f"[FCM] Firebase 에러: code={e.code}, message={e.cause}")  # ← Firebase 에러 코드
        print(f"[FCM] HTTP 상태: {e.http_response}")                      # ← HTTP 응답
        return False
    except Exception as e:
        print(f"[FCM] 알 수 없는 에러: {type(e).__name__}: {e}")
        print(traceback.format_exc())                                      # ← 전체 스택 트레이스
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
    spec_buf = None
    fft_buf  = None
    rms_buf  = None
    try:
        audio_bytes = base64.b64decode(item.wav_base64)
        predicted_class, confidence, candidate_scores, spec_buf, fft_buf = predict_sound_logic(audio_bytes, loaded_model)
        is_hornet = (predicted_class == "Hornet")
        label     = predicted_class

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e)) from e

    now              = datetime.datetime.now(KST)
    time_str         = now.strftime("%Y%m%d%H%M%S")
    prediction_seq   = time_str + str(item.device_id).zfill(2)
    predict_time_str = now.strftime("%Y-%m-%dT%H:%M:%S+09:00")
    duration = audio_bytes.__len__() / (SR * 2)  # 대략적인 duration 계산 (16-bit 오디오 가정)

    prediction_document = {
        "prediction_seq":   prediction_seq,
        "device_id":        item.device_id,
        "event_time":       item.event_time,
        "is_hornet":        is_hornet,
        "label":            label,
        "confidence":       confidence,
        "candidate_scores": candidate_scores,
        "predict_time":     predict_time_str,
        "duration":         DURATION,
        "created_at":       now.isoformat(),
    }

    image_url = None
    if is_hornet:
        file_name = f"{item.device_id}_{item.event_time}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        save_mobile_image(spec_buf, fft_buf, file_path)

        send_fcm_notification(item.device_id, item.event_time, confidence, f"{SERVER_URL}/images/{file_name}")
        image_url = f"{SERVER_URL}/images/{file_name}"
        prediction_document["spectrogram_path"] = file_path
        prediction_document["image_url"]        = image_url

    predictions_collection.insert_one(prediction_document)

    return {
        "status": 200,
        "prediction": {
            "prediction_seq":   prediction_seq,
            "is_hornet":        is_hornet,
            "label":            label,
            "confidence":       confidence,
            "candidate_scores": candidate_scores,
            "event_time":       item.event_time,
            "image_url":        image_url,       # 말벌이면 URL, 아니면 null
        },
        "meta": {
            "device_id":    item.device_id,
            "predict_time": predict_time_str,
            "duration":     DURATION,
            "sampling":     SR,
            "fft":          N_FFT,
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
    )

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
def get_prediction(device_id: str, event_time: str, prediction_seq: int ):
    """
    GET 요청 parameter 
    {overlord_api}/device_id=1&prediction_seq=2234
   
    """ 
    
    
    
    print("/v1/prediction")
    print(f"device_id={device_id}, prediction_seq={prediction_seq}")

    file_name = f"{device_id}_{event_time}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    img_url   = f"{SERVER_URL}/images/{file_name}"


    return {
        "status": 200,
        "prediction": {
                       "prediction_seq" : 2234,
                       "is_hornet": True,
                       "label": "hornet",
                       "img_url": img_url,
                       "confidence": 0.93,
                       "event_time" : "2026-04-10T23:30:00+09:00",
                       },
        "meta" : {
                  "device_id": 1234,
                  "predict_time": "2026-04-10T23:30:01+09:00",
                  "duration": 1.5,
                  "sampling": 22050,
                  "fft":4096
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


# ----------------------------------------
# 8. FastAPI 실행 (uvicorn)
# ----------------------------------------
# PyCharm에서 직접 실행하려면 아래 코드 블록을 그대로 두세요.
# 터미널에서 실행 시: uvicorn fastapi_basic:app --reload
if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="10.140.0.2", port=8000)

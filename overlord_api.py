# ==============================
# overlord api deamon
# ==============================

# FastAPI 프레임워크 임포트
from fastapi import FastAPI
from pydantic import BaseModel  # 입력 데이터 유효성 검사용
from typing import Optional

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
# predict 데이터 모델 정의 (POST 요청 시)
# ----------------------------------------
class PredictItem(BaseModel):
    device_id: str                
    event_time: str             
    wav_base64: str 
# ----------------------------------------
#  POST 요청 (Body 데이터 받기)
# ----------------------------------------
@app.post("/v1/predict")
def predict_sound(item: PredictItem):
    """
    JSON 형식의 데이터를 수신하여 서버에서 처리 후 결과 반환.
    POST 요청 (JSON Body):
    {
        "device_id": "1",
        "event_time": "2026-04-01T10:20:30",
        "wav_base64": "AEFSAFBAD..."
    }
    """
    print("/v1/predict")
    print(f"device_id={item.device_id}, event_time={item.event_time}")

    spectrogram_base64 = ""

    return {
        "status": 200,
        "prediction": { 
                       "predction_seq" : 2234, 
                       "is_hornet": True, 
                       "label": "hornet", 
                       "confidence": 0.93, 
                       "event_time" : item.event_time,
                       },
        "meta" : { 
                  "device_id": 1234, 
                  "predict_time": "2026-04-10T23:30:01+09:00", 
                  "duration": 1.5, 
                  "sampling": 22050, 
                  "fft":4096 
                  },
        "spectrogram_base64": spectrogram_base64
    }

# ----------------------------------------
#  Get 요청 (Body 데이터 받기)
# ----------------------------------------
@app.get("/v1/predictions")
def get_predictions(device_id : str, from_time: str, to_time: str):
    """
    GET 요청 parameter 
    {overlord_api}/device_id=1&from_time=2026-04-09T00:20:30&to_time=2026-04-10T23:20:30
    }
    """

    print("/v1/predictions")
    print(f"device_id={device_id}, from_time={from_time}, to_time={to_time}")

    return {
        "status": 200,
        "predictions": [
            {"predction_seq" : 2234, 
             "is_hornet": True, 
             "label": "hornet", 
             "confidence": 0.93, 
             "event_time" : "2026-04-09T24:30:00+09:00" 
             },
            {"predction_seq" : 2344, 
             "is_hornet": True, 
             "label": "hornet", 
             "confidence": 0.93, 
             "event_time" : "2026-04-10T24:30:00+09:00" 
             }
         ]
    }

# ----------------------------------------
#  Get 요청 (Body 데이터 받기)
# ----------------------------------------
@app.get("/v1/prediction")
def get_predictions(device_id: str,prediction_seq: int ):
    """
    GET 요청 parameter 
    {overlord_api}/device_id=1&prediction_seq=2234
   
    """ 
    print("/v1/prediction")
    print(f"device_id={device_id}, prediction_seq={prediction_seq}")

    spectrogram_base64 = ""


    return {
        "status": 200,
        "prediction": {
                       "predction_seq" : 2234,
                       "is_hornet": True,
                       "label": "hornet",
                       "confidence": 0.93,
                       "event_time" : "2026-04-10T23:30:00+09:00",
                       },
        "meta" : {
                  "device_id": 1234,
                  "predict_time": "2026-04-10T23:30:01+09:00",
                  "duration": 1.5,
                  "sampling": 22050,
                  "fft":4096
                  },
        "spectrogram_base64": spectrogram_base64

    }



# ----------------------------------------
# 8. FastAPI 실행 (uvicorn)
# ----------------------------------------
# PyCharm에서 직접 실행하려면 아래 코드 블록을 그대로 두세요.
# 터미널에서 실행 시: uvicorn fastapi_basic:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.140.0.2", port=8000)

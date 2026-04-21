#FROM python:3.12
# 추후 slim으로 최적화
FROM python:3.12-slim

# python env requirements 
COPY requirements.txt .
# 패키지 설치
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY overlord_api_hoon.py.py /app/overlord_api_hoon.py.py
COPY Khivemind_cnn_model.keras /app/Khivemind_cnn_model.keras

CMD ["uvicorn","overlord_api_hoon:app","--host","0.0.0.0","--port","8000","--reload"]


# -*- coding: utf-8 -*-
# 날씨로 태양광 발전량을 예측하는 인공지능 - Render(FastAPI) 배포용

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

# main.py와 같은 폴더에 있는 solar.xlsx를 읽는다
엑셀경로 = Path(__file__).parent / "solar.xlsx"

입력 = [
    "기상_일사(MJ/m2)",
    "기상_기온(°C)",
    "기상_습도(%)",
    "기상_풍속(m/s)",
    "기상_전운량(10분위)",
    "기상_현지기압(hPa)",
    "기상_강수량(mm)",
    "시각",
]
정답 = "발전량"

# 1~2단계. 데이터 읽기·정리
data = pd.read_excel(엑셀경로, sheet_name="날짜일시_옆으로통합")
data["일시"] = pd.to_datetime(data["일시"])
data = data.sort_values("일시").reset_index(drop=True)
data["시각"] = data["일시"].dt.hour
data["기상_강수량(mm)"] = data["기상_강수량(mm)"].fillna(0)
data["발전량"] = data["인버터2(kWh)"] + data["인버터3(kWh)"] + data["인버터4(kWh)"]

# 4단계. 학습/평가 나누기
나누는곳 = int(len(data) * 0.8)
while data["시각"][나누는곳] != 7:
    나누는곳 += 1
학습 = data[:나누는곳]
평가 = data[나누는곳:]

# 5단계. 표준화
scaler = StandardScaler()
X_학습 = scaler.fit_transform(학습[입력])
X_평가 = scaler.transform(평가[입력])
y_학습 = 학습[정답]
y_평가 = 평가[정답]

# 6~7단계. 모델 만들고 학습 (서버 시작할 때 한 번)
model = Sequential([
    Input(shape=(len(입력),)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
history = model.fit(
    X_학습, y_학습,
    epochs=300, batch_size=32, validation_split=0.2,
    callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
    verbose=0,
)

# 8~9단계. 평가
예측 = np.maximum(model.predict(X_평가, verbose=0).flatten(), 0)
평균오차 = float(mean_absolute_error(y_평가, 예측))
결정계수 = float(r2_score(y_평가, 예측))

# 12단계. 상관계수
상관 = data[입력[:-1]].corrwith(data[정답]).sort_values()

범위 = {이름: (float(data[이름].min()), float(data[이름].max())) for 이름 in 입력}


# ==================================================
# 웹 API
# ==================================================
app = FastAPI(title="태양광 발전량 예측 AI")


class 날씨조건(BaseModel):
    일사량: float
    기온: float
    습도: float
    풍속: float
    구름양: float
    기압: float
    강수량: float
    시각: float


@app.get("/")
def 홈():
    return {
        "안내": "/docs 에서 직접 예측해 볼 수 있습니다.",
        "학습횟수": len(history.history["loss"]),
        "평균절대오차_kWh": round(평균오차, 1),
        "결정계수_R2": round(결정계수, 2),
    }


@app.get("/상관계수")
def 상관계수():
    return {k: round(float(v), 2) for k, v in 상관.items()}


@app.post("/predict")
def 발전량_예측(조건: 날씨조건):
    값들 = [조건.일사량, 조건.기온, 조건.습도, 조건.풍속,
           조건.구름양, 조건.기압, 조건.강수량, 조건.시각]
    표 = pd.DataFrame([값들], columns=입력)

    경고 = []
    for 이름, 값 in zip(입력, 값들):
        최소, 최대 = 범위[이름]
        if 값 < 최소 or 값 > 최대:
            경고.append(f"{이름} = {값} 은(는) 학습 범위({최소} ~ {최대}) 밖이라 부정확할 수 있습니다.")

    결과 = float(model.predict(scaler.transform(표), verbose=0)[0][0])
    return {"예측발전량_kWh": round(max(결과, 0), 1), "경고": 경고}

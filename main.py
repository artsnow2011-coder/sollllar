# -*- coding: utf-8 -*-
# 날씨로 태양광 발전량을 예측하는 인공지능 - Render(FastAPI) 배포용

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Solar Power Prediction AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Weather(BaseModel):
    solar: float        # 일사량
    temp: float         # 기온
    humidity: float     # 습도
    wind: float         # 풍속
    cloud: float        # 구름양
    pressure: float     # 기압
    rain: float         # 강수량
    hour: float         # 시각


@app.get("/")
def home():
    return {
        "epochs": len(history.history["loss"]),
        "mae_kwh": round(평균오차, 1),
        "r2": round(결정계수, 2),
    }


@app.get("/correlation")
def correlation():
    return {k: round(float(v), 2) for k, v in 상관.items()}


@app.post("/predict")
def predict(w: Weather):
    값들 = [w.solar, w.temp, w.humidity, w.wind,
           w.cloud, w.pressure, w.rain, w.hour]
    표 = pd.DataFrame([값들], columns=입력)

    warnings = []
    for 이름, 값 in zip(입력, 값들):
        최소, 최대 = 범위[이름]
        if 값 < 최소 or 값 > 최대:
            warnings.append(f"{이름} = {값} 은(는) 학습 범위({최소} ~ {최대}) 밖이라 부정확할 수 있습니다.")

    결과 = float(model.predict(scaler.transform(표), verbose=0)[0][0])
    return {"prediction_kwh": round(max(결과, 0), 1), "warnings": warnings}

    결과 = float(model.predict(scaler.transform(표), verbose=0)[0][0])
    return {"예측발전량_kWh": round(max(결과, 0), 1), "경고": 경고}

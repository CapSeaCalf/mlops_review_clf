"""Заглушка під наступні лабораторні (FastAPI). Запуск: uvicorn src.api:app  (потрібні fastapi, uvicorn)."""
from fastapi import FastAPI
from pydantic import BaseModel

from predict import SentimentPredictor

app = FastAPI(title="Sentiment API")
_predictor = None


class Req(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Req):
    global _predictor
    _predictor = _predictor or SentimentPredictor()
    return _predictor.predict(req.texts)[["text", "p_positive", "label"]].to_dict("records")

# MLOps Review Sentiment Classifier

## Опис

Навчальний MLOps-проєкт: класифікатор тональності англомовних відгуків (negative / neutral / positive)
на основі FastText-ембедінгів і рекурентної нейромережі (PyTorch).
Нейтральний клас визначається за невизначеністю моделі (ентропія), а не окремим виходом.

## Production-сценарій

Автоматичне визначення тональності відгуків клієнтів із позначкою «неоднозначно» для ручної перевірки.

## Основні інструменти

- Ubuntu Server
- Python, PyTorch (CPU)
- Git / GitHub
- DVC
- Jupyter Notebook
- FastAPI
- Docker
- CI/CD
- Kubernetes
- Monitoring

## Структура проєкту

- `src/` — Python-код (модель, навчання, інференс, API)
- `data/` — дані (`raw/`, `processed/`, `samples/`)
- `datasets/` — зовнішні датасети
- `models/` — збережені моделі
- `notebooks/` — Jupyter Notebook
- `configs/` — конфігурації
- `tests/` — тести
- `docker/` — Docker-файли
- `reports/` — звіти
- `scripts/` — допоміжні скрипти

## Smoke test

```bash
python src/predict.py "doesn't have any flaws"
```
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY model/heart_model.joblib ./model/heart_model.joblib

USER 10001:10001
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]

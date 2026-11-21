FROM python:3.10-slim AS builder
WORKDIR /build
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    PORT=5000
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
COPY app/app.py .
RUN mkdir -p /app/models /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 5000
CMD ["sh", "-c", "python -m src.train_forecast --if-missing && gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 app:app"]
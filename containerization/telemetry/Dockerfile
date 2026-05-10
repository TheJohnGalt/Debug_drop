FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/telemetry/

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    usb-modeswitch \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyserial \
    fastapi \
    uvicorn \
    pymavlink \
    pynmea2

COPY containerization/telemetry/ /app/telemetry/

CMD ["python", "/app/telemetry/retranslator.py"]
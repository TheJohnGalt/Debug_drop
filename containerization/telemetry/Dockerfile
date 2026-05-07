FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/telemetry/

RUN apt-get update && apt-get install -y \
    gpsd \
    gpsd-clients \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pyserial

COPY containerization/telemetry/ /app/telemetry/

CMD ["python", "/app/telemetry/retranslator.py"]
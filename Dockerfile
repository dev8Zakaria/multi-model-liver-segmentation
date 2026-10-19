FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
COPY deploy/fetch_models.py ./deploy/
# Downloads the real ONNX files if the host cloned the repo without Git LFS.
RUN python deploy/fetch_models.py models

# Render sets PORT; 7860 is the local default.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

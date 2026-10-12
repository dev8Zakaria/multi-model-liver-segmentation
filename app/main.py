"""FastAPI server: serves the web interface and the inference API."""
import base64
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import inference

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
STATIC_DIR = os.path.join(inference.APP_DIR, "static")

app = FastAPI(title="Segmentation vasculaire hépatique", docs_url="/api/docs", redoc_url=None)


@app.middleware("http")
async def revalidate(request, call_next):
    # Browsers must revalidate (cheap, ETag-based) so a redeploy is picked up at once.
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


def _read_image(upload: UploadFile):
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Fichier trop volumineux (maximum 40 Mo).")
    if not data:
        raise HTTPException(400, "Fichier vide.")
    try:
        return inference.decode_image(data)
    except inference.InferenceError as exc:
        raise HTTPException(400, str(exc))


def _data_url(mime, payload):
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def list_models():
    models = [
        {**{k: v for k, v in m.items() if k != "file"}, "available": inference.model_available(m["id"])}
        for m in inference.REGISTRY["models"]
    ]
    return {"classes": inference.CLASSES, "models": models}


@app.post("/api/preview")
def preview(file: UploadFile = File(...)):
    """Browser-displayable JPEG of an upload (browsers cannot render TIFF)."""
    rgb = _read_image(file)
    return Response(inference.encode_preview(rgb), media_type="image/jpeg")


@app.post("/api/predict")
def predict(file: UploadFile = File(...), model: str = Form(...), mode: str = Form("auto")):
    rgb = _read_image(file)
    try:
        class_map, info = inference.segment(rgb, model, mode)
        info["image"] = _data_url("image/jpeg", inference.encode_preview(rgb))
        info["mask"] = _data_url("image/png", inference.encode_class_map(class_map))
    except inference.InferenceError as exc:
        raise HTTPException(400, str(exc))
    return info


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

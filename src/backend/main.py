"""
DL Server — TensorFlow/Keras Inference Service
Port: 8000
"""

import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schema import ResponsePredict
from inference.tf_model_inference import TFInference

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("DLServer")

# ---------------------------------------------------------------------------
# Model path — override qua environment variable trong docker-compose
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv("MODEL_PATH", "/models/dense_121_version_1.keras")

logger.info("🚀 Initializing TFInference — model_path=%s", MODEL_PATH)
infer_engine = TFInference(MODEL_PATH)
logger.info("✅ TFInference ready.")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DL Inference Server",
    description="TensorFlow/Keras CNN inference service cho phân loại buồng chuối.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal network — Gateway là caller duy nhất
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Monitoring"])
def health_check() -> dict:
    """Kiểm tra trạng thái server và model đã load."""
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "class_names": infer_engine.class_names,
    }


@app.post("/predict/dl", response_model=ResponsePredict, tags=["Inference"])
async def predict_dl(
    file: UploadFile = File(..., description="File ảnh buồng chuối"),
) -> ResponsePredict:
    """
    Nhận ảnh và chạy inference bằng TensorFlow/Keras CNN.

    Trả về label (Cut/Keep), confidence, và probabilities cho từng class.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File phải là ảnh (image/*), nhận được: {file.content_type}",
        )

    logger.info("📨 Received predict request — file=%s", file.filename)

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File ảnh bị rỗng")

    output = infer_engine.predict(image_bytes)

    if output.get("status") == "failed":
        logger.error("❌ Inference failed: %s", output.get("error"))
        raise HTTPException(status_code=500, detail=output.get("error", "Inference failed"))

    logger.info(
        "✅ DL Prediction — label=%s | confidence=%.4f",
        output["label"],
        output["confidence"],
    )

    return ResponsePredict(
        label=output["label"],
        confidence=output["confidence"],
        probabilities=output["probabilities"],
    )
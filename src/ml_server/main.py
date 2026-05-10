"""
ML Server — Sklearn Inference Service
Port: 8001
"""

import logging
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference.ml_model_inference import VALID_MODELS, ImagePredictor
from ml_server.schema import MLPredictResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MLServer")

# ---------------------------------------------------------------------------
# Paths từ environment variables (override trong docker-compose)
# ---------------------------------------------------------------------------
MODELS_DIR: str = os.getenv("MODELS_DIR", "/models")
SCALER_PATH: str = os.getenv("SCALER_PATH", "/models/scaler.pkl")
PCA_PATH: str = os.getenv("PCA_PATH", "/models/pca.pkl")

# ---------------------------------------------------------------------------
# Khởi tạo predictor — load tất cả models 1 lần duy nhất khi startup
# ---------------------------------------------------------------------------
logger.info("🚀 Initializing ImagePredictor (loading all sklearn models)...")
predictor = ImagePredictor(
    models_dir=MODELS_DIR,
    scaler_path=SCALER_PATH,
    pca_path=PCA_PATH,
)
logger.info("✅ ImagePredictor ready. Loaded models: %s", list(predictor.models.keys()))

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ML Inference Server",
    description="Sklearn inference service sử dụng HOG + LBP + Color features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gateway sẽ là caller duy nhất (internal network)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Monitoring"])
def health_check() -> dict:
    """Kiểm tra trạng thái server và danh sách models đã load."""
    return {
        "status": "ok",
        "loaded_models": list(predictor.models.keys()),
        "scaler_loaded": predictor.scaler is not None,
        "pca_loaded": predictor.pca is not None,
    }


@app.post("/predict/ml", response_model=MLPredictResponse, tags=["Inference"])
async def predict_ml(
    file: UploadFile = File(..., description="File ảnh buồng chuối"),
    model_name: str = Query(
        default="best_svm",
        description=f"Tên model cần dùng. Hợp lệ: {VALID_MODELS}",
    ),
) -> MLPredictResponse:
    """
    Nhận ảnh và chạy inference bằng sklearn model được chỉ định.

    - **file**: Ảnh định dạng JPG/PNG
    - **model_name**: Một trong `best_svm`, `best_xgboost`, `best_random_forest`, `best_histgradient`
    """
    # Validate content type
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File phải là ảnh (image/*), nhận được: {file.content_type}",
        )

    # Validate model name
    if model_name not in VALID_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"model_name '{model_name}' không hợp lệ. Hợp lệ: {VALID_MODELS}",
        )

    logger.info("📨 Received predict request — model=%s | file=%s", model_name, file.filename)

    # Đọc bytes từ upload
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File ảnh bị rỗng")

    # Inference
    result = predictor.predict(image_bytes=image_bytes, model_name=model_name)

    if result.get("status") == "failed":
        logger.error("❌ Inference failed: %s", result.get("error"))
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))

    return MLPredictResponse(**result)

"""
API Gateway — Routing & Forwarding Service
Port: 8080

Điều hướng:
  POST /predict/dl  → dl_server:8000/predict/dl
  POST /predict/ml  → ml_server:8001/predict/ml
"""

import logging
import os
import time

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict , List , Optional
from gateway.schema import HealthStatus, PredictResponse

VALID_MODELS: List[str] = [
    "best_svm",
    "best_xgboost",
    "best_random_forest",
    "best_histgradient",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Gateway")

# ---------------------------------------------------------------------------
# Downstream server URLs (override trong docker-compose)
# ---------------------------------------------------------------------------
DL_SERVER_URL: str = os.getenv("DL_SERVER_URL", "http://dl_server:8000")
ML_SERVER_URL: str = os.getenv("ML_SERVER_URL", "http://ml_server:8001")

# Timeout config (seconds)
REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "60.0"))

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Gateway",
    description="Gateway trung gian điều hướng request đến DL Server (TensorFlow) hoặc ML Server (Sklearn).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        'https://cs-406-bunch-banana-classification.vercel.app',
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared async HTTP client (reused across requests)
# ---------------------------------------------------------------------------
http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global http_client
    http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    logger.info("🚀 Gateway started. DL=%s | ML=%s", DL_SERVER_URL, ML_SERVER_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if http_client:
        await http_client.aclose()
    logger.info("🛑 Gateway shutdown complete.")


# ---------------------------------------------------------------------------
# Helper: forward multipart file đến downstream server
# ---------------------------------------------------------------------------
async def _forward_image(
    target_url: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    extra_params: dict | None = None,
) -> dict:
    """
    Forward ảnh đến downstream server và trả về JSON response.

    Args:
        target_url: URL đầy đủ của downstream endpoint.
        image_bytes: Raw bytes của ảnh.
        filename: Tên file gốc.
        content_type: MIME type (vd: image/jpeg).
        extra_params: Query params bổ sung.

    Returns:
        Parsed JSON từ downstream server.

    Raises:
        HTTPException: Nếu downstream trả về lỗi hoặc không kết nối được.
    """
    if http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client chưa được khởi tạo")

    files = {"file": (filename, image_bytes, content_type)}

    try:
        response = await http_client.post(
            target_url,
            files=files,
            params=extra_params or {},
        )
        response.raise_for_status()
        return response.json()

    except httpx.ConnectError:
        logger.error("❌ Cannot connect to downstream: %s", target_url)
        raise HTTPException(
            status_code=503,
            detail=f"Không thể kết nối đến inference server: {target_url}",
        )
    except httpx.TimeoutException:
        logger.error("❌ Timeout when forwarding to: %s", target_url)
        raise HTTPException(
            status_code=504,
            detail=f"Inference server timeout sau {REQUEST_TIMEOUT}s",
        )
    except httpx.HTTPStatusError as exc:
        logger.error(
            "❌ Downstream returned error %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.json().get("detail", exc.response.text),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthStatus, tags=["Monitoring"])
async def health_check() -> HealthStatus:
    """Kiểm tra trạng thái Gateway và 2 downstream servers."""
    if http_client is None:
        raise HTTPException(status_code=503, detail="Gateway chưa sẵn sàng")

    async def _ping(url: str) -> str:
        try:
            resp = await http_client.get(f"{url}/health", timeout=5.0)
            return "ok" if resp.status_code == 200 else f"error:{resp.status_code}"
        except Exception as exc:
            return f"unreachable:{exc}"

    dl_status = await _ping(DL_SERVER_URL)
    ml_status = await _ping(ML_SERVER_URL)

    logger.info("🏥 Health — DL=%s | ML=%s", dl_status, ml_status)
    return HealthStatus(dl_server=dl_status, ml_server=ml_status)


@app.post("/predict/dl", response_model=PredictResponse, tags=["Inference"])
async def predict_dl(
    file: UploadFile = File(..., description="File ảnh buồng chuối"),
) -> PredictResponse:
    """
    Forward ảnh đến **DL Server** (TensorFlow/Keras CNN).

    Trả về: label (Cut/Keep), confidence, probabilities.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File phải là ảnh (image/*), nhận được: {file.content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File ảnh bị rỗng")

    logger.info(
        "📤 Forwarding to DL Server — file=%s | size=%d bytes",
        file.filename,
        len(image_bytes),
    )

    t_start = time.perf_counter()
    data = await _forward_image(
        target_url=f"{DL_SERVER_URL}/predict/dl",
        image_bytes=image_bytes,
        filename=file.filename or "image.jpg",
        content_type=file.content_type,
    )
    latency_ms = (time.perf_counter() - t_start) * 1000

    logger.info("✅ DL predict — label=%s | latency=%.1fms", data.get("label"), latency_ms)

    return PredictResponse(
        label=data["label"],
        confidence=data["confidence"],
        probabilities=data["probabilities"],
        model=None,
        server="dl",
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/ml", response_model=PredictResponse, tags=["Inference"])
async def predict_ml(
    file: UploadFile = File(..., description="File ảnh buồng chuối"),
    model_name: str = Query(
        default="best_svm",
        description=f"Tên sklearn model. Hợp lệ: {VALID_MODELS}",
    ),
) -> PredictResponse:
    """
    Forward ảnh đến **ML Server** (Sklearn: SVM / XGBoost / Random Forest / HistGradient).

    - **model_name**: Chọn model sklearn muốn dùng.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File phải là ảnh (image/*), nhận được: {file.content_type}",
        )

    if model_name not in VALID_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"model_name '{model_name}' không hợp lệ. Hợp lệ: {VALID_MODELS}",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File ảnh bị rỗng")

    logger.info(
        "📤 Forwarding to ML Server — model=%s | file=%s | size=%d bytes",
        model_name,
        file.filename,
        len(image_bytes),
    )

    t_start = time.perf_counter()
    data = await _forward_image(
        target_url=f"{ML_SERVER_URL}/predict/ml",
        image_bytes=image_bytes,
        filename=file.filename or "image.jpg",
        content_type=file.content_type,
        extra_params={"model_name": model_name},
    )
    latency_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "✅ ML predict — model=%s | label=%s | latency=%.1fms",
        model_name,
        data.get("label"),
        latency_ms,
    )

    return PredictResponse(
        label=data["label"],
        confidence=data["confidence"],
        probabilities=data["probabilities"],
        model=data.get("model"),
        server="ml",
        latency_ms=round(latency_ms, 2),
    )

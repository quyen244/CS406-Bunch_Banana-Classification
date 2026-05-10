from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Response schema chung cho Gateway — trả về client."""

    label: str = Field(..., description="Nhãn dự đoán: 'Cut' hoặc 'Keep'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(...)
    model: Optional[str] = Field(default=None, description="Tên model sklearn (chỉ có ở ML)")
    server: str = Field(..., description="'dl' hoặc 'ml' — server nào xử lý request")
    latency_ms: float = Field(..., description="Thời gian xử lý ở inference server (ms)")


class HealthStatus(BaseModel):
    """Health check response."""

    gateway: str = Field(default="ok")
    dl_server: str = Field(...)
    ml_server: str = Field(...)

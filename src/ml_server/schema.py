from typing import Dict, Optional

from pydantic import BaseModel, Field


class MLPredictResponse(BaseModel):
    """Response schema cho ML server."""

    model: str = Field(..., description="Tên sklearn model đã dùng")
    label: str = Field(..., description="Nhãn dự đoán: 'Cut' hoặc 'Keep'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tự tin của nhãn dự đoán")
    probabilities: Dict[str, float] = Field(
        ..., description="Xác suất cho từng nhãn {'Cut': x, 'Keep': y}"
    )
    status: str = Field(default="success")

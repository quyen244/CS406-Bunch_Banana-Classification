"""
test_gateway.py — Integration tests cho API Gateway

Test coverage:
  1. Health check — Gateway + downstream servers
  2. DL predict — ảnh hợp lệ
  3. ML predict — ảnh hợp lệ, model mặc định
  4. ML predict — tất cả model names hợp lệ
  5. ML predict — model_name không hợp lệ → 422
  6. DL predict — file không phải ảnh → 400
  7. ML predict — file không phải ảnh → 400
  8. DL predict — file rỗng → 400
  9. Routing verification — DL response có server="dl", ML có server="ml"

Yêu cầu: Gateway phải đang chạy tại http://localhost:8080
Chạy: pytest tests/test_gateway.py -v --asyncio-mode=auto
"""

from typing import Any

import pytest
import httpx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_ML_MODELS = [
    "best_svm",
    "best_xgboost",
    "best_random_forest",
    "best_histgradient",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_image_payload(image_bytes: bytes, filename: str = "test.png") -> dict:
    """Tạo files payload cho httpx multipart upload."""
    return {"file": (filename, image_bytes, "image/png")}


def _make_text_payload(text_bytes: bytes) -> dict:
    """Tạo payload với file text (không phải ảnh)."""
    return {"file": ("document.txt", text_bytes, "text/plain")}


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Kiểm tra endpoint /health của Gateway."""

    def test_health_returns_200(self, base_url: str) -> None:
        """GET /health phải trả về HTTP 200."""
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{base_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_health_response_schema(self, base_url: str) -> None:
        """Response /health phải có đủ các field: gateway, dl_server, ml_server."""
        with httpx.Client(timeout=10.0) as client:
            data = client.get(f"{base_url}/health").json()

        assert "gateway" in data, "Missing field: gateway"
        assert "dl_server" in data, "Missing field: dl_server"
        assert "ml_server" in data, "Missing field: ml_server"

    def test_health_gateway_is_ok(self, base_url: str) -> None:
        """Gateway tự báo cáo trạng thái 'ok'."""
        with httpx.Client(timeout=10.0) as client:
            data = client.get(f"{base_url}/health").json()
        assert data["gateway"] == "ok"


class TestDLPredict:
    """Kiểm tra endpoint POST /predict/dl."""

    def test_dl_predict_valid_image_returns_200(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Ảnh hợp lệ gửi đến /predict/dl phải trả về HTTP 200."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            )
        assert response.status_code == 200, response.text

    def test_dl_predict_response_schema(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Response /predict/dl phải có đủ fields: label, confidence, probabilities, server."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            ).json()

        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "server" in data
        assert "latency_ms" in data

    def test_dl_predict_label_is_valid(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Label phải là 'Cut' hoặc 'Keep'."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            ).json()

        assert data["label"] in {"Cut", "Keep"}, f"Unexpected label: {data['label']}"

    def test_dl_predict_confidence_range(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Confidence phải nằm trong [0.0, 1.0]."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_dl_predict_server_field(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Field 'server' phải bằng 'dl' — xác nhận đúng routing."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            ).json()
        assert data["server"] == "dl", f"Expected server='dl', got '{data['server']}'"

    def test_dl_predict_invalid_file_type_returns_400(
        self, base_url: str, sample_text_bytes: bytes
    ) -> None:
        """File không phải ảnh phải bị từ chối với HTTP 400."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{base_url}/predict/dl",
                files=_make_text_payload(sample_text_bytes),
            )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_dl_predict_empty_file_returns_400(self, base_url: str) -> None:
        """File rỗng phải bị từ chối với HTTP 400."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{base_url}/predict/dl",
                files={"file": ("empty.png", b"", "image/png")},
            )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"


class TestMLPredict:
    """Kiểm tra endpoint POST /predict/ml."""

    def test_ml_predict_default_model_returns_200(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Ảnh hợp lệ với model mặc định (best_svm) phải trả về HTTP 200."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
            )
        assert response.status_code == 200, response.text

    def test_ml_predict_response_schema(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Response /predict/ml phải có đủ fields bao gồm 'model'."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
                params={"model_name": "best_svm"},
            ).json()

        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "model" in data
        assert "server" in data
        assert "latency_ms" in data

    def test_ml_predict_server_field(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Field 'server' phải bằng 'ml' — xác nhận đúng routing."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
            ).json()
        assert data["server"] == "ml", f"Expected server='ml', got '{data['server']}'"

    @pytest.mark.parametrize("model_name", VALID_ML_MODELS)
    def test_ml_predict_all_valid_models(
        self, base_url: str, sample_image_bytes: bytes, model_name: str
    ) -> None:
        """Tất cả 4 model names hợp lệ đều phải trả về HTTP 200 với đúng model field."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
                params={"model_name": model_name},
            )
        assert response.status_code == 200, f"Model '{model_name}' failed: {response.text}"
        data = response.json()
        assert data["model"] == model_name, (
            f"Expected model='{model_name}', got '{data['model']}'"
        )

    def test_ml_predict_invalid_model_name_returns_422(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """model_name không hợp lệ phải trả về HTTP 422 (Unprocessable Entity)."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
                params={"model_name": "invalid_model_xyz"},
            )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_ml_predict_invalid_file_type_returns_400(
        self, base_url: str, sample_text_bytes: bytes
    ) -> None:
        """File không phải ảnh phải bị từ chối với HTTP 400."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{base_url}/predict/ml",
                files=_make_text_payload(sample_text_bytes),
            )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_ml_predict_label_is_valid(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Label phải là 'Cut' hoặc 'Keep'."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
            ).json()
        assert data["label"] in {"Cut", "Keep"}, f"Unexpected label: {data['label']}"

    def test_ml_predict_probabilities_sum_to_one(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """Cut + Keep probabilities phải xấp xỉ 1.0."""
        with httpx.Client(timeout=120.0) as client:
            data = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
            ).json()
        prob_sum = sum(data["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum = {prob_sum}, expected ~1.0"


class TestRoutingVerification:
    """Kiểm tra Gateway điều hướng đúng theo path."""

    def test_dl_and_ml_have_different_server_fields(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """
        Cùng một ảnh, gửi lên /predict/dl và /predict/ml phải có server field khác nhau.
        Đây là test xác nhận Gateway thực sự route đến 2 server khác nhau.
        """
        with httpx.Client(timeout=120.0) as client:
            dl_data = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            ).json()
            ml_data = client.post(
                f"{base_url}/predict/ml",
                files=_make_image_payload(sample_image_bytes),
            ).json()

        assert dl_data["server"] == "dl"
        assert ml_data["server"] == "ml"
        assert dl_data["server"] != ml_data["server"]

    def test_latency_ms_is_positive(
        self, base_url: str, sample_image_bytes: bytes
    ) -> None:
        """latency_ms phải là số dương — xác nhận Gateway đo latency đúng."""
        with httpx.Client(timeout=120.0) as client:
            dl_data = client.post(
                f"{base_url}/predict/dl",
                files=_make_image_payload(sample_image_bytes),
            ).json()
        assert dl_data["latency_ms"] > 0

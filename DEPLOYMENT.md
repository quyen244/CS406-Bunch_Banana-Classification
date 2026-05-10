# 🍌 Banana Classification System — Hướng dẫn triển khai End-to-End

## Kiến trúc tổng quan

```
Browser (Next.js :3000)
        │
        ▼
API Gateway (:8080)          ← Điểm tiếp nhận duy nhất từ frontend
    │           │
    ▼           ▼
DL Server   ML Server        ← Internal Docker network, không expose ra ngoài
(:8000)     (:8001)
TF/Keras    HOG+LBP+Sklearn
```

---

## Yêu cầu hệ thống (Prerequisites)

| Công cụ | Phiên bản tối thiểu | Ghi chú |
|---------|---------------------|---------|
| Docker Desktop | 4.x+ | Bắt buộc |
| NVIDIA GPU Driver | 520+ | Chỉ cần cho DL Server |
| nvidia-container-toolkit | Latest | GPU passthrough cho Docker |
| Node.js | 18+ | Chạy Next.js frontend |
| Python | 3.10+ | Chỉ cần nếu chạy tests local |

> **CPU-only mode**: Nếu không có GPU, mở `dockerfile.tf_infer` và đổi dòng đầu thành:
> ```dockerfile
> FROM tensorflow/tensorflow:latest
> ```
> Sau đó xóa block `deploy.resources` trong `docker-compose.yml` của service `dl_server`.

---

## Bước 1: Clone và chuẩn bị

```bash
# Clone repo (hoặc đảm bảo bạn đang ở thư mục dự án)
cd BananaClassification

# Kiểm tra cấu trúc thư mục models/
ls models/
# Phải thấy đủ 7 files:
# dense_121_version_1.keras
# best_svm.pkl, best_xgboost.pkl, best_random_forest.pkl, best_histgradient.pkl
# scaler.pkl, pca.pkl
```

---

## Bước 2: Khởi động toàn bộ hệ thống

### Cách A — Dùng script tự động (khuyến nghị)

```bash
# Linux/macOS
chmod +x run.sh
./run.sh

# Windows (Git Bash)
bash run.sh
```

Script sẽ tự động:
1. Kiểm tra prerequisites
2. Verify model files
3. Build Docker images
4. Start containers và chờ health checks
5. Tạo `.env.local` cho Next.js
6. Chạy `npm run dev`

### Cách B — Thủ công từng bước

```bash
# 1. Build images
docker compose build --parallel

# 2. Start containers (detached)
docker compose up -d

# 3. Kiểm tra logs
docker compose logs -f

# 4. Cài Next.js deps
npm install

# 5. Tạo .env.local
echo "NEXT_PUBLIC_GATEWAY_URL=http://localhost:8080" > .env.local

# 6. Chạy frontend
npm run dev
```

---

## Bước 3: Xác nhận hệ thống hoạt động

### 3.1 Kiểm tra container status

```bash
docker compose ps
```

Kết quả mong đợi:
```
NAME                    STATUS
banana_dl_server        Up (healthy)
banana_ml_server        Up (healthy)
banana_gateway          Up (healthy)
```

### 3.2 Kiểm tra Gateway health

```bash
curl http://localhost:8080/health
```

Kết quả mong đợi:
```json
{
  "gateway": "ok",
  "dl_server": "ok",
  "ml_server": "ok"
}
```

### 3.3 Test DL prediction

```bash
curl -X POST http://localhost:8080/predict/dl \
  -F "file=@/path/to/banana_image.jpg"
```

Kết quả mong đợi:
```json
{
  "label": "Keep",
  "confidence": 0.8734,
  "probabilities": {"Cut": 0.1266, "Keep": 0.8734},
  "model": null,
  "server": "dl",
  "latency_ms": 245.5
}
```

### 3.4 Test ML prediction

```bash
# SVM (mặc định)
curl -X POST "http://localhost:8080/predict/ml?model_name=best_svm" \
  -F "file=@/path/to/banana_image.jpg"

# XGBoost
curl -X POST "http://localhost:8080/predict/ml?model_name=best_xgboost" \
  -F "file=@/path/to/banana_image.jpg"
```

### 3.5 Xem API docs

Mở trình duyệt tại: **http://localhost:8080/docs** (Swagger UI tự động)

---

## Bước 4: Chạy Integration Tests

```bash
# Cài pytest và httpx (lần đầu)
pip install pytest httpx

# Chạy toàn bộ test suite
# (Yêu cầu Gateway đang chạy tại localhost:8080)
pytest tests/ -v

# Chạy một test class cụ thể
pytest tests/test_gateway.py::TestDLPredict -v

# Chạy test parametrize cho tất cả ML models
pytest tests/test_gateway.py::TestMLPredict::test_ml_predict_all_valid_models -v
```

---

## Bước 5: Sử dụng giao diện web

1. Mở trình duyệt: **http://localhost:3000**
2. Chọn mô hình:
   - **Deep Learning** (TensorFlow CNN) — nhanh, chính xác cao
   - **Machine Learning** (Sklearn) — chọn thêm thuật toán: SVM, XGBoost, RF, HistGradient
3. Upload ảnh buồng chuối
4. Cắt/chỉnh sửa ảnh nếu cần
5. Click "Dự đoán ngay"
6. Xem kết quả: nhãn, độ tự tin, biểu đồ xác suất, latency

---

## Lệnh quản lý thông dụng

```bash
# Xem logs của từng service
docker compose logs gateway
docker compose logs dl_server
docker compose logs ml_server

# Xem logs real-time
docker compose logs -f gateway

# Restart một service
docker compose restart gateway

# Dừng tất cả containers
docker compose down

# Dừng và xóa volumes
docker compose down -v

# Rebuild sau khi sửa code
docker compose build gateway && docker compose up -d gateway

# Dùng script
./run.sh --down    # Dừng tất cả
./run.sh --no-build  # Start không build lại
```

---

## Troubleshooting

### ❌ Gateway báo dl_server unreachable

```bash
# Kiểm tra dl_server có healthy không
docker compose ps dl_server
docker compose logs dl_server

# Nguyên nhân thường gặp:
# - Model file không tồn tại trong ./models/
# - Lỗi GPU driver (thử CPU mode)
```

### ❌ ML Server lỗi "No sklearn models could be loaded"

```bash
# Kiểm tra file models
docker compose exec ml_server ls /models/

# Nếu thiếu file, copy vào:
# Đảm bảo ./models/ có đủ best_*.pkl, scaler.pkl, pca.pkl
```

### ❌ GPU không được nhận diện

```bash
# Kiểm tra nvidia-container-toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Nếu lỗi, chuyển sang CPU mode (sửa dockerfile.tf_infer + docker-compose.yml)
```

### ❌ Next.js lỗi "fetch failed" hoặc CORS error

```bash
# Kiểm tra .env.local có đúng URL không
cat .env.local
# Phải có: NEXT_PUBLIC_GATEWAY_URL=http://localhost:8080

# Restart Next.js sau khi sửa .env.local
npm run dev
```

---

## Cấu trúc thư mục

```
BananaClassification/
├── inference/
│   ├── tf_model_inference.py    # TFInference class
│   └── ml_model_inference.py    # ImagePredictor class (HOG+LBP+sklearn)
├── src/
│   ├── backend/                 # DL Server (TF, port 8000)
│   │   ├── main.py              # FastAPI app, endpoint /predict/dl
│   │   └── schema.py            # ResponsePredict schema
│   ├── gateway/                 # API Gateway (port 8080)
│   │   ├── main.py              # FastAPI app, routing /predict/dl & /predict/ml
│   │   └── schema.py            # PredictResponse, HealthStatus
│   ├── ml_server/               # ML Server (Sklearn, port 8001)
│   │   ├── main.py              # FastAPI app, endpoint /predict/ml
│   │   └── schema.py            # MLPredictResponse
│   └── app/                     # Next.js frontend
│       └── page.tsx             # Main UI với server/model selector
├── models/                      # Model files (bind mount vào containers)
├── tests/
│   ├── conftest.py              # pytest fixtures
│   └── test_gateway.py          # Integration tests
├── Dockerfile.gateway           # Gateway image
├── dockerfile.tf_infer          # DL Server image (GPU)
├── dockerfile.ml_infer          # ML Server image
├── docker-compose.yml           # Orchestration
├── run.sh                       # Automation script
└── DEPLOYMENT.md                # Tài liệu này
```

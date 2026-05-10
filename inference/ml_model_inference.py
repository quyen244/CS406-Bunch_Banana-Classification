import io
import logging
from typing import Dict, List, Optional

import cv2
import joblib
import numpy as np
from scipy.stats import skew
from skimage.feature import hog, local_binary_pattern

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MLInference")

# Danh sách model hợp lệ
VALID_MODELS: List[str] = [
    "best_svm",
    "best_xgboost",
    "best_random_forest",
    "best_histgradient",
]


class ImagePredictor:
    """
    Inference engine cho các sklearn models.

    Pipeline: bytes → BGR→RGB → feature extraction (HOG + LBP + Color) → PCA → Scaler → predict
    """

    def __init__(
        self,
        models_dir: str,
        scaler_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ) -> None:
        self.resize_size = (256, 256)
        self.models: Dict[str, object] = {}

        # --- Load tất cả sklearn models vào dict khi startup ---
        for model_name in VALID_MODELS:
            model_file = f"{models_dir}/{model_name}.pkl"
            try:
                self.models[model_name] = joblib.load(model_file)
                logger.info("✅ Loaded model: %s", model_name)
            except FileNotFoundError:
                logger.warning("⚠️  Model file not found, skipping: %s", model_file)
            except Exception as exc:
                logger.error("❌ Failed to load model %s: %s", model_name, exc)

        if not self.models:
            raise RuntimeError("No sklearn models could be loaded from: %s" % models_dir)

        # --- Load scaler (optional) ---
        self.scaler: Optional[object] = None
        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Loaded scaler from: %s", scaler_path)
            except Exception as exc:
                logger.warning("⚠️  Could not load scaler: %s", exc)

        # --- Load PCA (optional) ---
        self.pca: Optional[object] = None
        if pca_path:
            try:
                self.pca = joblib.load(pca_path)
                logger.info("✅ Loaded PCA from: %s", pca_path)
            except Exception as exc:
                logger.warning("⚠️  Could not load PCA: %s", exc)

    # ------------------------------------------------------------------
    # Private: Feature extraction (giống hệt training pipeline)
    # ------------------------------------------------------------------

    def _extract_features(self, img_rgb: np.ndarray) -> np.ndarray:
        """Trích xuất HOG + LBP + Color histograms + Color moments từ ảnh RGB."""
        # 1. Tiền xử lý (Phải dùng img_res cho tất cả các bước sau)
        img_res = cv2.resize(img_rgb, self.resize_size)
        gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    
        # 2. HOG features (8100 features)
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False,
        )
    
        # 3. LBP features (26 features)
        # n_points = 24 -> Số bins chuẩn là n_points + 2 = 26
        # Code cũ của bạn dùng np.arange(0, 28) tạo ra 27 bins -> Dư 1
        lbp = local_binary_pattern(gray, 24, 3, method="uniform")
        lbp_hist, _ = np.histogram(
            lbp.ravel(), 
            bins=np.arange(0, 27), # 0 đến 26 tạo ra 26 khoảng (bins)
            range=(0, 26)
        )
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
    
        # 4. Color histograms (96 features: 32 bins * 3 channels)
        # QUAN TRỌNG: Phải dùng img_res
        hist_feat = []
        for i in range(3):
            hist = cv2.calcHist([img_res], [i], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            hist_feat.extend(hist.flatten().tolist())
    
        # 5. Color moments (9 features: 3 stats * 3 channels)
        # QUAN TRỌNG: Phải dùng img_res
        moments = []
        for i in range(3):
            ch = img_res[:, :, i]
            # Giữ nguyên kiểu ép kiểu float như code mẫu của bạn
            moments += [float(np.mean(ch)), float(np.std(ch)), float(skew(ch.flatten()))]
    
        # Kết hợp lại: 8100 + 26 + 96 + 9 = 8231
        features = np.hstack([hog_feat, lbp_hist, hist_feat, moments])
        return features

    # ------------------------------------------------------------------
    # Private: Decode image bytes → RGB ndarray
    # ------------------------------------------------------------------

    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode raw bytes thành numpy array BGR, sau đó chuyển sang RGB."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.error("❌ cv2.imdecode returned None — invalid image bytes")
                return None
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.error("❌ Image decode error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public: Predict
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes, model_name: str) -> Dict:
        """
        Chạy inference cho ảnh.

        Args:
            image_bytes: Raw bytes của file ảnh (từ FastAPI UploadFile.read()).
            model_name: Tên model sklearn cần dùng (phải nằm trong VALID_MODELS).

        Returns:
            Dict chứa label, confidence, probabilities, model, status.
        """
        # Validate model name
        if model_name not in self.models:
            available = list(self.models.keys())
            logger.error("❌ Unknown model '%s'. Available: %s", model_name, available)
            return {
                "status": "failed",
                "error": f"Model '{model_name}' not found. Available: {available}",
            }

        # Decode ảnh
        img_rgb = self._decode_image(image_bytes)
        if img_rgb is None:
            return {"status": "failed", "error": "Cannot decode image"}

        try:
            # Feature extraction
            features = self._extract_features(img_rgb)

            # PCA transform (optional)
            if self.pca is not None:
                features = self.pca.transform(features.reshape(1 , -1))

            print("Features shape after PCA (if applied):", features.shape)

            # Scaling (optional)
            if self.scaler is not None:
                features_2d = self.scaler.transform(features)
            else:
                features_2d = features
            
            print("Features shape after scaling (if applied):", features_2d.shape)
            
            # Inference
            model = self.models[model_name]
            pred = model.predict(features_2d)[0]
            prob = model.predict_proba(features_2d)[0]

            label = "Keep" if int(pred) == 1 else "Cut"
            pred_idx = int(pred)

            logger.info(
                "✅ Prediction — model=%s | label=%s | confidence=%.4f",
                model_name,
                label,
                float(prob[pred_idx]),
            )

            return {
                "model": model_name,
                "label": label,
                "confidence": round(float(prob[pred_idx]), 4),
                "probabilities": {
                    "Cut": round(float(prob[0]), 4),
                    "Keep": round(float(prob[1]), 4),
                },
                "status": "success",
            }

        except Exception as exc:
            logger.error("❌ Prediction error for model '%s': %s", model_name, exc)
            return {"status": "failed", "error": str(exc)}

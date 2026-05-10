import os
import cv2
import logging
import numpy as np
from typing import Dict, Union, Optional
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TFInference")

class TFInference:
    def __init__(self, model_path: str):
        self.class_names = ['Cut', 'Keep']
        self.target_size = (256, 256)
        
        try:
            # Ưu tiên load model từ tham số truyền vào
            self.model = keras.models.load_model(model_path)
            logger.info(f"✅ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, img_input: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        try:
            # Trường hợp 1: Nếu là Bytes (Từ FastAPI UploadFile)
            if isinstance(img_input, bytes):
                nparr = np.frombuffer(img_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Trường hợp 2: Nếu là đường dẫn File (String)
            elif isinstance(img_input, str):
                img = cv2.imread(img_input)
            
            # Trường hợp 3: Nếu đã là mảng Numpy
            else:
                img = img_input

            if img is None: return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=0)
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {str(e)}")
            return None

    def predict(self, img_input: Union[str, np.ndarray, bytes]) -> Dict:
        processed_img = self.preprocess_image(img_input)
        if processed_img is None:
            return {"status": "failed", "error": "Invalid image data"}

        pred_prob = self.model.predict(processed_img, verbose=0)
        prob_value = pred_prob.item()
        idx = (prob_value > 0.5)
        
        return {
            "label": self.class_names[idx],
            "confidence": round((1 - prob_value) if idx == 0 else (prob_value), 4),
            "probabilities": {
                "Cut": round(prob_value, 4),
                "Keep": round(1 - prob_value, 4)
            },
            "status": "success"
        }
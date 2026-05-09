from fastapi import FastAPI , Depends, File , HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .schema import ResponsePredict
from inference.tf_model_inference import TFInference
import os 

MODEL_PATH = os.getenv("MODEL_PATH", "model/banana_classifier.keras")
# Khởi tạo engine inference
infer_engine = TFInference(MODEL_PATH)

app = FastAPI()

origins = [
    'localhost',
    'http://localhost:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def health_check():
    return {"message": "Hello World"}


@app.post('/predict', response_model=ResponsePredict)
async def predict(file: UploadFile = File(...)): 
    # Kiểm tra định dạng file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image")

    # Đọc bytes từ file stream
    image_bytes = await file.read()
    
    # Thực hiện dự đoán
    output = infer_engine.predict(image_bytes)
    
    if output["status"] == "failed":
        raise HTTPException(status_code=500, detail=output["error"])
        
    return ResponsePredict(
        label=output["label"],
        confidence=output["confidence"],
        probabilities=output["probabilities"]
    )
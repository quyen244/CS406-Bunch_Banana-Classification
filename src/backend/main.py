from fastapi import FastAPI , Depends, File , HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .schema import ResponsePredict

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
async def predict(file : UploadFile = File(...)): 
    
    return ResponsePredict(
        label="Cut",
        confidence=0.95,
        probabilities={
            "Cut": 0.95,
            "Keep": 0.05
        }
    )
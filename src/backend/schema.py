from pydantic import BaseModel
from typing import Dict

# # predict request body
# class RequestPredict(BaseModel):
#     image_url: str

class ResponsePredict(BaseModel):
    label: str 
    confidence: float 
    probabilities: Dict[str, float]

    
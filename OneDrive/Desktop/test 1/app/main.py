from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import __version__ as model_version, predict_pipeline

app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    try:
        language = predict_pipeline(payload.text)
        return {"language": language}
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return {"error": "Internal Server Error", "details": str(e)}
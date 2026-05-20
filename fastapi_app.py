from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")


class TicketInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Ticket Classifier API Running"}


@app.post("/predict")
def predict(data: TicketInput):
    try:
        input_data = pd.DataFrame(
            [[0] * len(features)],
            columns=features
        )

        prediction = model.predict(input_data)

        return {
            "prediction": str(prediction[0])
        }

    except Exception as e:
        return {
            "error": str(e)
        }
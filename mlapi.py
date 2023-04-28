from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class predictionData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

with open('RF_Model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def prediabetes_endPoint(data:predictionData):
    df = pd.DataFrame([data.dict().values()], columns = data.dict().keys())
    yhat = model.predict(df)
    return int(yhat)
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

class predictionData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: float
    Insulin: float
    BMI: int
    DiabetesPedigreeFunction: float
    Age: int

with open('New_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.post('/')
async def prediabetes_endPoint(data:predictionData):
    if data.Age <= 30:
        data.Age = 1
    elif data.Age <= 40:
        data.Age = 2
    elif data.Age <= 50:
        data.Age = 3
    elif data.Age <= 60:
        data.Age = 4
    else:
        data.Age = 5

    if data.Glucose <= 60:
        data.Glucose = 1
    elif data.Glucose <= 80:
        data.Glucose = 2
    elif data.Glucose <= 140:
        data.Glucose = 3
    elif data.Glucose <= 180:
        data.Glucose = 4
    else:
        data.Glucose = 5

    if data.BloodPressure <= 60:
        data.BloodPressure = 1
    elif data.BloodPressure <= 75:
        data.BloodPressure = 2
    elif data.BloodPressure <= 90:
        data.BloodPressure = 3
    elif data.BloodPressure <= 100:
        data.BloodPressure = 4
    else:
        data.BloodPressure = 5

    if data.BMI <= 19:
        data.BMI = 1
    elif data.BMI <= 24:
        data.BMI = 2
    elif data.BMI <= 30:
        data.BMI = 3
    elif data.BMI <= 40:
        data.BMI = 4
    else:
        data.BMI = 5

    df = pd.DataFrame([data.dict().values()], columns = data.dict().keys())
    newDF = scaler.transform(df)
    yhat = model.predict(newDF)
    return int(yhat)
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model_weights.pkl")
scaler = joblib.load("scaler_weights.pkl")

class CancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

@app.get("/")
def home():
    return {"message": "Breast Cancer Classification API Running"}

@app.post("/predict")
def predict(data: CancerData):
    input_data = np.array([[
        data.mean_radius,
        data.mean_texture,
        data.mean_perimeter,
        data.mean_area,
        data.mean_smoothness,
        data.mean_compactness,
        data.mean_concavity,
        data.mean_concave_points,
        data.mean_symmetry,
        data.mean_fractal_dimension,
        data.radius_error,
        data.texture_error,
        data.perimeter_error,
        data.area_error,
        data.smoothness_error,
        data.compactness_error,
        data.concavity_error,
        data.concave_points_error,
        data.symmetry_error,
        data.fractal_dimension_error,
        data.worst_radius,
        data.worst_texture,
        data.worst_perimeter,
        data.worst_area,
        data.worst_smoothness,
        data.worst_compactness,
        data.worst_concavity,
        data.worst_concave_points,
        data.worst_symmetry,
        data.worst_fractal_dimension
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    result = int((prediction > 0.5).astype("int32")[0][0])

    return {"prediction": result}
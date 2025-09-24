from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import pandas as pd
from app.services.recommendation import generate_recommendations

router = APIRouter()

class HealthData(BaseModel):
    user_id : int
    steps : int
    workout_minutes : int
    HR_rest : float
    HR_active : float
    BMI : float
    calories_burned : float

@router.post("/predict")
async def get_recommendations(data: List[HealthData]):
    df = pd.DataFrame([item.dict() for item in data])
    result_df = generate_recommendations(df)
    return result_df.to_dict(orient='records')

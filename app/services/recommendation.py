import os
import joblib
import numpy as np
import pandas as pd

kmeans_model_path = os.path.join(os.path.dirname(__file__), 'f:/health-tracker/app/models/k-means_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'f:/health-tracker/app/models/scaler.joblib')
kmeans = joblib.load(kmeans_model_path)
scaler = joblib.load(scaler_path)

xgboost_model_path = os.path.join(os.path.dirname(__file__), 'f:/health-tracker/app/models/xgboost_model.joblib')
scaler_xgboost_path = os.path.join(os.path.dirname(__file__), 'f:/health-tracker/app/models/scaler_xgboost.joblib')
xgb_model = joblib.load(xgboost_model_path)
scaler_xgboost = joblib.load(scaler_xgboost_path)

clsuter_features = ['steps','workout_minutes','calories_burned','HR_rest','HR_active','BMI']
xgb_features = ['steps','HR_rest','HR_active','workout_minutes','BMI']

def assign_custer(indut_data:pd.DataFrame) -> pd.Series:
    X_scaled = scaler.transform(indut_data[clsuter_features])
    return kmeans.predict(X_scaled)

def predict_calories(indut_data:pd.DataFrame) -> np.ndarray:
    X_scaled = scaler_xgboost.transform(indut_data[xgb_features])
    return xgb_model.predict(X_scaled)

def generate_dynamic_recommendations(row):
    cluster = row['cluster']
    rec = ""
    if cluster == 0:
        rec = ("Maintain your activity level and keep tracking diet", "Focus on balanced nutrition","add some light training sessions")

    elif cluster == 1:
        rec = ("Great Workout Routine","Add more cardio sessions + Strength Training sessions mix for fat loss","Maintain a balanced diet and monitor Calories")
    elif cluster == 2:
        rec = ("Seems to be overweight","Add more cardio sessions for fat loss"," manage diet with more nutritious and protien rich food and avoid junk food","Increase daily steps target to 10 - 15k")
    elif cluster == 3:
        rec = ("Boost overall movement! Increase daily steps, incorporate light-to-moderate workouts," "and maintain consistent activity patterns.""Focus on a balanced diet to support your activity level.") 
    
    if row["predicted_calories"] < 2000:
        rec += " Consider increasing intensity or duration slightly to burn more calories."
    if row["HR_active"] < 120:
        rec += " Try incorporating more cardio to elevate your heart rate during activities."
    
    return rec

def generate_recommendations(input_data:pd.DataFrame) -> pd.DataFrame:
    df = input_data.copy()
    df['cluster'] = assign_custer(df)
    df['predicted_calories'] = predict_calories(df)
    df['recommendations'] = df.apply(generate_dynamic_recommendations, axis=1)
    return df


import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import joblib 

data_path = os.path.join(os.path.dirname(__file__),'f:/health-tracker/data/processed/health_data_clustered.csv')
df = pd.read_csv(data_path)
print(df.head())

target = 'calories_burned'
features = ['steps','HR_rest','HR_active','workout_minutes','BMI']

if 'clsuter' in df.columns:
    features.append('cluster')

X = df[features]
y = df[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBRegressor(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42
)

xgb_model.fit(X_train_scaled,y_train)

y_pred = xgb_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"XGBoost Evaluation: \n RMSE :{rmse:.2f}, MAE: {mae:.2f}, R2_Score: {r2:.3f}")

model_path = os.path.join(os.path.dirname(__file__),'f:/health-tracker/app/models/xgboost_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__),'f:/health-tracker/app/models/scaler_xgboost.joblib')

joblib.dump(xgb_model,model_path)
joblib.dump(scaler,scaler_path)
print(f"XGBoost model saved at: {model_path}")
print(f"Scaler saved at: {scaler_path}")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

data_path = os.path.join(os.path.dirname(__file__),'f:/health-tracker/data/raw/synthetic_data.csv')
df = pd.read_csv(data_path)
# print(df.head())

features = ['steps','workout_minutes','calories_burned','HR_rest','HR_active','BMI']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
silhouette_scores = []
k_range = range(2,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k,init='k-means++',random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled,labels))

plt.figure(figsize=(8,5))
plt.plot(k_range,silhouette_scores,'ro-',label = 'Silhouette Score')
plt.xlabel("number of clsuters(K)")
plt.ylabel("Silhouette Score")
plt.show()
print(silhouette_scores)

plt.figure(figsize=(8,5))
plt.plot(k_range,wcss,'bo-',label = 'WCSS')
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.show()
print(wcss)

best_k = 4
kmeans = KMeans(n_clusters=best_k,init='k-means++',random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(df.head())

model_path = os.path.join(os.path.dirname(__file__),'f:/health-tracker/app/ml/k-means_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__),'f:/health-tracker/app/ml/scaler.joblib')

joblib.dump(kmeans,model_path)
joblib.dump(scaler,scaler_path)

print(f"K-MEANS++ MODEL IS SAVED AT {model_path}")
print(f"SCaler saved at {scaler_path}")

df.to_csv(os.path.join(os.path.dirname(__file__), 'f:/health-tracker/data/processed/health_data_clustered.csv'), index=False)
print("Clustered data saved.")


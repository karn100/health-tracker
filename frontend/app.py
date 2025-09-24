import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Health Tracker MVP", layout="wide")

st.title("💓 Health Tracker MVP")
st.markdown("""
Enter your health data and get personalized recommendations.
""")

# -----------------------------
# 1. User Input Sidebar
# -----------------------------
st.sidebar.header("Enter Your Health Data")

user_id = st.sidebar.number_input("User ID", min_value=1, value=1)
steps = st.sidebar.number_input("Steps", min_value=0, value=8000)
workout_minutes = st.sidebar.number_input("Workout Minutes", min_value=0, value=45)
HR_rest = st.sidebar.number_input("Resting Heart Rate", min_value=40.0, value=70.0)
HR_active = st.sidebar.number_input("Active Heart Rate", min_value=80.0, value=120.0)
BMI = st.sidebar.number_input("BMI", min_value=10.0, value=25.0)
calories_burned = st.sidebar.number_input("Calories Burned", min_value=0.0, value=1900.0)

# -----------------------------
# 2. Send Data to Backend API
# -----------------------------
if st.button("Get Recommendation"):
    input_data = [{
        "user_id": user_id,
        "steps": steps,
        "workout_minutes": workout_minutes,
        "HR_rest": HR_rest,
        "HR_active": HR_active,
        "BMI": BMI,
        "calories_burned": calories_burned
    }]

    api_url = "http://127.0.0.1:8001/health/predict"
    response = requests.post(api_url, json=input_data)
    
    if response.status_code == 200:
        result = response.json()[0]
        
        st.subheader("✅ Prediction Result")
        st.write(f"**Cluster:** {result['cluster']}")
        st.write(f"**Predicted Calories Burned:** {result['predicted_calories']:.2f}")
        st.write(f"**Personalized Recommendation:** {result['recommendations']}")
        
        # -----------------------------
        # 3. Bar Chart: User Stats
        # -----------------------------
        st.subheader("Your Activity Overview")
        st.bar_chart({
            'Steps': result['steps'],
            'Workout Minutes': result['workout_minutes'],
            'Calories Burned': result['calories_burned']
        })
        
    else:
        st.error("Error connecting to API. Please make sure the backend is running.")

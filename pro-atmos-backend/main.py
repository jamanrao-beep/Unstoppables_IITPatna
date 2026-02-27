from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import os

# Initialize FastAPI
app = FastAPI(title="Pro Atmos Guard API")

# --- 1. ENABLE CORS (Critical for React Frontend) ---
# This allows your React app (running on localhost:3000) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATA STRUCTURE ---
class SensorData(BaseModel):
    temperature: float
    humidity: float
    pm25: float
    aqi: float  # Raw value from ESP32

# --- 3. ML PREDICTION HELPER ---
def get_ai_prediction(temp, hum, pm25):
    try:
        # Load the model trained in the previous step
        if os.path.exists('aqi_model.pkl'):
            model = joblib.load('aqi_model.pkl')
            prediction = model.predict([[temp, hum, pm25]])
            return round(float(prediction[0]), 2)
        else:
            # Fallback logic if model file is missing
            return round((pm25 * 1.5) + (temp * 0.5), 2)
    except Exception as e:
        print(f"ML Error: {e}")
        return 0.0

# Store the latest data in memory so React can "GET" it
latest_data = {
    "temperature": 0,
    "humidity": 0,
    "pm25": 0,
    "raw_aqi": 0,
    "ai_predicted_aqi": 0,
    "status": "Waiting for sensor..."
}

# --- 4. ENDPOINTS ---

@app.get("/")
async def health_check():
    return {"message": "Pro Atmos Guard Backend is Live!"}

@app.post("/api/sensor-data")
async def receive_sensor_data(data: SensorData):
    global latest_data
    
    # Run ML Prediction
    ai_val = get_ai_prediction(data.temperature, data.humidity, data.pm25)
    
    # Determine Status
    status_msg = "Healthy" if ai_val < 100 else "Hazardous" if ai_val > 200 else "Moderate"
    
    # Update global storage
    latest_data = {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "pm25": data.pm25,
        "raw_aqi": data.aqi,
        "ai_predicted_aqi": ai_val,
        "status": status_msg
    }
    
    print(f"📥 Data Received | AI Predicted AQI: {ai_val} | Status: {status_msg}")
    
    return {"message": "Data processed by Pro Atmos AI", "ai_aqi": ai_val}

@app.get("/api/get-latest")
async def send_to_react():
    # This is what your React Frontend will call every 3 seconds
    return latest_data

if __name__ == "__main__":
    # Run with host 0.0.0.0 so the bridge (Serveo/LocalTunnel) can see it
    uvicorn.run(app, host="0.0.0.0", port=8000)
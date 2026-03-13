from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import os
import serial
import threading
import time

# -------------------------
# 1️⃣ Initialize FastAPI
# -------------------------
app = FastAPI(title="Pro Atmos Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 2️⃣ Data structure
# -------------------------
class SensorData(BaseModel):
    mq5: int
    mq7: int
    temperature: float
    humidity: float
    aqi: int

latest_data = {
    "mq5": 0,
    "mq7": 0,
    "temperature": 0,
    "humidity": 0,
    "raw_aqi": 0,
    "ai_predicted_aqi": 0,
    "status": "Waiting for sensor..."
}

# -------------------------
# 3️⃣ ML prediction helper
# -------------------------
def get_ai_prediction(temp, hum, pm25):
    try:
        if os.path.exists('aqi_model.pkl'):
            model = joblib.load('aqi_model.pkl')
            prediction = model.predict([[temp, hum, pm25]])
            return round(float(prediction[0]), 2)
        else:
            # Fallback if model file is missing
            return round((pm25 * 1.5) + (temp * 0.5), 2)
    except Exception as e:
        print(f"ML Error: {e}")
        return 0.0

# -------------------------
# 4️⃣ FastAPI Endpoints
# -------------------------
@app.get("/")
async def health_check():
    return {"message": "Pro Atmos Guard Backend is Live!"}

@app.get("/api/get-latest")
async def send_to_react():
    return latest_data

# -------------------------
# 5️⃣ Serial Reading Thread
# -------------------------
def read_serial_data(port="COM3", baud=115200):
    global latest_data
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # wait for ESP32 to initialize
        print(f"📡 Listening to ESP32 on {port}...")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("MQ5 Gas Level:"):
                try:
                    # Read 6 consecutive lines from your Serial Monitor
                    mq5_line = line
                    mq7_line = ser.readline().decode('utf-8').strip()
                    temp_line = ser.readline().decode('utf-8').strip()
                    hum_line = ser.readline().decode('utf-8').strip()
                    aqi_line = ser.readline().decode('utf-8').strip()
                    quality_line = ser.readline().decode('utf-8').strip()
                    ser.readline()  # skip separator line

                    # Parse values
                    mq5 = int(mq5_line.split(":")[1].strip())
                    mq7 = int(mq7_line.split(":")[1].strip())
                    temperature = float(temp_line.split(":")[1].strip())
                    humidity = float(hum_line.split(":")[1].strip())
                    aqi = int(aqi_line.split(":")[1].strip())

                    # Run AI prediction
                    ai_val = get_ai_prediction(temperature, humidity, aqi)

                    # Determine status
                    status_msg = (
                        "Healthy" if ai_val < 100
                        else "Hazardous" if ai_val > 200
                        else "Moderate"
                    )

                    # Update global latest_data
                    latest_data = {
                        "mq5": mq5,
                        "mq7": mq7,
                        "temperature": temperature,
                        "humidity": humidity,
                        "raw_aqi": aqi,
                        "ai_predicted_aqi": ai_val,
                        "status": status_msg
                    }

                    print(f"📥 Serial Data | AI AQI: {ai_val} | Status: {status_msg}")

                except Exception as e:
                    print(f"⚠️ Serial parse error: {e}")
    except Exception as e:
        print(f"❌ Serial connection error: {e}")

# Start serial listener in a separate thread
threading.Thread(target=read_serial_data, daemon=True).start()

# -------------------------
# 6️⃣ Run FastAPI
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
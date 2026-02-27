import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. GENERATE TRAINING DATA (Simulating historical patterns)
def train_aqi_model():
    print("🤖 Training Pro Atmos AI Model...")
    
    # Create 1000 rows of synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    temp = np.random.uniform(15, 45, n_samples)      # 15°C to 45°C
    hum = np.random.uniform(30, 90, n_samples)       # 30% to 90%
    pm25 = np.random.uniform(0, 300, n_samples)      # 0 to 300 µg/m³
    
    # Formula for synthetic AQI: PM2.5 is the main driver, 
    # but high humidity makes it worse (traps particles)
    aqi = (pm25 * 1.8) + (hum * 0.2) + (temp * 0.1) + np.random.normal(0, 5, n_samples)

    df = pd.DataFrame({
        'temperature': temp,
        'humidity': hum,
        'pm25': pm25,
        'aqi': aqi
    })

    # 2. TRAIN THE MODEL
    X = df[['temperature', 'humidity', 'pm25']]
    y = df['aqi']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. SAVE THE MODEL
    joblib.dump(model, 'aqi_model.pkl')
    print("✅ Model trained and saved as 'aqi_model.pkl'")

# 4. PREDICTION FUNCTION (For your FastAPI)
def predict_aqi(temp, hum, pm25):
    try:
        model = joblib.load('aqi_model.pkl')
        input_data = np.array([[temp, hum, pm25]])
        prediction = model.predict(input_data)
        return round(float(prediction[0]), 2)
    except:
        # Fallback if model file isn't found
        return round((pm25 * 1.8), 2)

if __name__ == "__main__":
    train_aqi_model()
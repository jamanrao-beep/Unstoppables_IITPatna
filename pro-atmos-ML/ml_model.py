# ==========================================
# SMART CITY AQI + PM2.5 + PM10 AI PLATFORM
# Advanced Hackathon Edition
# ==========================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.interpolate import griddata
from datetime import timedelta

st.set_page_config(layout="wide")

# ============================
# LOAD DATA
# ============================

@st.cache_data
def load_data():
    df = pd.read_csv("aqi_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

data = load_data()

st.title("🌍 Smart AI Air Quality Monitoring System")

# ============================
# FEATURE ENGINEERING
# ============================

data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month

features = ['AQI', 'PM2.5', 'PM10', 'hour', 'day', 'month']

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features])

# ============================
# CREATE SEQUENCES
# ============================

SEQ_LEN = 24

X, y = [], []

for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i][:3])  # Predict AQI, PM2.5, PM10

X, y = np.array(X), np.array(y)

# ============================
# LSTM MODEL (Multi Output)
# ============================

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(3))  # AQI, PM2.5, PM10

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=8, batch_size=16, verbose=0)

# ============================
# RANDOM FOREST ENSEMBLE
# ============================

rf = RandomForestRegressor()
rf.fit(X.reshape(X.shape[0], -1), y)

# ============================
# 24-HOUR FORECAST
# ============================

last_seq = scaled[-SEQ_LEN:]
future_predictions = []

current_seq = last_seq.copy()

for i in range(24):

    lstm_pred = model.predict(current_seq.reshape(1, SEQ_LEN, 6), verbose=0)
    rf_pred = rf.predict(current_seq.reshape(1, -1))

    combined = (lstm_pred[0] + rf_pred[0]) / 2
    future_predictions.append(combined)

    new_row = current_seq[-1].copy()
    new_row[:3] = combined
    current_seq = np.vstack((current_seq[1:], new_row))

future_predictions = np.array(future_predictions)

# Inverse scale
dummy = np.zeros((24, 6))
dummy[:, :3] = future_predictions
forecast = scaler.inverse_transform(dummy)[:, :3]

future_time = [data['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(24)]

# ============================
# PLOT FORECAST
# ============================

st.subheader("📈 24-Hour Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(x=data['timestamp'], y=data['AQI'], name="Historical AQI"))
fig.add_trace(go.Scatter(x=future_time, y=forecast[:,0], name="Predicted AQI"))

fig.add_trace(go.Scatter(x=data['timestamp'], y=data['PM2.5'], name="Historical PM2.5"))
fig.add_trace(go.Scatter(x=future_time, y=forecast[:,1], name="Predicted PM2.5"))

fig.add_trace(go.Scatter(x=data['timestamp'], y=data['PM10'], name="Historical PM10"))
fig.add_trace(go.Scatter(x=future_time, y=forecast[:,2], name="Predicted PM10"))

st.plotly_chart(fig, use_container_width=True)

# ============================
# ALERT SYSTEM
# ============================

st.subheader("🚨 Air Quality Alerts")

latest_pm25 = forecast[0][1]
latest_pm10 = forecast[0][2]

if latest_pm25 > 150:
    st.error("⚠️ Severe PM2.5 Alert!")
elif latest_pm25 > 80:
    st.warning("⚠️ Moderate PM2.5 Alert")
else:
    st.success("PM2.5 Normal")

if latest_pm10 > 200:
    st.error("⚠️ Severe PM10 Alert!")
elif latest_pm10 > 100:
    st.warning("⚠️ Moderate PM10 Alert")
else:
    st.success("PM10 Normal")

# ============================
# SPATIAL INTERPOLATION GRID
# ============================

st.subheader("🗺️ Spatial AQI Heat Distribution Grid")

np.random.seed(42)
lat = np.random.uniform(22.3, 22.4, 30)
lon = np.random.uniform(82.6, 82.7, 30)
pm25_vals = np.random.uniform(20, 250, 30)
pm10_vals = np.random.uniform(40, 300, 30)

grid_x, grid_y = np.mgrid[min(lat):max(lat):200j,
                          min(lon):max(lon):200j]

grid_pm25 = griddata((lat, lon), pm25_vals, (grid_x, grid_y), method='cubic')
grid_pm10 = griddata((lat, lon), pm10_vals, (grid_x, grid_y), method='cubic')

# PM2.5 Heatmap
plt.figure()
plt.title("PM2.5 Heat Map")
plt.imshow(grid_pm25, extent=(min(lon), max(lon), min(lat), max(lat)), origin='lower')
plt.colorbar(label="PM2.5")
st.pyplot(plt)

# PM10 Heatmap
plt.figure()
plt.title("PM10 Heat Map")
plt.imshow(grid_pm10, extent=(min(lon), max(lon), min(lat), max(lat)), origin='lower')
plt.colorbar(label="PM10")
st.pyplot(plt)

# ============================
# DYNAMIC AQI GRID CALCULATION
# ============================

st.subheader("📊 Dynamic AQI Grid (Based on PM2.5 & PM10)")

aqi_grid = (0.6 * grid_pm25) + (0.4 * grid_pm10)

plt.figure()
plt.title("Dynamic AQI Distribution Grid")
plt.imshow(aqi_grid, extent=(min(lon), max(lon), min(lat), max(lat)), origin='lower')
plt.colorbar(label="AQI")
st.pyplot(plt)

st.success("System Fully Operational 🚀")

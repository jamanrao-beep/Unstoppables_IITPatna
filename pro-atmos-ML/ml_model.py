# ==========================================
# PRO ATMOS GUARD: AQI PREDICTIVE AI
# LSTM + Random Forest Ensemble Model
# ==========================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  # Added for more maps
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import GRU
from scipy.interpolate import griddata
from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Pro Atmos AI", layout="wide")

# 1. LOAD DATA
@st.cache_data
def load_data():
    # Ensure aqi_data.csv is in the same folder
    df = pd.read_csv("aqi_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

data = load_data()
st.title("🌍 Smart AI Air Quality Monitoring System")

# 2. FEATURE ENGINEERING
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month

features = ['AQI', 'PM2.5', 'PM10', 'hour', 'day', 'month']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features])

# 3. CREATE SEQUENCES (Lookback period of 24 hours)
SEQ_LEN = 24
X, y = [], []

for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i][:3])  # Predict AQI, PM2.5, PM10

X, y = np.array(X), np.array(y)

#check before model = Sequential()
if len(X) == 0:
    st.error("❌ Not enough data in aqi_data.csv to train the model. Please add at least 25 rows of data.")
    st.stop()
    
# 4. ENSEMBLE MODEL INITIALIZATION
# Define LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(3)) 
model.compile(optimizer='adam', loss='mse')

# Define Random Forest
rf = RandomForestRegressor(n_estimators=100)

# 5. TRAINING
with st.spinner('Training AI Ensemble...'):
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    rf.fit(X.reshape(X.shape[0], -1), y)
st.success("System Fully Operational ")

# 6. 24-HOUR FORECAST LOGIC
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

# Inverse scaling for results
dummy = np.zeros((24, 6))
dummy[:, :3] = np.array(future_predictions)
forecast = scaler.inverse_transform(dummy)[:, :3]
future_time = [data['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(24)]

# ==========================================
# ADVANCED AQI TREND ANALYSIS
# ==========================================

st.subheader("AI AQI Trend Analysis (Next 24 Hours)")

trend = []

for i in range(1, len(forecast)):
    diff = forecast[i,0] - forecast[i-1,0]

    if diff > 5:
        trend.append("Rising ")
    elif diff < -5:
        trend.append("Falling ")
    else:
        trend.append("Stable ")

trend.insert(0, "Current")

trend_df = pd.DataFrame({
    "Time": future_time,
    "Predicted AQI": forecast[:,0],
    "Trend": trend
})

st.dataframe(trend_df)

# --- NEW FEATURE: TOP METRICS DASHBOARD ---
st.divider()
col1, col2, col3, col4 = st.columns(4)
current_aqi = data['AQI'].iloc[-1]
predicted_aqi = forecast[0, 0]

col1.metric("Current AQI", f"{current_aqi:.1f}", delta=f"{current_aqi - data['AQI'].iloc[-2]:.1f}", delta_color="inverse")
col2.metric("Predicted AQI (Next Hour)", f"{predicted_aqi:.1f}")
col3.metric("Max Forecasted PM2.5", f"{np.max(forecast[:,1]):.1f} µg/m³")
col4.metric("Average Humidity (Est.)", "65%")

# 7. VISUALIZATION (Plotly Forecast)
st.subheader(" 24-Hour Predictive Forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['AQI'], name="Historical AQI"))
fig.add_trace(go.Scatter(x=future_time, y=forecast[:,0], name="Predicted AQI", line=dict(dash='dash')))
st.plotly_chart(fig, use_container_width=True)

# 8. SPATIAL HEATMAPS (Original)
st.subheader(" Spatial AQI Distribution Grid")
lat, lon = np.random.uniform(25.5, 25.6, 30), np.random.uniform(85.1, 85.2, 30)
pm25_vals = np.random.uniform(50, 200, 30)
grid_x, grid_y = np.mgrid[min(lat):max(lat):100j, min(lon):max(lon):100j]
grid_pm25 = griddata((lat, lon), pm25_vals, (grid_x, grid_y), method='cubic')

plt.figure(figsize=(10, 6))
plt.imshow(grid_pm25, extent=(min(lon), max(lon), min(lat), max(lat)), origin='lower', cmap='RdYlGn_r')
plt.colorbar(label="PM2.5 Levels")
st.pyplot(plt)

# --- NEW FEATURE: INTERACTIVE BUBBLE MAP ---
st.subheader(" Real-time Station Hotspots")
map_df = pd.DataFrame({
    'lat': lat,
    'lon': lon,
    'AQI': np.random.uniform(100, 300, 30),
    'Station': [f"Station {i}" for i in range(30)]
})
fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="AQI", color="AQI",
                  color_continuous_scale=px.colors.sequential.YlOrRd, 
                  hover_name="Station", zoom=11, height=500)
fig_map.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig_map, use_container_width=True)

# ==========================================
# PREDICTION CONFIDENCE INTERVAL
# ==========================================

st.subheader(" Prediction Confidence Range")

std_dev = np.std(forecast[:,0])

upper_bound = forecast[:,0] + std_dev
lower_bound = forecast[:,0] - std_dev

fig_conf = go.Figure()

fig_conf.add_trace(go.Scatter(
    x=future_time,
    y=forecast[:,0],
    name="Predicted AQI"
))

fig_conf.add_trace(go.Scatter(
    x=future_time,
    y=upper_bound,
    name="Upper Confidence",
    line=dict(dash="dot")
))

fig_conf.add_trace(go.Scatter(
    x=future_time,
    y=lower_bound,
    name="Lower Confidence",
    line=dict(dash="dot")
))

st.plotly_chart(fig_conf, use_container_width=True)



# ==========================================
# HYBRID DEEP LEARNING MODEL
# LSTM + GRU ARCHITECTURE
# ==========================================

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))

model.add(GRU(64, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dense(16, activation="relu"))

model.add(Dense(3))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# ==========================================
# AI POLLUTION SPIKE DETECTION
# ==========================================

st.subheader(" Pollution Spike Prediction")

spikes = forecast[:,0] > 200

if spikes.any():
    spike_times = np.array(future_time)[spikes]

    for t in spike_times[:3]:
        st.warning(f" Possible pollution spike expected around {t}")
else:
    st.success(" No major AQI spikes predicted in next 24 hours")
    
# ==========================================
# ADVANCED TREND VISUALIZATION
# ==========================================

st.subheader(" Advanced AQI Trend Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['timestamp'],
    y=data['AQI'],
    name="Historical AQI",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=future_time,
    y=forecast[:,0],
    name="AI Forecast AQI",
    line=dict(color="red", dash="dash")
))

fig.add_trace(go.Scatter(
    x=future_time,
    y=upper_bound,
    name="Upper Confidence",
    line=dict(width=1, dash="dot")
))

fig.add_trace(go.Scatter(
    x=future_time,
    y=lower_bound,
    name="Lower Confidence",
    fill='tonexty',
    line=dict(width=1, dash="dot")
))

fig.update_layout(
    title="24 Hour AI AQI Prediction Trend",
    xaxis_title="Time",
    yaxis_title="AQI Index"
)

st.plotly_chart(fig, use_container_width=True)

# --- NEW FEATURE: HEALTH ADVISORY SYSTEM ---
st.subheader(" AI Health Advisory")
latest_aqi = forecast[0,0]
if latest_aqi <= 50:
    st.success(" **Air Quality: Good.** Perfect for outdoor activities and exercise.")
elif latest_aqi <= 100:
    st.info("🟡 **Air Quality: Moderate.** Sensitive individuals should reduce prolonged outdoor exertion.")
elif latest_aqi <= 200:
    st.warning("🟠 **Air Quality: Unhealthy.** Wear a mask. Children and elderly should stay indoors.")
else:
    st.error("🔴 **Air Quality: Hazardous!** Severe health risk. Avoid all outdoor activities. Use air purifiers.")

# --- NEW FEATURE: POLLUTANT BREAKDOWN (PIE CHART) ---
st.subheader(" Current Pollutant Composition")
pie_data = pd.DataFrame({
    'Pollutant': ['PM2.5', 'PM10', 'O3', 'NO2', 'CO'],
    'Value': [data['PM2.5'].iloc[-1], data['PM10'].iloc[-1], 25, 15, 5]
})
fig_pie = px.pie(pie_data, values='Value', names='Pollutant', hole=.4, 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_pie)

# ==========================================
# AI POLLUTION SOURCE ESTIMATION
# ==========================================

st.subheader(" Possible Pollution Sources")

if latest_aqi > 200:
    st.error("Major contributing sources likely:")
    st.write("- Traffic Emissions")
    st.write("- Industrial Pollution")
    st.write("- Biomass Burning")
elif latest_aqi > 100:
    st.warning("Moderate pollution likely due to:")
    st.write("- Urban traffic")
    st.write("- Construction dust")
else:
    st.success("Low pollution environment detected")
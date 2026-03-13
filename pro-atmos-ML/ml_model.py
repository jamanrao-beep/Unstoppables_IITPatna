
#SMART AI AIR QUALITY MONITORING SYSTEM
# GNN + LSTM-TRANSFORMER + RANDOM FOREST


import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    Add, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Pro Atmos AI", layout="wide")

# FILE CHECK


if not os.path.exists("aqi_data.csv"):
    st.error("aqi_data.csv not found. Please place dataset in project folder.")
    st.stop()


# LOAD DATA


@st.cache_data
def load_data():
    df = pd.read_csv("aqi_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    return df

data = load_data()

st.title("Smart AI Air Quality Monitoring System")

# ==========================================
# FEATURE ENGINEERING
# ==========================================

data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month

features = ['AQI', 'PM2.5', 'PM10', 'hour', 'day', 'month']

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features])

# ==========================================
# CREATE TIME SEQUENCES
# ==========================================

SEQ_LEN = 24

X, y = [], []

for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i][:3])

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    st.error("Dataset too small. Need at least 25 rows.")
    st.stop()

# ==========================================
# LSTM + TRANSFORMER MODEL
# ==========================================

def build_lstm_transformer(input_shape):

    inputs = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(x, x)

    x = Add()([x, attention])
    x = LayerNormalization()(x)

    ff = Dense(128, activation="relu")(x)
    ff = Dense(64)(ff)

    x = Add()([x, ff])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)

    outputs = Dense(3)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


lstm_transformer_model = build_lstm_transformer(
    (X.shape[1], X.shape[2])
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

# ==========================================
# GNN SPATIAL MODEL
# ==========================================

class AQIGraphModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)

        self.fc = torch.nn.Linear(8, 1)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        return self.fc(x)

# GPU SUPPORT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
  
def build_graph(data, num_nodes):

    node_features = torch.tensor(
        data[['AQI', 'PM2.5', 'PM10']].values[:num_nodes],
        dtype=torch.float
    )

    target = torch.tensor(
        data[['AQI']].values[:num_nodes],
        dtype=torch.float
    )

    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return node_features, edge_index, target


# Build graph
num_nodes = min(len(data), 50)
node_features, edge_index, target = build_graph(data, num_nodes)

data_graph = Data(
    x=node_features,
    edge_index=edge_index,
    y=target
)


data_graph = Data(
        x=node_features,
        edge_index=edge_index,
        y=target
    )

model = AQIGraphModel().to(device)

optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

loss_fn = torch.nn.MSELoss()
def train_gnn():

 for epoch in range(200):

        model.train()
        optimizer.zero_grad()

        pred = model(
            data_graph.x.to(device),
            data_graph.edge_index.to(device)
        )

        loss = loss_fn(pred, data_graph.y.to(device))

        loss.backward()
        optimizer.step()

 return model, data_graph

gnn_model, gnn_data = train_gnn()

# ==========================================
# TRAIN MODELS
# ==========================================

with st.spinner("Training AI Models..."):

    lstm_transformer_model.fit(
        X, y,
        epochs=20,
        batch_size=16,
        verbose=0
    )

rf_model.fit(
    X[:,-1,:],
    y
)

st.success("AI System Operational")

# ==========================================
# MODEL EVALUATION
# ==========================================

pred_train = lstm_transformer_model.predict(X)

mae = mean_absolute_error(
    y[:,0],
    pred_train[:,0]
)

st.write("Model Training MAE:", round(mae,3))

# ==========================================
# 24 HOUR FORECAST
# ==========================================

last_seq = scaled[-SEQ_LEN:]

future_predictions = []

current_seq = last_seq.copy()

for i in range(24):

    lstm_pred = lstm_transformer_model.predict(
        current_seq.reshape(1,SEQ_LEN,6),
        verbose=0
    )

    rf_pred = rf_model.predict(
    current_seq[-1].reshape(1, -1)
    )

    combined = (lstm_pred[0] + rf_pred[0]) / 2

    future_predictions.append(combined)

    new_row = current_seq[-1].copy()

    new_row[:3] = combined

    current_seq = np.vstack((current_seq[1:], new_row))

dummy = np.zeros((24,6))

dummy[:,:3] = np.array(future_predictions)

forecast = scaler.inverse_transform(dummy)[:,:3]

future_time = [
    data['timestamp'].iloc[-1] + timedelta(hours=i+1)
    for i in range(24)
]

# ==========================================
# CONFIDENCE INTERVAL
# ==========================================

std_dev = np.std(np.array(future_predictions)[:,0])

upper = forecast[:,0] + std_dev
lower = forecast[:,0] - std_dev

# ==========================================
# FORECAST VISUALIZATION
# ==========================================

st.subheader("24 Hour AQI Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['timestamp'],
    y=data['AQI'],
    name="Historical AQI"
))

fig.add_trace(go.Scatter(
    x=future_time,
    y=forecast[:,0],
    name="Predicted AQI",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=future_time,
    y=upper,
    name="Upper Confidence",
    line=dict(dash="dot")
))

fig.add_trace(go.Scatter(
    x=future_time,
    y=lower,
    name="Lower Confidence",
    fill="tonexty"
))

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# GNN SPATIAL PREDICTION
# ==========================================

gnn_model.eval()

with torch.no_grad():

    spatial_pred = gnn_model(
        gnn_data.x.to(device),
        gnn_data.edge_index.to(device)
    )

spatial_df = pd.DataFrame({
    "Station":[f"Station {i}" for i in range(len(spatial_pred))],
    "Predicted AQI":spatial_pred.cpu().numpy().flatten()
})

st.subheader("Spatial AQI Prediction")

st.dataframe(spatial_df)

# ==========================================
# HEATMAP
# ==========================================

# ==========================================
# HEATMAP USING REAL DATA
# ==========================================

if "latitude" in data.columns and "longitude" in data.columns:

    lat = data["latitude"].values
    lon = data["longitude"].values
    pm25 = data["PM2.5"].values

else:
    # fallback simulation
    lat = np.random.uniform(25.5,25.6,30)
    lon = np.random.uniform(85.1,85.2,30)
    pm25 = np.random.uniform(50,200,30)
grid_x, grid_y = np.mgrid[
    min(lat):max(lat):100j,
    min(lon):max(lon):100j
]

grid_pm25 = griddata(
    (lat,lon),
    pm25,
    (grid_x,grid_y),
    method="linear"
)

plt.figure(figsize=(10,6))

plt.imshow(
    grid_pm25,
    extent=(min(lon),max(lon),min(lat),max(lat)),
    origin='lower',
    cmap='RdYlGn_r'
)

plt.colorbar(label="PM2.5")

st.pyplot(plt)

# ==========================================
# BLIND SPOT DETECTION USING IDW
# ==========================================

st.subheader("AI Blind Spot Detection (Inverse Distance Weighting)")

# Simulated sensor locations (replace with real station coordinates if available)
sensor_lat = np.random.uniform(25.52, 25.60, 15)
sensor_lon = np.random.uniform(85.12, 85.20, 15)

# Simulated AQI readings from sensors
sensor_aqi = np.random.uniform(60, 220, 15)

# Create prediction grid
grid_lat = np.linspace(min(sensor_lat), max(sensor_lat), 100)
grid_lon = np.linspace(min(sensor_lon), max(sensor_lon), 100)

grid_lat_mesh, grid_lon_mesh = np.meshgrid(grid_lat, grid_lon)

# ==========================================
# IDW FUNCTION
# ==========================================

def inverse_distance_weighting(x, y, z, xi, yi, power=2):

    interpolated = np.zeros_like(xi)

    x = x.reshape(-1,1,1)
    y = y.reshape(-1,1,1)
    z = z.reshape(-1,1,1)

    dist = np.sqrt((x - xi)**2 + (y - yi)**2)

    dist[dist == 0] = 1e-10

    weights = 1 / (dist ** power)

    weighted_values = weights * z

    interpolated = np.sum(weighted_values, axis=0) / np.sum(weights, axis=0)

    return interpolated
# ==========================================
# COMPUTE IDW INTERPOLATION
# ==========================================

idw_grid = inverse_distance_weighting(
    sensor_lat,
    sensor_lon,
    sensor_aqi,
    grid_lat_mesh,
    grid_lon_mesh
)

# ==========================================
# DETECT BLIND SPOTS
# ==========================================

blind_spots = []

threshold_distance = 0.01   # distance threshold
threshold_aqi = 150         # pollution threshold

for i in range(idw_grid.shape[0]):
    for j in range(idw_grid.shape[1]):

        point_lat = grid_lat_mesh[i,j]
        point_lon = grid_lon_mesh[i,j]

        # distance to nearest sensor
        dist = np.min(
            np.sqrt(
                (sensor_lat - point_lat)**2 +
                (sensor_lon - point_lon)**2
            )
        )

        if dist > threshold_distance and idw_grid[i,j] > threshold_aqi:
            blind_spots.append((point_lat, point_lon, idw_grid[i,j]))

blind_spots = pd.DataFrame(
    blind_spots,
    columns=["Latitude","Longitude","Estimated AQI"]
)

# ==========================================
# VISUALIZE BLIND SPOTS
# ==========================================

fig_blind = px.scatter_mapbox(
    blind_spots,
    lat="Latitude",
    lon="Longitude",
    color="Estimated AQI",
    size="Estimated AQI",
    zoom=10,
    height=500,
    color_continuous_scale="RdYlGn_r"
)

fig_blind.update_layout(
    mapbox_style="open-street-map",
    title="Detected Air Pollution Blind Spots"
)

st.plotly_chart(fig_blind, use_container_width=True)

# ==========================================
# DISPLAY TABLE
# ==========================================

st.write("Potential locations where new sensors should be deployed:")

st.dataframe(blind_spots.head(20))

# ==========================================
# HEALTH ADVISORY
# ==========================================

st.subheader("AI Health Advisory")

latest = forecast[0,0]

if latest <= 50:

    st.success("Air Quality Good")

elif latest <= 100:

    st.info("Air Quality Moderate")

elif latest <= 200:

    st.warning("Unhealthy air. Wear mask")

else:

    st.error("Hazardous AQI. Stay indoors")
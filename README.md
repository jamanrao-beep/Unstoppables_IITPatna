# 🌍 Pro Atmos Guard
**Advanced Real-Time Air Quality Monitoring & Predictive Analytics**

Pro Atmos Guard is an end-to-end IoT solution designed to monitor environmental hazards, visualize air quality data in real-time, and utilize Machine Learning to predict future AQI trends. Built for **Technex '26 @ IIT Patna**.

---

## 👥 The Team
* **Joginapally Aman Rao** - IoT Integration & Backend Architecture
* **Perumalla Neha** - Frontend Development & UI/UX
* **Mahalakshmi Pattamsetti** - ML Modeling & Data Analysis
* **Krishna Teja Degala** - Hardware Simulation & Design

---

## 🚀 The System Architecture

### 1. Hardware Simulation (Wokwi)
We utilize an **ESP32** microcontroller simulated in Wokwi to interface with:
* **DHT22**: For precise Temperature and Humidity readings.
* **PM2.5 & CO Sensors**: Simulated via high-precision potentiometers to represent fluctuating air pollutants.
* **LCD (I2C)**: For local real-time data display.

### 2. The Bridge (IoT to Cloud)
Since the simulation runs in a virtual environment, we use an **SSH Tunnel (Serveo/Ngrok)** to bridge the gap.
* The ESP32 sends secure `POST` requests containing JSON payloads.
* A custom **User-Agent** and **Bypass-Headers** strategy ensures data bypasses security filters.

### 3. Backend (FastAPI & Python)
Our Python backend acts as the "Central Brain":
* **Data Ingestion**: Receives sensor data via a RESTful API.
* **ML Integration**: The backend passes incoming data through a **Random Forest Regressor** model to determine the Air Quality Index (AQI) category and predict trends.
* **CORS Management**: Seamlessly serves data to the React frontend.

### 4. Frontend (React.js)
A modern, high-performance dashboard featuring:
* **Glassmorphism UI**: For a clean, futuristic aesthetic.
* **Real-time Polling**: Dynamically updates charts and gauges as soon as the ESP32 detects a change.
* **Predictive AI Insights**: Visual indicators that trigger when the ML model predicts hazardous conditions.

---

## 🛠️ Installation & Setup

### Backend
1. Navigate to your project root.
2. Install dependencies: `pip install fastapi uvicorn scikit-learn joblib`
3. Start the server: `python main.py`
4. Start ML: Run streamlit run ml_model.py in your terminal.

### Bridge (Tunnel)
1. In a new terminal, run: `ssh -R 80:localhost:8000 serveo.net`
2. Copy the generated URL and paste it into the Wokwi `serverName` variable.

---

## 📡 Deployment
* **Frontend**: Vercel
* **Tunneling**: Serveo / LocalTunnel
* **Hardware Simulation**: Wokwi Engine

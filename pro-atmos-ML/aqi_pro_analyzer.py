import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import base64

# Set page configuration
st.set_page_config(page_title="Pro Atmos - Professional Analyzer", layout="wide")

st.title("Pro Atmos: Professional AQI Network & Risk Reporting")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("🌐 Network Configuration")
    num_stations = st.number_input("Total Active Stations", min_value=1, max_value=50, value=3)
    
    st.header("⚙️ Algorithm Tuning")
    idw_p = st.slider("IDW Power (Spatial Decay)", 1.0, 5.0, 3.5)
    max_radius = st.slider("Max Coverage (Degrees)", 0.1, 2.0, 0.5)

# ==========================================
# NEW: EPA PM2.5 TO AQI CONVERSION LOGIC
# ==========================================
def calculate_aqi_from_pm25(pm):
    if pm <= 12.0: return ((50 - 0) / (12.0 - 0)) * (pm - 0) + 0
    elif pm <= 35.4: return ((100 - 51) / (35.4 - 12.1)) * (pm - 12.1) + 51
    elif pm <= 55.4: return ((150 - 101) / (55.4 - 35.5)) * (pm - 35.5) + 101
    elif pm <= 150.4: return ((200 - 151) / (150.4 - 55.5)) * (pm - 55.5) + 151
    elif pm <= 250.4: return ((300 - 201) / (250.4 - 150.5)) * (pm - 150.5) + 201
    elif pm <= 350.4: return ((400 - 301) / (350.4 - 250.5)) * (pm - 250.5) + 301
    else: return ((500 - 401) / (500.4 - 350.5)) * (pm - 350.5) + 401

# ==========================================
# 1. DYNAMIC STATION ENTRY
# ==========================================
st.subheader("1. Station Deployment Data")
station_data = []

# Layout for inputs
for i in range(int(num_stations)):
    with st.expander(f"📍 Station {i+1} Settings", expanded=(i==0)):
        c1, c2, c3, c4 = st.columns(4)
        lat = c1.number_input(f"Latitude", value=25.5 + (i * 0.05), key=f"lat{i}", format="%.4f")
        lon = c2.number_input(f"Longitude", value=85.1 + (i * 0.05), key=f"lon{i}", format="%.4f")
        aqi = c3.number_input(f"Measured AQI", value=100 + (i * 15), key=f"aqi{i}")
        pm25 = c4.number_input(f"PM2.5 (µg/m³)", value=35.0 + (i * 10), key=f"pm25{i}")
        
        # Calculate derived AQI for reference
        derived_aqi = calculate_aqi_from_pm25(pm25)
        st.caption(f"💡 EPA Calculated AQI from PM2.5: **{derived_aqi:.1f}**")
        
        station_data.append({"Station": f"ST-{i+1}", "Lat": lat, "Lon": lon, "AQI": aqi, "PM2.5": pm25})

df = pd.DataFrame(station_data)

# ==========================================
# 2. SPATIAL RISK MAP
# ==========================================
st.markdown("### 🗺️ Network Visualization")
# Updated to show PM2.5 in hover and labels
fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="AQI", size="AQI",
                        hover_name="Station",
                        hover_data={"AQI": True, "PM2.5": True, "Lat": True, "Lon": True},
                        color_continuous_scale="RdYlGn_r", zoom=9, height=500)
fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. ADVANCED POINT QUERY
# ==========================================
st.divider()
st.subheader("2. Targeted Location Analysis")

qc1, qc2, qc3 = st.columns([1, 1, 1])
target_lat = qc1.number_input("Target Latitude", value=25.6, format="%.4f")
target_lon = qc2.number_input("Target Longitude", value=85.2, format="%.4f")

# Calculation Logic (Updated for PM2.5)
def get_aqi_estimate(t_lat, t_lon, data_df):
    lats, lons, aqis, pms = data_df["Lat"].values, data_df["Lon"].values, data_df["AQI"].values, data_df["PM2.5"].values
    dists = np.sqrt((lats - t_lat)**2 + (lons - t_lon)**2)
    m_dist = np.min(dists)
    
    if m_dist > max_radius: return None, None, m_dist
    if m_dist < 1e-7: 
        idx = np.argmin(dists)
        return aqis[idx], pms[idx], m_dist
    
    w = 1 / (dists ** idw_p)
    return np.sum(w * aqis) / np.sum(w), np.sum(w * pms) / np.sum(w), m_dist

if qc3.button("Generate Risk Assessment", use_container_width=True):
    est_aqi, est_pm, d = get_aqi_estimate(target_lat, target_lon, df)
    
    if est_aqi is None:
        st.error(f"❌ TOO FAR TO DISPLAY: Nearest sensor is {d:.2f}° away. (Limit: {max_radius}°)")
    else:
        # Results Display
        r1, r2, r3 = st.columns([1, 1, 2])
        r1.metric("Calculated AQI", f"{est_aqi:.1f}")
        r2.metric("Estimated PM2.5", f"{est_pm:.1f}")
        
        # Health Logic
        status, advice = "", ""
        if est_aqi <= 50: status, advice = "GOOD", "Fresh air. No health risk detected."
        elif est_aqi <= 100: status, advice = "MODERATE", "Acceptable. Sensitive individuals should monitor symptoms."
        elif est_aqi <= 200: status, advice = "UNHEALTHY", "Avoid prolonged outdoor exertion."
        else: status, advice = "HAZARDOUS", "STAY INDOORS. Dangerous levels of pollution."
        
        # EPA Auto-Check based on estimated PM2.5
        epa_aqi = calculate_aqi_from_pm25(est_pm)
        
        with r3:
            st.info(f"**Zone Status:** {status}\n\n**AI Advisory:** {advice}")
            st.warning(f"**EPA Reference:** Based on PM2.5 alone, the AQI would be **{epa_aqi:.1f}**.")

        # --- PDF REPORT GENERATION ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Pro Atmos: Air Quality Risk Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Target Coordinates: {target_lat}, {target_lon}", ln=True)
        pdf.cell(200, 10, f"Estimated AQI (Sensor Network): {est_aqi:.1f}", ln=True)
        pdf.cell(200, 10, f"Estimated PM2.5: {est_pm:.1f} ug/m3", ln=True)
        pdf.cell(200, 10, f"EPA Derived AQI: {epa_aqi:.1f}", ln=True)
        pdf.cell(200, 10, f"Condition: {status}", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Health Advisory: {advice}")
        
        pdf_output = pdf.output(dest="S").encode("latin-1")
        b64_pdf = base64.b64encode(pdf_output).decode('utf-8')
        pdf_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="AQI_PM25_Report.pdf">Download PDF Report</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

# ==========================================
# 4. ZONE CLASSIFICATION (Summary)
# ==========================================
with st.expander("📊 View Network Summary Stats"):
    st.write(df.describe())
    # Chart now shows both AQI and PM2.5 for comparison
    st.bar_chart(df.set_index("Station")[["AQI", "PM2.5"]])
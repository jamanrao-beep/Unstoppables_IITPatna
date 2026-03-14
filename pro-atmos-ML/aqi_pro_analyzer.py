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
# 1. DYNAMIC STATION ENTRY
# ==========================================
st.subheader("1. Station Deployment Data")
station_data = []

# Layout for inputs
for i in range(int(num_stations)):
    with st.expander(f"📍 Station {i+1} Settings", expanded=(i==0)):
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input(f"Latitude", value=25.5 + (i * 0.05), key=f"lat{i}", format="%.4f")
        lon = c2.number_input(f"Longitude", value=85.1 + (i * 0.05), key=f"lon{i}", format="%.4f")
        aqi = c3.number_input(f"Measured AQI", value=100 + (i * 15), key=f"aqi{i}")
        station_data.append({"Station": f"ST-{i+1}", "Lat": lat, "Lon": lon, "AQI": aqi})

df = pd.DataFrame(station_data)

# ==========================================
# 2. SPATIAL RISK MAP
# ==========================================
st.markdown("### 🗺️ Network Visualization")
fig = px.scatter_mapbox(df, lat="Lat", lon="Lon", color="AQI", size="AQI",
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

# Calculation Logic
def get_aqi_estimate(t_lat, t_lon, data_df):
    lats, lons, aqis = data_df["Lat"].values, data_df["Lon"].values, data_df["AQI"].values
    dists = np.sqrt((lats - t_lat)**2 + (lons - t_lon)**2)
    m_dist = np.min(dists)
    
    if m_dist > max_radius: return None, m_dist
    if m_dist < 1e-7: return aqis[np.argmin(dists)], m_dist
    
    w = 1 / (dists ** idw_p)
    return np.sum(w * aqis) / np.sum(w), m_dist

if qc3.button("Generate Risk Assessment", use_container_width=True):
    est_aqi, d = get_aqi_estimate(target_lat, target_lon, df)
    
    if est_aqi is None:
        st.error(f"❌ TOO FAR TO DISPLAY: Nearest sensor is {d:.2f}° away. (Limit: {max_radius}°)")
    else:
        # Results Display
        r1, r2 = st.columns([1, 2])
        r1.metric("Calculated AQI", f"{est_aqi:.1f}")
        
        # Health Logic
        status, advice = "", ""
        if est_aqi <= 50: status, advice = "GOOD", "Fresh air. No health risk detected."
        elif est_aqi <= 100: status, advice = "MODERATE", "Acceptable. Sensitive individuals should monitor symptoms."
        elif est_aqi <= 200: status, advice = "UNHEALTHY", "Avoid prolonged outdoor exertion."
        else: status, advice = "HAZARDOUS", "STAY INDOORS. Dangerous levels of pollution."
        
        with r2:
            st.info(f"**Zone Status:** {status}\n\n**AI Advisory:** {advice}")

        # --- PDF REPORT GENERATION ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Pro Atmos: Air Quality Risk Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Target Coordinates: {target_lat}, {target_lon}", ln=True)
        pdf.cell(200, 10, f"Estimated AQI: {est_aqi:.1f}", ln=True)
        pdf.cell(200, 10, f"Condition: {status}", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Health Advisory: {advice}")
        
        pdf_output = pdf.output(dest="S").encode("latin-1")
        b64_pdf = base64.b64encode(pdf_output).decode('utf-8')
        pdf_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="AQI_Report.pdf">Download PDF Report</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

# ==========================================
# 4. ZONE CLASSIFICATION (Summary)
# ==========================================
with st.expander("📊 View Network Summary Stats"):
    st.write(df.describe())
    st.bar_chart(df.set_index("Station")["AQI"])
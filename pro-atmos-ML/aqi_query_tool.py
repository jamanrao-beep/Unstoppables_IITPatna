import streamlit as st
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Pro Atmos - AQI Query Tool", layout="wide")

st.title("Pro Atmos: Interactive AQI Deployment & Query")
st.info("This tool allows you to manually deploy virtual stations and query the estimated AQI at any specific coordinate using Inverse Distance Weighting.")

# ==========================================
# 1. STATION DEPLOYMENT SECTION
# ==========================================
st.subheader("1. Station Deployment")
num_stations = st.number_input("How many stations would you like to deploy?", min_value=1, max_value=100, value=5)

station_data = []

st.write("### Enter Station Details")
# Create columns for headers
hcol1, hcol2, hcol3 = st.columns(3)
hcol1.write("**Latitude**")
hcol2.write("**Longitude**")
hcol3.write("**Approximate AQI**")

# Use a loop to create input rows for each station
for i in range(int(num_stations)):
    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input(f"Lat", value=25.5 + (i * 0.01), key=f"lat_{i}", format="%.6f", label_visibility="collapsed")
    with col2:
        lon = st.number_input(f"Lon", value=85.1 + (i * 0.01), key=f"lon_{i}", format="%.6f", label_visibility="collapsed")
    with col3:
        aqi = st.number_input(f"AQI", value=100, key=f"aqi_{i}", label_visibility="collapsed")
    
    station_data.append({"lat": lat, "lon": lon, "aqi": aqi})

# Convert to arrays for calculation
s_lats = np.array([s['lat'] for s in station_data])
s_lons = np.array([s['lon'] for s in station_data])
s_aqis = np.array([s['aqi'] for s in station_data])

# ==========================================
# 2. RANDOM POINT QUERY SECTION
# ==========================================
st.divider()
st.subheader("2. Query AQI at a Random Location")

q_col1, q_col2 = st.columns(2)
query_lat = q_col1.number_input("Enter target Latitude:", format="%.6f")
query_lon = q_col2.number_input("Enter target Longitude:", format="%.6f")

if st.button("Calculate AQI"):
    # Calculate Euclidean distances from the query point to all stations
    distances = np.sqrt((s_lats - query_lat)**2 + (s_lons - query_lon)**2)
    min_dist = np.min(distances)
    
    # DISTANCE THRESHOLD: Define how far is "too far" 
    # (e.g., 1.0 degree is roughly 111km)
    DISTANCE_THRESHOLD = 1.0 
    
    if min_dist > DISTANCE_THRESHOLD:
        st.error(f"The location ({query_lat}, {query_lon}) is too far from any active stations to display a reliable AQI.")
    else:
        # Inverse Distance Weighting (IDW) logic
        # A higher power (p) makes closer stations much more influential
        power = 3.5
        
        # Avoid division by zero if the user queries the exact location of a station
        if min_dist < 1e-9:
            final_aqi = s_aqis[np.argmin(distances)]
        else:
            weights = 1 / (distances ** power)
            final_aqi = np.sum(weights * s_aqis) / np.sum(weights)
        
        # Display Results
        st.metric(label=f"Estimated AQI at ({query_lat}, {query_lon})", value=round(final_aqi, 2))
        
        # Health Advisory Logic
        if final_aqi <= 50:
            st.success("Air Quality Category: Good")
        elif final_aqi <= 100:
            st.info("Air Quality Category: Moderate")
        elif final_aqi <= 200:
            st.warning("Air Quality Category: Unhealthy")
        else:
            st.error("Air Quality Category: Hazardous")

# Optional: Show a small table of the deployed stations for reference
with st.expander("View Deployed Stations Table"):
    st.table(pd.DataFrame(station_data))
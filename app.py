import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Air Quality Monitor",
    page_icon="",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0a1a;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #111133;
    }
    [data-testid="stMetric"] {
        background-color: #1a1a2e;
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stButton > button {
        background-color: #00d4ff;
        color: black;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        padding: 12px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #00d4ff;
        border-radius: 8px;
    }
    p, li {
        color: #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("airquality_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_columns.json", "r") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, feature_columns = load_model()

# Header
st.title("Air Quality & Productivity Monitor")
st.markdown("Real-time air quality monitoring and worker productivity prediction powered by Machine Learning.")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("Sensor Readings")
st.sidebar.markdown("Adjust sliders to simulate environmental conditions.")

CO = st.sidebar.slider("CO Concentration (mg/m³)", 0.3, 8.1, 2.0, step=0.1)
NO2 = st.sidebar.slider("NO2 Concentration (µg/m³)", 19.0, 194.0, 80.0, step=1.0)
NOx = st.sidebar.slider("NOx Concentration (µg/m³)", 12.0, 478.0, 100.0, step=5.0)
Benzene = st.sidebar.slider("Benzene C6H6 (µg/m³)", 0.5, 39.2, 5.0, step=0.5)
Temperature = st.sidebar.slider("Temperature (°C)", 6.3, 29.3, 15.0, step=0.5)
Humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 100.0, 50.0, step=1.0)

# Main KPI layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "CO Level", f"{CO} mg/m³",
        delta="High" if CO > 5 else "Normal",
        delta_color="inverse" if CO > 5 else "normal"
    )

with col2:
    st.metric(
        "NO2 Level", f"{NO2} µg/m³",
        delta="High" if NO2 > 100 else "Normal",
        delta_color="inverse" if NO2 > 100 else "normal"
    )

with col3:
    st.metric(
        "Temperature", f"{Temperature} °C",
        delta="High" if Temperature > 25 else "Normal",
        delta_color="inverse" if Temperature > 25 else "normal"
    )

st.markdown("---")

# Predict button
if st.button("Predict Productivity & Air Quality Risk", type="primary"):

    # Build input dataframe matching training features
    input_dict = {col: 0 for col in feature_columns}

    # Map slider values to feature names
    mapping = {
        'CO(GT)': CO,
        'NO2(GT)': NO2,
        'NOx(GT)': NOx,
        'C6H6(GT)': Benzene,
        'T': Temperature,
        'RH': Humidity,
        'CO_Lag_1hr': CO,
        'CO_Lag_2hr': CO,
        'CO_3hr_MA': CO,
        'CO_24hr_MA': CO,
        'Temp_Lag_1hr': Temperature,
        'Humid_Dev': abs(Humidity - 50),
        'Temp_Dev': abs(Temperature - 14.6),
        'Synthetic_PM2.5': (NOx * 0.1) + (CO * 2.5),
        'Rolling_PM2.5_Avg': (NOx * 0.1) + (CO * 2.5),
        'Hour': 12,
        'Pollution_Spikes': 1 if (CO > 6 or NO2 > 150) else 0,
    }

    for key, value in mapping.items():
        if key in input_dict:
            input_dict[key] = value

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    productivity = model.predict(input_scaled)[0]
    productivity = np.clip(productivity, 0, 100)

    # Rule based safety override using WHO thresholds
    if CO > 7 or NO2 > 150 or NOx > 350 or Benzene > 25:
        productivity = min(productivity, 45)
    elif CO > 5 or NO2 > 100 or NOx > 250 or Benzene > 15:
        productivity = min(productivity, 65)
    elif CO > 3 or NO2 > 70 or NOx > 150 or Benzene > 8:
        productivity = min(productivity, 78)

    # Individual factor recommendations
    recommendations = []

    if CO > 7:
        recommendations.append("CO is critically high — evacuate the area immediately and check combustion equipment and ventilation systems.")
    elif CO > 5:
        recommendations.append("CO is elevated — inspect fuel-burning equipment and increase fresh air supply immediately.")
    elif CO > 3:
        recommendations.append("CO is slightly above normal — monitor closely and ensure adequate ventilation.")

    if NO2 > 150:
        recommendations.append("NO2 is critically high — evacuate and identify the combustion or industrial source immediately.")
    elif NO2 > 100:
        recommendations.append("NO2 is elevated — reduce traffic or industrial activity nearby and open windows or activate ventilation.")
    elif NO2 > 70:
        recommendations.append("NO2 is slightly elevated — increase air circulation and limit exposure time for sensitive workers.")

    if NOx > 350:
        recommendations.append("NOx is critically high — halt nearby vehicle or industrial operations and ventilate immediately.")
    elif NOx > 250:
        recommendations.append("NOx is elevated — check nearby diesel engines or industrial burners and improve air circulation.")
    elif NOx > 150:
        recommendations.append("NOx is slightly above normal — monitor and reduce exposure for workers with respiratory conditions.")

    if Benzene > 25:
        recommendations.append("Benzene is critically high — this is a serious carcinogen risk. Evacuate immediately and identify the chemical source.")
    elif Benzene > 15:
        recommendations.append("Benzene is elevated — check for solvent, fuel or chemical leaks and ventilate the space urgently.")
    elif Benzene > 8:
        recommendations.append("Benzene is slightly above safe levels — inspect chemical storage areas and improve ventilation.")

    if Temperature > 29:
        recommendations.append("Temperature is critically high — activate air conditioning immediately and provide cooling stations for workers.")
    elif Temperature > 25:
        recommendations.append("Temperature is elevated — turn on air conditioning or fans and ensure workers stay hydrated.")
    elif Temperature > 22:
        recommendations.append("Temperature is slightly warm — monitor comfort levels and consider adjusting the thermostat.")

    if Humidity > 85:
        recommendations.append("Humidity is critically high — activate dehumidifiers immediately to prevent mould and heat stress.")
    elif Humidity > 70:
        recommendations.append("Humidity is elevated — run dehumidifiers and improve air circulation to reduce moisture buildup.")
    elif Humidity > 60:
        recommendations.append("Humidity is slightly high — monitor and consider running a dehumidifier.")

    # All factors high — evacuation override
    high_count = sum([
        CO > 5, NO2 > 100, NOx > 250, Benzene > 15,
        Temperature > 25, Humidity > 70
    ])

    if high_count >= 4:
        recommendations = ["MULTIPLE CRITICAL FACTORS DETECTED — Evacuate all workers immediately. Contact emergency services and conduct a full environmental inspection before re-entry."]

    # No issues
    if not recommendations:
        recommendations.append("All pollutant levels are within safe limits. Conditions are optimal for worker productivity.")

    # Risk classification
    if productivity >= 80:
        risk = "Safe"
        risk_color = "green"
    elif productivity >= 60:
        risk = "Moderate Risk"
        risk_color = "orange"
    else:
        risk = "High Risk"
        risk_color = "red"

    # Results
    st.markdown("## Prediction Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown(f"### Air Quality Risk: **:{risk_color}[{risk}]**")
        st.markdown(f"### Predicted Productivity Score: **{productivity:.1f}%**")

        st.markdown("#### Recommendations:")
        for rec in recommendations:
            st.markdown(rec)

    with res_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=productivity,
            title={'text': "Worker Productivity Score", 'font': {'color': 'white'}},
            number={'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': risk_color},
                'bgcolor': '#1a1a2e',
                'steps': [
                    {'range': [0, 60], 'color': "#3d0000"},
                    {'range': [60, 80], 'color': "#3d3000"},
                    {'range': [80, 100], 'color': "#003d00"}
                ],
                'threshold': {
                    'line': {'color': "#00d4ff", 'width': 4},
                    'thickness': 0.75,
                    'value': productivity
                }
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Pollutant Impact Summary")
    impact_data = pd.DataFrame({
        'Pollutant': ['CO', 'NO2', 'NOx', 'Benzene', 'Temperature', 'Humidity'],
        'Your Reading': [CO, NO2, NOx, Benzene, Temperature, Humidity],
        'Safe Threshold': [5.0, 100.0, 250.0, 8.0, 25.0, 70.0],
        'Status': [
            'High' if CO > 5 else 'Safe',
            'High' if NO2 > 100 else 'Safe',
            'High' if NOx > 250 else 'Safe',
            'High' if Benzene > 8 else 'Safe',
            'High' if Temperature > 25 else 'Safe',
            'High' if Humidity > 70 else 'Safe',
        ]
    })
    st.dataframe(impact_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with Python and Streamlit.")
st.markdown("*In production, sensor readings would be pulled live from IoT devices via API.*")

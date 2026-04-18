import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}
.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center; color:#00ffd5;'>🚦 Traffic AI Dashboard</h1>
<p style='text-align:center; color:gray;'>AI-powered Traffic Analysis & Visualization</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚦 Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx"])

# ---------------- LOAD DATA ----------------
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    file_path = os.path.join("data", "traffic_data.xlsx")
    df = pd.read_excel(file_path)

# ---------------- VALIDATION ----------------
required_cols = ["time", "vehicle_count", "congestion_level"]
if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain: time, vehicle_count, congestion_level")
    st.stop()

# ---------------- LOADING ----------------
with st.spinner("Processing data..."):
    time.sleep(1)

# ---------------- FILTER ----------------
times = list(df["time"])
start, end = st.sidebar.select_slider(
    "Select Time Range",
    options=times,
    value=(times[0], times[-1])
)

df_filtered = df[(df["time"] >= start) & (df["time"] <= end)]

if df_filtered.empty:
    st.warning("No data for selected range")
    st.stop()

latest = df_filtered.iloc[-1]

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("⏰ Time", latest["time"])
col2.metric("🚗 Vehicles", int(latest["vehicle_count"]))
col3.metric("🚦 Traffic", latest["congestion_level"])

st.markdown("---")

# ---------------- SUMMARY ----------------
st.subheader("📊 Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df_filtered))
c2.metric("Average Vehicles", int(df_filtered["vehicle_count"].mean()))
c3.metric("Max Vehicles", int(df_filtered["vehicle_count"].max()))

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analytics", "🗺️ Map"])

# ================= TAB 1 =================
with tab1:
    st.subheader("📊 Traffic Data")
    st.dataframe(df_filtered, use_container_width=True)

    st.subheader("📈 Trend")
    fig_line = px.line(df_filtered, x="time", y="vehicle_count", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("📊 Visual Analytics")

    # Bar Chart
    fig_bar = px.bar(df_filtered, x="time", y="vehicle_count", color="vehicle_count")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Area Chart
    fig_area = px.area(df_filtered, x="time", y="vehicle_count")
    st.plotly_chart(fig_area, use_container_width=True)

    # Histogram
    fig_hist = px.histogram(df_filtered, x="vehicle_count", nbins=10)
    st.plotly_chart(fig_hist)

    # Scatter
    fig_scatter = px.scatter(df_filtered, x="time", y="vehicle_count",
                             color="congestion_level", size="vehicle_count")
    st.plotly_chart(fig_scatter)

    # Moving Average
    df_filtered["moving_avg"] = df_filtered["vehicle_count"].rolling(3).mean()
    fig_ma = px.line(df_filtered, x="time", y=["vehicle_count", "moving_avg"])
    st.plotly_chart(fig_ma)

    # Gauge
    value = int(latest["vehicle_count"])
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Current Vehicles"},
        gauge={
            'axis': {'range': [0, 150]},
            'steps': [
                {'range': [0, 60], 'color': "green"},
                {'range': [60, 100], 'color': "orange"},
                {'range': [100, 150], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)

    # AI Prediction
    from sklearn.linear_model import LinearRegression

    df_model = df_filtered.copy()
    df_model["time_num"] = range(len(df_model))

    model = LinearRegression()
    model.fit(df_model[["time_num"]], df_model["vehicle_count"])

    prediction = model.predict([[len(df_model)]])[0]

    st.success(f"🤖 Predicted Vehicles (Next 30 mins): {int(prediction)}")

# ================= TAB 3 =================
with tab3:
    st.subheader("🗺️ Traffic Map")

    m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)

    for i in range(len(df_filtered)):
        lat = 28.6139 + i * 0.001
        lon = 77.2090 + i * 0.001
        vc = int(df_filtered.iloc[i]["vehicle_count"])

        if vc > 120:
            color = "red"
        elif vc > 70:
            color = "orange"
        else:
            color = "green"

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=f"{vc} vehicles",
            color=color,
            fill=True
        ).add_to(m)

    st.components.v1.html(m._repr_html_(), height=500)

# ---------------- PEAK ----------------
st.markdown("---")
peak = df_filtered.loc[df_filtered["vehicle_count"].idxmax()]
low = df_filtered.loc[df_filtered["vehicle_count"].idxmin()]

st.success(f"🚨 Peak: {peak['time']} ({peak['vehicle_count']})")
st.info(f"📉 Lowest: {low['time']} ({low['vehicle_count']})")

# ---------------- DOWNLOAD ----------------
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Data", csv, "traffic.csv")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>Built by Ishita Sharma</p>
""", unsafe_allow_html=True)
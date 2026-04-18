import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
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
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center; color:#00ffd5;'>🚦 Traffic AI Dashboard</h1>
<p style='text-align:center; color:gray;'>AI-based Traffic Analysis & Prediction</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚦 Traffic AI")

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
    st.error("❌ Dataset must contain: time, vehicle_count, congestion_level")
    st.stop()

# ---------------- LOADING ----------------
with st.spinner("Analyzing traffic data..."):
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

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df_filtered))
col2.metric("Average Vehicles", int(df_filtered["vehicle_count"].mean()))
col3.metric("Max Vehicles", int(df_filtered["vehicle_count"].max()))

st.markdown("---")

# ---------------- INSIGHTS ----------------
st.subheader("🧠 Insights")

avg = df_filtered["vehicle_count"].mean()

if avg > 100:
    st.warning("🚨 High traffic detected")
elif avg > 60:
    st.info("⚠️ Moderate traffic")
else:
    st.success("✅ Smooth traffic")

# ---------------- CHART ----------------
st.subheader("📈 Traffic Trend")

fig = px.line(df_filtered, x="time", y="vehicle_count", markers=True)
st.plotly_chart(fig, use_container_width=True)

# ---------------- PIE ----------------
st.subheader("📊 Traffic Distribution")

fig2 = px.pie(df_filtered, names="congestion_level")
st.plotly_chart(fig2)

# ---------------- AI PREDICTION ----------------
from sklearn.linear_model import LinearRegression

df_model = df_filtered.copy()
df_model["time_num"] = range(len(df_model))

X = df_model[["time_num"]]
y = df_model["vehicle_count"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(np.array([[len(df_model)]]))[0]

st.markdown(f"""
<div style="background:#111827;padding:20px;border-radius:10px;text-align:center">
<h3 style='color:#00ffd5;'>🤖 AI Prediction</h3>
<h1 style='color:white;'>{int(prediction)} Vehicles</h1>
<p style='color:gray;'>Next 30 minutes</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- MAP ----------------
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

st.markdown("🟢 Low | 🟠 Medium | 🔴 High")

st.markdown("---")

# ---------------- PEAK & LOW ----------------
st.subheader("📉 Traffic Analysis")

peak = df_filtered.loc[df_filtered["vehicle_count"].idxmax()]
low = df_filtered.loc[df_filtered["vehicle_count"].idxmin()]

st.success(f"🚨 Peak: {peak['time']} ({peak['vehicle_count']} vehicles)")
st.info(f"📉 Lowest: {low['time']} ({low['vehicle_count']} vehicles)")

# ---------------- DOWNLOAD ----------------
csv = df_filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Filtered Data",
    csv,
    "traffic_data.csv",
    "text/csv"
)

# ---------------- RAW DATA ----------------
with st.expander("📄 View Raw Data"):
    st.dataframe(df)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>Built by Ishita Sharma 🚀</p>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import folium
from streamlit_folium import st_folium

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #00ffcc;
        text-align: center;
    }
    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0 0 10px rgba(0,255,204,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<p class="title">🚦 Traffic AI Dashboard</p>', unsafe_allow_html=True)

# ------------------ FILE UPLOAD ------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

# ------------------ LOAD DATA ------------------
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_excel("data/traffic_data.xlsx")

st.success("✅ Dataset Loaded Successfully")

# ------------------ FILTER ------------------
st.sidebar.header("🎯 Filter Data")

time_options = df["time"].unique()
selected_time = st.sidebar.multiselect("Select Time", time_options, default=time_options)

df_filtered = df[df["time"].isin(selected_time)]

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)

if not df_filtered.empty:
    latest = df_filtered.iloc[-1]
else:
    st.warning("⚠️ No data available for selected filter")
    st.stop()

with col1:
    st.markdown('<div class="card">⏰ Time<br><h2>{}</h2></div>'.format(latest["time"]), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">🚗 Vehicles<br><h2>{}</h2></div>'.format(latest["vehicle_count"]), unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">🚦 Traffic<br><h2>{}</h2></div>'.format(latest["congestion_level"]), unsafe_allow_html=True)

# ------------------ DATA TABLE ------------------
st.subheader("📊 Traffic Data")
st.dataframe(df_filtered, use_container_width=True)

# ------------------ CHART ------------------
st.subheader("📈 Traffic Trend")

fig, ax = plt.subplots()
ax.plot(df_filtered["time"], df_filtered["vehicle_count"], marker='o')
plt.xticks(rotation=45)

st.pyplot(fig)
# ================= HEATMAP =================
import seaborn as sns

st.subheader("🔥 Traffic Heatmap")

df_heat = df_filtered.copy()
df_heat["hour"] = df_heat["time"].str.split(":").str[0].astype(int)

pivot = df_heat.pivot_table(values="vehicle_count", index="hour", aggfunc="mean")

fig2, ax2 = plt.subplots()
sns.heatmap(pivot, cmap="coolwarm", annot=True, ax=ax2)

st.pyplot(fig2)


# ================= AI PREDICTION =================
from sklearn.linear_model import LinearRegression
import numpy as np

st.subheader("🧠 AI Prediction (Next 30 mins)")

df_model = df_filtered.copy()
df_model["time_num"] = range(len(df_model))

X = df_model[["time_num"]]
y = df_model["vehicle_count"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[len(df_model)]])[0]

st.success(f"🚗 Predicted Vehicles: {int(prediction)}")


# ================= MAP =================
st.subheader("🗺️ Traffic Map")

try:
    m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)

    for i in range(len(df_filtered)):
        lat = 28.6139 + (i * 0.001)
        lon = 77.2090 + (i * 0.001)

        vehicle_count = int(df_filtered.iloc[i]["vehicle_count"])

        color = "red" if vehicle_count > 100 else "green"

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=f"Vehicles: {vehicle_count}",
            color=color,
            fill=True
        ).add_to(m)

    # ✅ FINAL FIX (NO st_folium)
    st.components.v1.html(m._repr_html_(), height=500)

except Exception as e:
    st.error(f"Map error: {e}")
# ================= PEAK HOUR =================
st.subheader("📉 Peak Hour")

peak = df_filtered.loc[df_filtered["vehicle_count"].idxmax()]

st.error(f"🚨 Peak Time: {peak['time']} with {peak['vehicle_count']} vehicles")
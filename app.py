import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import folium

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background-color:#1f2937;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow: 0 0 15px rgba(0,255,204,0.2);
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align: center; color: #00ffd5;'>🚦 Traffic AI Dashboard</h1>
<p style='text-align: center; color: gray;'>
Analyze traffic patterns, detect peak hours, and predict congestion using AI.
</p>
""", unsafe_allow_html=True)

# ------------------ FILE UPLOAD ------------------
st.sidebar.markdown("## 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

# ------------------ LOAD DATA ------------------
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    file_path = os.path.join("data", "traffic_data.xlsx")
    df = pd.read_excel(file_path)

st.success("✅ Dataset Loaded Successfully")

# ------------------ FILTER ------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Filter Data")

time_options = df["time"].unique()
selected_time = st.sidebar.multiselect(
    "Select Time",
    time_options,
    default=time_options
)

df_filtered = df[df["time"].isin(selected_time)]

if df_filtered.empty:
    st.warning("⚠️ No data available for selected filter")
    st.stop()

latest = df_filtered.iloc[-1]

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)

def card(title, value):
    return f"""
    <div class="card">
        <h4 style='color:gray'>{title}</h4>
        <h2 style='color:#00ffd5'>{value}</h2>
    </div>
    """

with col1:
    st.markdown(card("⏰ Time", latest["time"]), unsafe_allow_html=True)

with col2:
    st.markdown(card("🚗 Vehicles", latest["vehicle_count"]), unsafe_allow_html=True)

with col3:
    st.markdown(card("🚦 Traffic", latest["congestion_level"]), unsafe_allow_html=True)

st.markdown("---")

# ------------------ DATA TABLE ------------------
st.subheader("📊 Traffic Data")
st.dataframe(df_filtered, use_container_width=True)

# ------------------ LINE CHART ------------------
st.subheader("📈 Traffic Trend")

fig, ax = plt.subplots()
ax.plot(df_filtered["time"], df_filtered["vehicle_count"], marker='o')
ax.set_facecolor("#0e1117")
ax.grid(True, linestyle="--", alpha=0.3)
plt.xticks(rotation=45)

st.pyplot(fig)

st.markdown("---")

# ------------------ HEATMAP ------------------
st.subheader("🔥 Traffic Heatmap")

df_heat = df_filtered.copy()
df_heat["hour"] = df_heat["time"].str.split(":").str[0].astype(int)

pivot = df_heat.pivot_table(values="vehicle_count", index="hour", aggfunc="mean")

fig2, ax2 = plt.subplots()
sns.heatmap(pivot, cmap="YlOrRd", annot=True, linewidths=0.5, ax=ax2)

st.pyplot(fig2)

st.markdown("---")

# ------------------ AI PREDICTION ------------------
st.subheader("🧠 AI Prediction (Next 30 mins)")

df_model = df_filtered.copy()
df_model["time_num"] = range(len(df_model))

X = df_model[["time_num"]]
y = df_model["vehicle_count"]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[len(df_model)]])[0]

st.success(f"🚗 Predicted Vehicles: {int(prediction)}")

st.markdown("---")

# ------------------ MAP ------------------
st.subheader("🗺️ Traffic Map")

try:
    m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)

    for i in range(len(df_filtered)):
        lat = 28.6139 + (i * 0.001)
        lon = 77.2090 + (i * 0.001)

        vehicle_count = int(df_filtered.iloc[i]["vehicle_count"])

        if vehicle_count > 120:
            color = "red"
        elif vehicle_count > 70:
            color = "orange"
        else:
            color = "green"

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=f"Vehicles: {vehicle_count}",
            color=color,
            fill=True
        ).add_to(m)

    st.components.v1.html(m._repr_html_(), height=500)

except Exception as e:
    st.error(f"Map error: {e}")

st.markdown("---")

# ------------------ BAR + PIE ------------------
st.subheader("📊 Traffic Analysis")

col1, col2 = st.columns(2)

with col1:
    fig3, ax3 = plt.subplots()
    ax3.bar(df_filtered["time"], df_filtered["vehicle_count"])
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with col2:
    counts = df_filtered["congestion_level"].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.pie(counts, labels=counts.index, autopct="%1.1f%%")
    st.pyplot(fig4)

st.markdown("---")

# ------------------ PEAK HOUR ------------------
st.subheader("📉 Peak Hour")

peak = df_filtered.loc[df_filtered["vehicle_count"].idxmax()]

st.markdown(f"""
<div style="
    background: linear-gradient(90deg, #ff4b4b, #ff7b00);
    padding:15px;
    border-radius:10px;
    color:white;
    text-align:center;
    font-size:18px;
">
🚨 Peak Time: {peak['time']} with {peak['vehicle_count']} vehicles
</div>
""", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray'>
Built by Ishita Sharma | Traffic AI Dashboard
</p>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import folium

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")

# ---------------- CUSTOM UI ----------------
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
    transition:0.3s;
}
.card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,204,0.5);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center; color:#00ffd5;'>🚦 Traffic AI Dashboard</h1>
<p style='text-align:center; color:gray;'>AI Powered Traffic Analysis & Prediction</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🚦 Traffic AI")
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

# ---------------- LOAD DATA ----------------
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    file_path = os.path.join("data", "traffic_data.xlsx")
    df = pd.read_excel(file_path)

# ---------------- FILTER ----------------
time_options = df["time"].unique()
selected_time = st.sidebar.multiselect("Select Time", time_options, default=time_options)

df_filtered = df[df["time"].isin(selected_time)]

if df_filtered.empty:
    st.warning("No data")
    st.stop()

latest = df_filtered.iloc[-1]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "🗺️ Map"])

# ================= TAB 1 =================
with tab1:
    col1, col2, col3 = st.columns(3)

    def card(title, value):
        return f"""
        <div class="card">
            <h4 style='color:gray'>{title}</h4>
            <h2 style='color:#00ffd5'>{value}</h2>
        </div>
        """

    col1.markdown(card("⏰ Time", latest["time"]), unsafe_allow_html=True)
    col2.markdown(card("🚗 Vehicles", latest["vehicle_count"]), unsafe_allow_html=True)
    col3.markdown(card("🚦 Traffic", latest["congestion_level"]), unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 Data")
    st.dataframe(df_filtered, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("📈 Traffic Trend")

    fig = px.line(df_filtered, x="time", y="vehicle_count", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Heatmap")

    df_heat = df_filtered.copy()
    df_heat["hour"] = df_heat["time"].str.split(":").str[0].astype(int)
    pivot = df_heat.pivot_table(values="vehicle_count", index="hour")

    fig2, ax2 = plt.subplots()
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📊 Distribution")

    fig3 = px.pie(df_filtered, names="congestion_level")
    st.plotly_chart(fig3)

    # AI Prediction
    from sklearn.linear_model import LinearRegression

    df_model = df_filtered.copy()
    df_model["time_num"] = range(len(df_model))

    X = df_model[["time_num"]]
    y = df_model["vehicle_count"]

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(np.array([[len(df_model)]]))[0]

    st.markdown(f"""
    <div style="background:#111827;padding:20px;border-radius:12px;text-align:center">
        <h3 style='color:#00ffd5;'>🤖 AI Prediction</h3>
        <h1 style='color:white;'>{int(prediction)} Vehicles</h1>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("🗺️ Traffic Map")

    m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)

    for i in range(len(df_filtered)):
        lat = 28.6139 + (i * 0.001)
        lon = 77.2090 + (i * 0.001)
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

    st.markdown("""
    🟢 Low Traffic | 🟠 Medium | 🔴 High
    """)

# ---------------- PEAK ----------------
st.markdown("---")

peak = df_filtered.loc[df_filtered["vehicle_count"].idxmax()]

st.markdown(f"""
<div style="background:linear-gradient(90deg,#ff4b4b,#ff7b00);
padding:15px;border-radius:10px;text-align:center;color:white;">
🚨 Peak Time: {peak['time']} with {peak['vehicle_count']} vehicles
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>Built by Ishita Sharma</p>
""", unsafe_allow_html=True)
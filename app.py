import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Banglish Depression Model Evaluation",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #f5f7fb; }

.header {
    background: linear-gradient(135deg, #5f2c82, #49a09d);
    padding: 2.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
}

.header h1 {
    font-size: 2.7rem;
    margin-bottom: 0.3rem;
}

.header p {
    font-size: 1.1rem;
    opacity: 0.95;
}

.section {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

with open("model_metrics.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data).T.reset_index()
df.rename(columns={"index": "Model"}, inplace=True)

st.markdown("""
<div class="header">
    <h1>🧠 Banglish Depression Detection</h1>
    <p>Performance Evaluation of Machine Learning & Deep Learning Models</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section'>🔍 Select Model</div>", unsafe_allow_html=True)

selected_model = st.selectbox("", df["Model"])
row = df[df["Model"] == selected_model].iloc[0]

st.markdown("<div class='section'>📊 Performance Metrics</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{row['Accuracy']*100:.2f}%")
c2.metric("Precision", f"{row['Precision']*100:.2f}%")
c3.metric("Recall", f"{row['Recall']*100:.2f}%")
c4.metric("F1-Score", f"{row['F1-score']*100:.2f}%")

st.divider()

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8,5))
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [row[m] for m in labels]
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v*100:.2f}%", ha="center")
    ax.set_title(f"{selected_model} Metrics")
    st.pyplot(fig)

with right:
    ranked = df.sort_values("F1-score", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    bars = ax2.barh(ranked["Model"], ranked["F1-score"])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    for i, v in enumerate(ranked["F1-score"]):
        ax2.text(v + 0.02, i, f"{v*100:.2f}%", va="center")
    ax2.set_title("Model Ranking (F1-Score)")
    st.pyplot(fig2)

st.divider()

st.markdown("<div class='section'>📊 Compare All Models</div>", unsafe_allow_html=True)

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(df["Model"]))
w = 0.2

fig3, ax3 = plt.subplots(figsize=(12,6))

for i, m in enumerate(metrics):
    ax3.bar(x + i*w, df[m], w, label=m)

ax3.set_xticks(x + w*1.5)
ax3.set_xticklabels(df["Model"], rotation=25, ha="right")
ax3.set_ylim(0, 1.05)
ax3.legend()
ax3.set_title("Overall Performance Comparison")

for i, m in enumerate(metrics):
    for j, v in enumerate(df[m]):
        ax3.text(j + i*w, v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=8)

st.pyplot(fig3)

st.divider()

with st.expander("📋 View Complete Metrics Table"):
    st.dataframe(df.set_index("Model").style.format("{:.2%}"), use_container_width=True)

st.markdown("""
<hr>
<div style="text-align:center; color:#777;">
    <p>Department of Computer Science & Engineering</p>
    <p>Banglish Depression Detection – Model Evaluation Dashboard</p>
</div>
""", unsafe_allow_html=True)

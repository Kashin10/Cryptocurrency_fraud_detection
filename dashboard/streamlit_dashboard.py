
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Crypto Fraud Dashboard",
    layout="wide"
)

st.title(
    "Explainable GNN Cryptocurrency Fraud Detection"
)

scores = np.random.uniform(0.2, 1.0, 100)

df = pd.DataFrame({
    "transaction": range(100),
    "fraud_score": scores
})

fig = px.scatter(
    df,
    x="transaction",
    y="fraud_score",
    title="Fraud Scores"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

threshold = st.slider(
    "Fraud Threshold",
    0.0,
    1.0,
    0.8
)

st.dataframe(
    df[df["fraud_score"] > threshold]
)

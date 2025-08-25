# streamlit_ui.py
"""
Run: streamlit run streamlit_ui.py
"""
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feedback AI Ops (Gemini)", layout="wide")
st.title("Intelligent User Feedback Analysis — Monitor & Override (Gemini)")

generated = "generated_tickets.csv"
logfile = "processing_log.csv"
metrics = "metrics.csv"

colp = st.columns(3)
with colp[0]:
    st.write("Generated Tickets CSV:", os.path.abspath(generated))
with colp[1]:
    st.write("Processing Log CSV:", os.path.abspath(logfile))
with colp[2]:
    st.write("Metrics CSV:", os.path.abspath(metrics))

def load_df(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

tickets_df = load_df(generated)
log_df = load_df(logfile)
metrics_df = load_df(metrics)

st.header("Metrics")
st.dataframe(metrics_df, use_container_width=True)

st.header("Processing Log")
st.dataframe(log_df, use_container_width=True)

st.header("Tickets")
if not tickets_df.empty:
    edited = st.data_editor(
        tickets_df,
        num_rows="dynamic",
        use_container_width=True,
        key="tickets_editor"
    )
    st.info("Edit priorities/titles before exporting.")
    if st.button("Export Edited Tickets"):
        edited.to_csv("generated_tickets.edited.csv", index=False)
        st.success("Exported to generated_tickets.edited.csv")
else:
    st.warning("No tickets found. Run the pipeline first.")

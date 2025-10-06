import streamlit as st, pandas as pd, io
st.set_page_config(page_title="Upload Test", layout="wide")
st.title("Upload Test (CSV/XLSX)")
up = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx"])
if up:
    buf = io.BytesIO(up.read())
    try:
        df = pd.read_csv(buf)
    except Exception:
        buf.seek(0); df = pd.read_excel(buf)
    st.success(f"Rows: {len(df)} | Cols: {len(df.columns)}")
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("Pick a small file to test.")

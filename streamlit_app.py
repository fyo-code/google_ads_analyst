import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Ads Analyst", layout="wide")
st.title("AI Ads Assistant — Google (MVP)")

def load_df(upload):
    """Read CSV or XLSX reliably on Streamlit Cloud."""
    name = upload.name.lower()
    raw = upload.read()            # read once
    buf = io.BytesIO(raw)

    if name.endswith(".csv"):
        # try standard CSV; if dialect is odd, fall back to python engine
        try:
            buf.seek(0)
            return pd.read_csv(buf)
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, sep=None, engine="python")
    elif name.endswith((".xlsx", ".xls")):
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {name}")

uploaded = st.file_uploader(
    "Upload CSV or XLSX",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

dfs = []
if uploaded:
    for up in uploaded:
        try:
            df = load_df(up)
            st.success(f"Loaded **{up.name}** — {len(df):,} rows × {len(df.columns)} cols")
            st.dataframe(df.head(10))
            dfs.append(df)
        except Exception as e:
            st.error(f"Failed to parse **{up.name}** → {e}")

# (…below this, keep your existing analysis code that uses `dfs` / combined data…)

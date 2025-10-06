# app_google.py — Google Ads AI Analyst (MVP+charts)
# Run:  streamlit run app_google.py

import os
import io
import json
import math
import pandas as pd
import numpy as np
import streamlit as st

# ---------- UI ----------
st.set_page_config(page_title="AI Ads Assistant — Google", layout="wide")
st.title("AI Ads Assistant — Google (MVP)")
st.caption("Upload one or more Google Ads CSVs → set an aim (optional) → AI findings + visuals.")

# ---------- Helpers: I/O & normalization ----------
def read_google_csv(file) -> pd.DataFrame:
    """Robust read for Google Ads exported CSVs (various dialects)."""
    # Some exports include BOM / different delimiters / localized decimals.
    raw = file.read()
    buf = io.BytesIO(raw)
    try:
        df = pd.read_csv(buf, sep=",", engine="python")
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, sep=";", engine="python")
    # strip whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    # drop completely empty columns
    df = df.dropna(axis=1, how="all")
    return df

def to_records(df: pd.DataFrame):
    """Make a records list that is JSON-safe (numbers as numbers, not strings with %)."""
    out = []
    for _, r in df.iterrows():
        rec = {}
        for c, v in r.items():
            if isinstance(v, str):
                rec[c] = v.strip()
            else:
                rec[c] = v
        out.append(rec)
    return out

def coerce_numeric_inplace(df: pd.DataFrame, cols: list[str]):
    for col in cols:
        if col and col in df:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace("%", "", regex=False)
                df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

def find_col(df: pd.DataFrame, patterns: list[str]):
    for c in df.columns:
        cl = c.lower()
        if any(p in cl for p in patterns):
            return c
    return None

# ---------- Context builder ----------
def build_context(uploaded_files):
    """
    Build a unified context from multiple Google Ads CSVs.
    Returns: (context_dict, theme_df_for_quick_peek)
    """
    tables = {}
    theme_df = None

    for up in uploaded_files:
        name = up.name
        try:
            df = read_google_csv(up)
            # Keep a small preview for debug
            if theme_df is None:
                theme_df = df.head(10).copy()
            tables[name] = to_records(df)
        except Exception as e:
            tables[name] = [{"error": f"Failed to parse: {e}"}]

    context = {
        "platform": "google_ads",
        "tables": tables,
        "notes": {},
    }
    return context, theme_df

# ---------- System prompt (Google-tuned, goal-aware & flexible) ----------
SYSTEM_PROMPT = """
You are a senior performance marketing analyst for Google Ads.
Your job: speed up reporting, catch human errors, and surface *concrete* insights from any mix of reports
(keywords, search terms, campaigns, asset groups, placements, devices, audiences, products, landing pages, etc.).
Scan the *entire* dataset (no pruning). If something important is missing, say so briefly and adapt using best proxies.

GOAL-AWARE EVALUATION
Judge performance by the entity’s stated or inferred goal (Sales/Leads/Traffic/Awareness/Local).
Use a primary KPI hierarchy per goal and fall back if the top metric is absent. Always state when using a proxy.

- SALES: ROAS → CPA (purchase) → Conversion value → Conversions → CVR → CPC
- LEADS: CPL → Leads → CVR → CPC
- TRAFFIC: CPC → CTR → Clicks → LPV → CPM
- AWARENESS: Impressions/Reach → CPM → Frequency → View rate
- LOCAL/STORE: Store visits → Cost/Visit → Impressions/Reach

If no goal is provided, infer it from available metrics (e.g., presence of conversions/value → Sales/Leads;
only click metrics → Traffic; impression-only → Awareness). Say what you inferred.

FORMAT (plain English, no JSON). Write these 6 sections in order. Name specific entities and always include numbers.
1) Findings — 5–10 bullets with concrete numbers (overall + key entities). Mention inferred goals & limitations.
2) Red flags — 3–7 bullets of human-error catchers: spend>0 & 0 conv; broken tracking (clicks w/out impressions); outlier CPC/CPM; wrong objective vs outcome; extreme query mismatch.
3) Correlations & Patterns — 3–5 bullets: keywords/themes that beat averages; device/geo/daypart lift; placements (Search/Partners/YouTube/Discover/Maps) differences; diminishing returns (spend↑ ROAS↓); CTR↔CPA or CVR↔ROAS relationships.
4) Winners & Losers — top/bottom 3 by goal-aware KPI (per goal if mixed). State KPI used and value vs avg/peer.
5) Suggestions / Action plan — 4–7 items. For each: What to do, Why (metrics), How to test (A/B or budget move, timeframe), Priority.
6) Watchlist — 5 KPIs to monitor next period (e.g., CTR, CPC, CVR, CPA/ROAS, Spend distribution). One-line rationale helpful.

STYLE
Short, decision-oriented bullets. Quantify everything. Never invent metrics. Use proxies only when necessary and say so.
If some sections are thin, say why and focus on strong signals. Also surface any extra pattern that matters, beyond the checklist.
"""

# ---------- LLM caller with graceful failure ----------
def call_llm(context: dict) -> dict:
    """Call OpenAI; return dict with '_raw' (markdown) or empty on failure."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return {}
        client = OpenAI(api_key=api_key)
        user_content = json.dumps(context, ensure_ascii=False)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": f"Here is the parsed Google Ads data:\n{user_content}\n\nReturn the 6 sections now."}
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        return {"_raw": text}
    except Exception:
        return {}

# ---------- Local rule-based fallback (always shows something) ----------
def simple_rule_analysis(context: dict):
    findings, red_flags, patterns = [], [], []
    winners, losers, actions, watchlist = [], [], [], []

    # choose first non-empty table
    df = None
    for name, recs in context.get("tables", {}).items():
        if recs and "error" not in recs[0]:
            df = pd.DataFrame(recs)
            break

    if df is None or df.empty:
        return {
            "findings": ["No readable tables. Upload a CSV that includes clicks/imp/cost at minimum."],
            "red_flags": [], "patterns": [], "winners": [], "losers": [],
            "actions": [], "watchlist": ["CTR", "CPC", "CPA", "ROAS", "Spend"]
        }

    # likely columns
    name_col = find_col(df, ["keyword","search term","campaign","ad group","asset group","product","landing page","network","asset","ad"])
    clicks   = find_col(df, ["clicks"])
    impr     = find_col(df, ["impr","impressions"])
    cost     = find_col(df, ["cost"])
    conv     = find_col(df, ["conversions","all conv."])
    convval  = find_col(df, ["conv. value","all conv. value","conversion value"])
    ctr      = find_col(df, ["ctr"])
    cpc      = find_col(df, ["avg. cpc","cpc"])
    cvr      = find_col(df, ["conv. rate","conversion rate"])
    cpa      = find_col(df, ["cpa","cost / conv","cost/conv"])
    roas     = find_col(df, ["roas","conv. value / cost","value / cost"])

    coerce_numeric_inplace(df, [clicks, impr, cost, conv, convval, ctr, cpc, cvr, cpa, roas])

    # derive metrics if missing
    if ctr is None and clicks and impr and (df[impr] > 0).any():
        df["__ctr"] = df[clicks] / df[impr]; ctr = "__ctr"
    if cpc is None and cost and clicks and (df[clicks] > 0).any():
        df["__cpc"] = df[cost] / df[clicks]; cpc = "__cpc"
    if cvr is None and conv and clicks and (df[clicks] > 0).any():
        df["__cvr"] = df[conv] / df[clicks]; cvr = "__cvr"
    if roas is None and convval and cost and (df[cost] > 0).any():
        df["__roas"] = df[convval] / df[cost]; roas = "__roas"
    if cpa is None and cost and conv and (df[conv] > 0).any():
        df["__cpa"] = df[cost] / df[conv]; cpa = "__cpa"

    # findings
    if cost in df:
        findings.append(f"Total cost: {df[cost].sum():,.2f}")
    if clicks in df and impr in df and df[impr].sum() > 0:
        findings.append(f"Overall CTR: {df[clicks].sum()/df[impr].sum():.2%}")
    if conv in df and clicks in df and df[clicks].sum() > 0:
        findings.append(f"Overall CVR: {df[conv].sum()/df[clicks].sum():.2%}")
    if convval in df and cost in df and df[cost].sum() > 0:
        findings.append(f"Overall ROAS: {df[convval].sum()/df[cost].sum():.2f}")

    # red flags
    if cost in df and conv in df:
        zero_conv_spend = df[(df[cost] > 0) & ((df[conv].fillna(0)) == 0)].head(5)
        for _, r in zero_conv_spend.iterrows():
            label = r.get(name_col, "entity")
            red_flags.append(f"{label}: spend {r[cost]:.2f} with 0 conversions")

    # winners/losers by best available metric
    df2 = df.copy()
    if roas in df2:
        df2 = df2.sort_values(roas, ascending=False)
        winners = [f"{r.get(name_col,'entity')} — ROAS {r[roas]:.2f}" for _, r in df2.head(3).iterrows()]
        losers  = [f"{r.get(name_col,'entity')} — ROAS {r[roas]:.2f}" for _, r in df2.tail(3).iterrows()]
    elif cpa in df2:
        df2 = df2[df2[cpa] > 0].sort_values(cpa, ascending=True)
        winners = [f"{r.get(name_col,'entity')} — CPA {r[cpa]:.2f}" for _, r in df2.head(3).iterrows()]
        losers  = [f"{r.get(name_col,'entity')} — CPA {r[cpa]:.2f}" for _, r in df2.tail(3).iterrows()]
    elif ctr in df2:
        df2 = df2.sort_values(ctr, ascending=False)
        winners = [f"{r.get(name_col,'entity')} — CTR {r[ctr]:.2%}" for _, r in df2.head(3).iterrows()]
        losers  = [f"{r.get(name_col,'entity')} — CTR {r[ctr]:.2%}" for _, r in df2.tail(3).iterrows()]

    if winners:
        actions.append({"title":"Scale winners", "why": winners[0], "how_to_test":"+15–25% budget; monitor CPA/ROAS 3–5 days", "priority":"high"})
    if red_flags:
        actions.append({"title":"Pause zero-conversion spenders", "why": red_flags[0], "how_to_test":"pause 5–7 days then re-evaluate", "priority":"high"})

    watchlist = ["CTR","CPC","CVR","CPA/ROAS","Spend distribution"]

    return {
        "findings": findings or ["Data parsed but limited KPIs present."],
        "red_flags": red_flags, "patterns": patterns,
        "winners": winners, "losers": losers,
        "actions": actions, "watchlist": watchlist
    }

# ---------- Charts (robust) ----------
def render_charts(context):
    import altair as alt

    st.sidebar.checkbox("Show charts (optional)", value=True, key="show_charts_google")
    if not st.session_state.get("show_charts_google"):
        return

    st.write("### Visual Performance Overview")

    # choose first sensible table
    df = None
    for name, recs in context.get("tables", {}).items():
        if recs and "error" not in recs[0]:
            _df = pd.DataFrame(recs)
            if len(_df.columns) >= 3:
                df = _df
                break

    if df is None or df.empty:
        st.info("No suitable table found for charts yet — upload a table with clicks/imp/cost.")
        return

    df = df.copy()

    # column detection
    def pick(patterns): return find_col(df, patterns)

    name_col = pick(["keyword","search term","campaign","ad group","asset group","product title","item id","landing page","network","asset","ad"])
    clicks_col = pick(["clicks"])
    impr_col   = pick(["impr","impressions"])
    cost_col   = pick(["cost"])
    ctr_col    = pick(["ctr"])
    cpc_col    = pick(["avg. cpc","cpc"])
    roas_col   = pick(["roas","conv. value / cost","value / cost"])
    conv_val_col = pick(["conv. value","all conv. value","conversion value"])
    conv_col   = pick(["conversions","all conv."])
    cvr_col    = pick(["conv. rate","conversion rate"])
    cpa_col    = pick(["cpa","cost / conv","cost/conv"])

    for col in [clicks_col, impr_col, cost_col, ctr_col, cpc_col, roas_col, conv_val_col, conv_col, cvr_col, cpa_col]:
        if col and col in df:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace("%","",regex=False)
                df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not name_col:
        df["Entity"] = [f"Row {i+1}" for i in range(len(df))]
        name_col = "Entity"

    # derive metrics if missing
    if not ctr_col and clicks_col and impr_col and (df[impr_col] > 0).any():
        df["__ctr_auto"] = df[clicks_col] / df[impr_col]; ctr_col = "__ctr_auto"
    if not cpc_col and cost_col and clicks_col and (df[clicks_col] > 0).any():
        df["__cpc_auto"] = df[cost_col] / df[clicks_col]; cpc_col = "__cpc_auto"
    if not cvr_col and conv_col and clicks_col and (df[clicks_col] > 0).any():
        df["__cvr_auto"] = df[conv_col] / df[clicks_col]; cvr_col = "__cvr_auto"
    if not roas_col and conv_val_col and cost_col and (df[cost_col] > 0).any():
        df["__roas_auto"] = df[conv_val_col] / df[cost_col]; roas_col = "__roas_auto"
    if not cpa_col and cost_col and conv_col and (df[conv_col] > 0).any():
        df["__cpa_auto"] = df[cost_col] / df[conv_col]; cpa_col = "__cpa_auto"

    # limit to top spend for readability
    if cost_col in df:
        df = df.sort_values(cost_col, ascending=False)
    df = df.head(200)

    # 1) Efficiency scatter (CTR vs CPC)
    if name_col and ctr_col and cpc_col:
        st.write("#### Efficiency (CTR vs CPC)")
        scatter_df = df.dropna(subset=[ctr_col, cpc_col]).copy()
        chart = alt.Chart(scatter_df).mark_circle(size=70).encode(
            x=alt.X(f"{ctr_col}:Q", title="CTR"),
            y=alt.Y(f"{cpc_col}:Q", title="Avg. CPC"),
            tooltip=[name_col] + [c for c in [ctr_col, cpc_col, cost_col, roas_col, cpa_col, conv_col] if c in scatter_df],
        )
        if cost_col in scatter_df:
            chart = chart.encode(size=alt.Size(f"{cost_col}:Q", title="Cost", legend=None))
        if roas_col in scatter_df:
            chart = chart.encode(color=alt.Color(f"{roas_col}:Q", title="ROAS"))
        else:
            chart = chart.encode(color=alt.value("#1f77b4"))
        st.altair_chart(chart.properties(height=420), use_container_width=True)
    else:
        st.info("Need enough data to compute CTR and CPC (we try to derive them automatically).")

    # 2) Quadrants (X = CVR or CTR, Y = ROAS or 1/CPA)
    x_col = cvr_col or ctr_col
    y_col = None
    y_title = ""
    dfq = None
    if roas_col:
        y_col = roas_col
        y_title = "ROAS"
        if x_col:
            dfq = df.dropna(subset=[x_col, y_col]).copy()
    elif cpa_col:
        if x_col:
            dfq = df.dropna(subset=[x_col, cpa_col]).copy()
            if not dfq.empty:
                dfq["__inv_cpa"] = np.where(dfq[cpa_col] > 0, 1.0/dfq[cpa_col], np.nan)
                y_col = "__inv_cpa"
                y_title = "Efficiency (1/CPA)"

    if x_col and y_col and dfq is not None and not dfq.empty:
        x_med = float(dfq[x_col].median())
        y_med = float(dfq[y_col].median())

        def quadrant(row):
            if row[x_col] >= x_med and row[y_col] >= y_med: return "Winner"
            if row[x_col] >= x_med and row[y_col] <  y_med: return "Test (Cheap clicks?)"
            if row[x_col] <  x_med and row[y_col] >= y_med: return "Scale if volume exists"
            return "Loser"

        dfq["Quadrant"] = dfq.apply(quadrant, axis=1)

        st.write("#### Performance Quadrants")
        chart = alt.Chart(dfq).mark_circle(size=70).encode(
            x=alt.X(f"{x_col}:Q", title=("CVR" if x_col==cvr_col else "CTR")),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color("Quadrant:N"),
            tooltip=[name_col, x_col, y_col] + [c for c in [cost_col, roas_col, cpa_col, conv_col] if c in dfq],
        )
        if cost_col in dfq:
            chart = chart.encode(size=alt.Size(f"{cost_col}:Q", title="Cost", legend=None))

        vline = alt.Chart(pd.DataFrame({x_col:[x_med]})).mark_rule(strokeDash=[4,4], color="#888").encode(x=alt.X(f"{x_col}:Q"))
        hline = alt.Chart(pd.DataFrame({y_col:[y_med]})).mark_rule(strokeDash=[4,4], color="#888").encode(y=alt.Y(f"{y_col}:Q"))
        st.altair_chart((chart + vline + hline).properties(height=420), use_container_width=True)
    else:
        st.info("Not enough data for the quadrant chart (we try ROAS, or 1/CPA as fallback).")

# ---------- Main UI ----------
uploads = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
business_aim = st.text_input("Overall business aim (optional)", placeholder="e.g., Find the most profitable keywords, which segments to scale, and where budget is wasted.")
with st.expander("Per-entity goals/notes (optional)"):
    st.caption("You can paste free-form notes per campaign/ad group/keyword. (Optional)")
    # For MVP we keep this simple — can be wired later into context["notes"].

# Analyze button
if st.button("Analyze", type="primary", disabled=not uploads):
    context, theme_df = build_context(uploads)
    context["business_aim"] = business_aim or ""

    # persist for reruns
    st.session_state["context"] = context
    st.session_state["theme_df"] = theme_df
    st.session_state["charts_drawn_once"] = False  # reset

    with st.expander("Show JSON (debug)"):
        st.json(context)

    # Charts first
    render_charts(context)

    # AI call
    ai_output = call_llm(context)
    if not ai_output.get("_raw") and not ai_output.get("findings"):
        ai_output = simple_rule_analysis(context)

    # Render report
    if ai_output.get("_raw"):
        st.subheader("AI report")
        st.markdown(ai_output["_raw"])
    else:
        st.subheader("AI report")
        for sec, title in [
            ("findings", "Findings"),
            ("red_flags", "Red flags"),
            ("patterns", "Correlations & patterns"),
        ]:
            st.markdown(f"**{title}**")
            for x in ai_output.get(sec, []):
                st.write(f"• {x}")
        st.markdown("**Suggestions / Action plan**")
        for a in ai_output.get("actions", []):
            st.markdown(
                f"**{a.get('title','')}**  \n"
                f"Why: {a.get('why','')}  \n"
                f"How to test: {a.get('how_to_test','')}  \n"
                f"Priority: {a.get('priority','')}"
            )
        st.markdown("**Watchlist**")
        st.write(", ".join(ai_output.get("watchlist", [])))

# Also allow charts to persist after rerun without clicking button again
if "context" in st.session_state and not st.session_state.get("charts_drawn_once"):
    try:
        render_charts(st.session_state["context"])
        st.session_state["charts_drawn_once"] = True
    except Exception:
        pass

# Sidebar: key status
if os.getenv("OPENAI_API_KEY"):
    st.sidebar.success("✅ OpenAI key detected.")
else:
    st.sidebar.warning("ℹ️ No OpenAI key found. You’ll still get the rule-based analysis and charts.")

import os
import requests
import pandas as pd
import streamlit as st
import altair as alt
from pymongo import MongoClient

# =========================
# CONFIG
# =========================
PROM_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB  = os.getenv("MONGO_DB", "game_mlop")

# =========================
# HELPERS
# =========================
def db():
    return MongoClient(MONGO_URI)[MONGO_DB]

def prom_query(query: str):
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10)
    r.raise_for_status()
    return r.json()

def prom_alerts():
    r = requests.get(f"{PROM_URL}/api/v1/alerts", timeout=10)
    r.raise_for_status()
    return r.json()

def pick_scalar(resp):
    try:
        return float(resp["data"]["result"][0]["value"][1])
    except Exception:
        return 0.0
    
def show_df_1based(df, height=180):
    if df.empty:
        st.info("Data kosong.")
        return
    df2 = df.reset_index(drop=True).copy()
    df2.index = df2.index + 1
    st.dataframe(df2, use_container_width=True, height=height)


def clean_df(df, cols):
    """Drop null + cast ke string (AMAN untuk Altair)"""
    if df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df = df[df[c].notna()]
            df[c] = df[c].astype(str)
    return df

def bar(df, x, y, height=240):
    if df.empty:
        st.info("Data kosong.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", sort="-y", title=None),
            y=alt.Y(f"{y}:Q", title=None),
            tooltip=[x, y],
        )
        .properties(height=height)
    )

    st.altair_chart(chart, use_container_width=True)

# =========================
# PAGE
# =========================
st.set_page_config(
    page_title="MLOps Dashboard",
    layout="wide"
)

st.title("YouTube Gaming Trend Analysis")

# =====================================================
# 1) MONITORING (PROMETHEUS) — DI ATAS
# =====================================================
st.subheader("Monitoring Inference (Prometheus)")

c1, c2, c3, c4 = st.columns(4)

rps_ok = prom_query(
    'rate(app_inference_requests_total{endpoint="/predict",status="success"}[1m])'
)
rps_er = prom_query(
    'rate(app_inference_requests_total{endpoint="/predict",status="error"}[1m])'
)
p95 = prom_query(
    'histogram_quantile(0.95, sum(rate(app_inference_latency_seconds_bucket{endpoint="/predict"}[1m])) by (le))'
)
inprog = prom_query("max(app_inference_in_progress)")

c1.metric("RPS success (1m)", f"{pick_scalar(rps_ok):.3f}")
c2.metric("RPS error (1m)", f"{pick_scalar(rps_er):.3f}")
c3.metric("Latency p95 (s)", f"{pick_scalar(p95):.3f}")
c4.metric("In-progress", f"{pick_scalar(inprog):.0f}")

# =========================
# ALERTS (tanpa grafik)
# =========================
st.markdown("### Alerts (FIRING)")

alerts = prom_alerts()
rows = []

for a in alerts.get("data", {}).get("alerts", []):
    if a.get("state") == "firing":
        rows.append({
            "alertname": a.get("labels", {}).get("alertname", ""),
            "severity": a.get("labels", {}).get("severity", ""),
            "summary": a.get("annotations", {}).get("summary", ""),
        })

df_alert = pd.DataFrame(rows)

if df_alert.empty:
    st.success("Tidak ada alert yang FIRING.")
else:
    st.dataframe(df_alert, use_container_width=True, height=160)

st.divider()

# =====================================================
# 2) YOUTUBE ANALYTICS (MONGODB)
# =====================================================
st.subheader("YouTube Gaming Ranking")

topn = st.slider("Top N", 5, 30, 10)

database = db()

# Info scrape terakhir (opsional)
run = database.get_collection("scrape_runs").find_one(sort=[("ts", -1)])
if run:
    st.caption(
        f"Last scrape run: status={run.get('status')} | "
        f"ts={run.get('ts')} | "
        f"unique_videos={run.get('unique_videos')}"
    )

colA, colB, colC = st.columns(3)

# =========================
# Top Channels (Total Views)
# =========================
pipeline_ch = [
    {"$group": {"_id": "$channel_title", "total_views": {"$sum": "$view_count"}}},
    {"$project": {"channel_title": "$_id", "total_views": 1, "_id": 0}},
    {"$sort": {"total_views": -1}},
    {"$limit": int(topn)},
]
df_ch = pd.DataFrame(database["videos_with_genre"].aggregate(pipeline_ch))
df_ch = clean_df(df_ch, ["channel_title"])
df_ch2 = df_ch.copy()
df_ch2.insert(0, "No", range(1, len(df_ch2) + 1))

# =========================
# Genre paling banyak DITONTON
# =========================
pipeline_genre_views = [
    {"$match": {"primary_genre": {"$ne": None}}},
    {"$group": {"_id": "$primary_genre", "total_views": {"$sum": "$view_count"}}},
    {"$project": {"genre": "$_id", "total_views": 1, "_id": 0}},
    {"$sort": {"total_views": -1}},
    {"$limit": int(topn)},
]
df_gv = pd.DataFrame(database["videos_with_genre"].aggregate(pipeline_genre_views))
df_gv = clean_df(df_gv, ["genre"])
df_gv2 = df_gv.copy()
df_gv2.insert(0, "No", range(1, len(df_gv2) + 1))

# =========================
# Top Games (Total Videos)
# =========================
pipeline_game = [
    {"$match": {"games_list": {"$exists": True, "$ne": []}}},
    {"$unwind": "$games_list"},
    {"$group": {"_id": "$games_list", "total_videos": {"$sum": 1}}},
    {"$project": {"game": "$_id", "total_videos": 1, "_id": 0}},
    {"$sort": {"total_videos": -1}},
    {"$limit": int(topn)},
]
df_gm = pd.DataFrame(database["videos_with_genre"].aggregate(pipeline_game))
df_gm = clean_df(df_gm, ["game"])
df_gm2 = df_gm.copy()
df_gm2.insert(0, "No", range(1, len(df_gm2) + 1))



with colA:
    st.markdown("#### Top Channels (Total Views)")
    bar(df_ch, "channel_title", "total_views")
    st.data_editor(
            df_ch2,
            use_container_width=True,
            height=180,
            hide_index=True,
            disabled=True
        )


with colB:
    st.markdown("#### Top Genres by Views")
    bar(df_gv, "genre", "total_views")
    st.data_editor(
            df_gv2,
            use_container_width=True,
            height=180,
            hide_index=True,
            disabled=True
        )

with colC:
    st.markdown("#### Top Games (Total Videos)")
    if df_gm.empty:
        st.warning("Belum ada games_list.")
    else:
        bar(df_gm, "game", "total_videos")
        st.data_editor(
            df_gm2,
            use_container_width=True,
            height=180,
            hide_index=True,
            disabled=True
        )

st.divider()

# =====================================================
# 3) LIVE PREDICTIONS (PROMETHEUS)
# =====================================================
st.subheader("Live Predictions (Inference Output)")

preds = prom_query("sum(app_inference_prediction_per_genre_total) by (genre)")
rows = []

for it in preds.get("data", {}).get("result", []):
    rows.append({
        "genre": it["metric"].get("genre", ""),
        "total": float(it["value"][1])
    })

df_pred = pd.DataFrame(rows).sort_values("total", ascending=False)
df_pred = clean_df(df_pred, ["genre"])

bar(df_pred, "genre", "total", height=220)

st.caption(
    "Distribusi genre hasil prediksi dari request yang masuk. "
    "Digunakan sebagai indikasi minat (bukan growth historis)."
)

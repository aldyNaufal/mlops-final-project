import os, requests
import pandas as pd
import streamlit as st
from pymongo import MongoClient

PROM_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB  = os.getenv("MONGO_DB", "game_mlop")

def db():
    return MongoClient(MONGO_URI)[MONGO_DB]

def prom_query(query: str):
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10)
    r.raise_for_status()
    return r.json()

st.set_page_config(page_title="MLOps + YouTube Dashboard", layout="wide")
st.title("Dashboard: Monitoring Inference + YouTube Analytics")

tab1, tab2 = st.tabs(["Monitoring (Prometheus)", "YouTube Analytics (MongoDB)"])

with tab1:
    st.subheader("Requests / Latency / Errors")

    col1, col2, col3 = st.columns(3)
    rps_ok = prom_query('rate(app_inference_requests_total{endpoint="/predict",status="success"}[1m])')
    rps_er = prom_query('rate(app_inference_requests_total{endpoint="/predict",status="error"}[1m])')
    p95    = prom_query('histogram_quantile(0.95, sum(rate(app_inference_latency_seconds_bucket{endpoint="/predict"}[1m])) by (le))')

    def pick_scalar(resp):
        try:
            return float(resp["data"]["result"][0]["value"][1])
        except:
            return 0.0

    col1.metric("RPS success (1m)", f"{pick_scalar(rps_ok):.3f}")
    col2.metric("RPS error (1m)", f"{pick_scalar(rps_er):.3f}")
    col3.metric("Latency p95 (s)", f"{pick_scalar(p95):.3f}")

    err_by_type = prom_query("sum(rate(app_inference_error_total[5m])) by (type)")
    rows = []
    for it in err_by_type.get("data", {}).get("result", []):
        rows.append({"type": it["metric"].get("type","?"), "rate_5m": float(it["value"][1])})
    df_err = pd.DataFrame(rows).sort_values("rate_5m", ascending=False)
    st.write("Error by type (rate / 5m)")
    st.dataframe(df_err, use_container_width=True)

    preds = prom_query("sum(app_inference_prediction_per_genre_total) by (genre)")
    rows = []
    for it in preds.get("data", {}).get("result", []):
        rows.append({"genre": it["metric"].get("genre","?"), "total": float(it["value"][1])})
    df_pred = pd.DataFrame(rows).sort_values("total", ascending=False)
    st.write("Predictions per genre (total)")
    st.bar_chart(df_pred.set_index("genre")["total"])

with tab2:
    st.subheader("Top YouTube Gaming (90 days from scraper)")

    topn = st.slider("Top N", 5, 50, 15)

    database = db()

    # Top Channel by total views
    pipeline = [
        {"$group": {"_id": "$channel_title",
                    "total_views": {"$sum": "$view_count"},
                    "video_ids": {"$addToSet": "$video_id"}}},
        {"$addFields": {"total_videos": {"$size": "$video_ids"}}},
        {"$project": {"channel_title": "$_id", "total_views": 1, "total_videos": 1, "_id": 0}},
        {"$sort": {"total_views": -1}},
        {"$limit": int(topn)},
    ]
    ch = list(database["videos_with_genre"].aggregate(pipeline))
    df_ch = pd.DataFrame(ch)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Top Channels by Total Views")
        if not df_ch.empty:
            st.bar_chart(df_ch.set_index("channel_title")["total_views"])
        st.dataframe(df_ch, use_container_width=True)

    # Top Game by total videos (butuh games_list dari scraper)
    pipeline_game = [
        {"$unwind": "$games_list"},
        {"$group": {"_id": "$games_list",
                    "total_views": {"$sum": "$view_count"},
                    "video_ids": {"$addToSet": "$video_id"}}},
        {"$addFields": {"total_videos": {"$size": "$video_ids"}}},
        {"$project": {"game": "$_id", "total_views": 1, "total_videos": 1, "_id": 0}},
        {"$sort": {"total_videos": -1}},
        {"$limit": int(topn)},
    ]
    gm = list(database["videos_with_genre"].aggregate(pipeline_game))
    df_gm = pd.DataFrame(gm)

    with c2:
        st.write("Top Games by Total Videos")
        if df_gm.empty:
            st.warning("Belum ada games_list. Pastikan scraper sudah menambah kolom games_list.")
        else:
            st.bar_chart(df_gm.set_index("game")["total_videos"])
            st.dataframe(df_gm, use_container_width=True)

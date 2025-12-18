import os
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne

def get_db():
    uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    db_name = os.getenv("MONGO_DB", "game_mlop")
    client = MongoClient(uri)
    return client[db_name]

def ensure_indexes(db):
    db.videos_raw.create_index("video_id", unique=True)
    db.videos_with_genre.create_index("video_id", unique=True)

def upsert_by_video_id(db, collection: str, df):
    col = db[collection]
    ops = []
    for r in df.to_dict(orient="records"):
        vid = r.get("video_id")
        if not vid:
            continue
        r["updated_at"] = datetime.now(timezone.utc)
        ops.append(UpdateOne({"video_id": vid}, {"$set": r}, upsert=True))
    if ops:
        col.bulk_write(ops, ordered=False)

def replace_collection(db, collection: str, df):
    col = db[collection]
    col.delete_many({})
    if len(df) > 0:
        rows = df.to_dict(orient="records")
        ts = datetime.now(timezone.utc)
        for r in rows:
            r["updated_at"] = ts
        col.insert_many(rows)

def log_run(db, status: str, meta: dict):
    db.pipeline_runs.insert_one({
        "ts": datetime.now(timezone.utc),
        "status": status,
        **meta
    })

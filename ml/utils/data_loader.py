import os
import pandas as pd
from pymongo import MongoClient

def load_xy_from_mongo():
    uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    dbn = os.getenv("MONGO_DB", "game_mlop")
    db = MongoClient(uri)[dbn]

    cur = db["videos_with_genre"].find(
        {"primary_genre": {"$ne": None}},
        {"title": 1, "description": 1, "tags": 1, "primary_genre": 1, "_id": 0}
    )

    rows = list(cur)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Mongo videos_with_genre kosong. Jalankan scraper dulu.")

    def combine(row):
        parts = []
        for c in ("title","description","tags"):
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        return " ".join(parts)

    df["text"] = df.apply(combine, axis=1)
    X = df["text"]
    y = df["primary_genre"]
    return X, y

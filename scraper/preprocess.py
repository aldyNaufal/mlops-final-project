import pandas as pd
from genre_rules import GENRE_KEYWORDS, GAME_KEYWORDS

def _combine_text(title, description, tags):
    parts = []
    for v in (title, description, tags):
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " ".join(parts).lower()

def detect_genres_from_text(title, description, tags):
    text = _combine_text(title, description, tags)
    if not text:
        return []
    found = []
    for genre, kws in GENRE_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                found.append(genre)
                break
    return found

def detect_games_from_text(title, description, tags):
    text = _combine_text(title, description, tags)
    if not text:
        return []
    found = []
    for game, kws in GAME_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                found.append(game)
                break
    return found

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    genres_all, primary_genre = [], []
    games_all, primary_game = [], []

    for _, row in df.iterrows():
        g_list = detect_genres_from_text(row.get("title",""), row.get("description",""), row.get("tags",""))
        genres_all.append(g_list)
        primary_genre.append(g_list[0] if g_list else None)

        game_list = detect_games_from_text(row.get("title",""), row.get("description",""), row.get("tags",""))
        games_all.append(game_list)
        primary_game.append(game_list[0] if game_list else None)

    out = df.copy()
    out["genres_list"] = genres_all
    out["primary_genre"] = primary_genre
    out["games_list"] = games_all
    out["primary_game"] = primary_game
    return out

def explode_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for g in (r.get("genres_list") or []):
            rows.append({
                "video_id": r["video_id"],
                "genre": g,
                "title": r.get("title"),
                "channel_title": r.get("channel_title"),
                "published_at": r.get("published_at"),
                "view_count": r.get("view_count", 0),
                "like_count": r.get("like_count", 0),
                "comment_count": r.get("comment_count", 0),
            })
    return pd.DataFrame(rows)

def aggregate_genre_ranking(df_genre: pd.DataFrame) -> pd.DataFrame:
    if df_genre.empty:
        return pd.DataFrame(columns=["genre","total_views","total_videos","avg_views"])
    return (df_genre.groupby("genre")
            .agg(total_views=("view_count","sum"),
                 total_videos=("video_id","nunique"),
                 avg_views=("view_count","mean"))
            .reset_index()
            .sort_values("total_views", ascending=False))

def aggregate_channel_ranking(df_videos: pd.DataFrame) -> pd.DataFrame:
    if df_videos.empty:
        return pd.DataFrame(columns=["channel_title","total_views","total_videos","avg_views"])
    return (df_videos.groupby("channel_title")
            .agg(total_views=("view_count","sum"),
                 total_videos=("video_id","nunique"),
                 avg_views=("view_count","mean"))
            .reset_index()
            .sort_values("total_views", ascending=False))

def aggregate_game_ranking(df_videos: pd.DataFrame) -> pd.DataFrame:
    if df_videos.empty:
        return pd.DataFrame(columns=["game","total_views","total_videos","avg_views"])
    # explode games_list
    temp = df_videos[["video_id","games_list","view_count"]].explode("games_list").dropna()
    temp = temp.rename(columns={"games_list":"game"})
    return (temp.groupby("game")
            .agg(total_views=("view_count","sum"),
                 total_videos=("video_id","nunique"),
                 avg_views=("view_count","mean"))
            .reset_index()
            .sort_values("total_videos", ascending=False))

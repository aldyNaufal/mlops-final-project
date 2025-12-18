import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from genre_rules import GENRE_KEYWORDS
from youtube_client import youtube_search, youtube_get_videos_stats
from preprocess import (
    add_labels, explode_by_genre,
    aggregate_genre_ranking, aggregate_channel_ranking, aggregate_game_ranking
)
from mongo_store import get_db, ensure_indexes, upsert_by_video_id, replace_collection, log_run

MIN_VIDS_PER_GENRE = 25

def load_api_key():
    load_dotenv()
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise ValueError("YOUTUBE_API_KEY belum di-set di .env")
    return key

def run_pipeline():
    api_key = load_api_key()
    db = get_db()
    ensure_indexes(db)

    now = datetime.now(timezone.utc)
    published_after = (now - timedelta(days=90)).isoformat()
    published_before = now.isoformat()

    genre_to_ids = {g: set() for g in GENRE_KEYWORDS.keys()}
    all_ids = set()

    for genre, keywords in GENRE_KEYWORDS.items():
        for kw in keywords:
            if len(genre_to_ids[genre]) >= MIN_VIDS_PER_GENRE:
                break
            needed = MIN_VIDS_PER_GENRE - len(genre_to_ids[genre])
            max_results = max(needed * 2, 10)
            vids = youtube_search(api_key, kw, max_results=max_results,
                                 published_after=published_after,
                                 published_before=published_before)
            for vid in vids:
                genre_to_ids[genre].add(vid)
                all_ids.add(vid)

    if not all_ids:
        log_run(db, "empty", {"published_after": published_after, "published_before": published_before})
        return

    df_videos = youtube_get_videos_stats(api_key, list(all_ids))

    # RAW + labels
    df_with = add_labels(df_videos)

    # exploded & ranking
    df_exploded = explode_by_genre(df_with)
    df_genre_rank = aggregate_genre_ranking(df_exploded)
    df_channel_rank = aggregate_channel_ranking(df_with)
    df_game_rank = aggregate_game_ranking(df_with)

    # save to mongo
    upsert_by_video_id(db, "videos_raw", df_videos)
    upsert_by_video_id(db, "videos_with_genre", df_with)
    replace_collection(db, "videos_genre_exploded", df_exploded)
    replace_collection(db, "genre_view_ranking", df_genre_rank)
    replace_collection(db, "channel_view_ranking", df_channel_rank)
    replace_collection(db, "game_video_ranking", df_game_rank)

    log_run(db, "success", {
        "unique_videos": int(len(all_ids)),
        "raw_rows": int(df_videos.shape[0]),
        "published_after": published_after,
        "published_before": published_before,
    })

def main():
    run_pipeline()

if __name__ == "__main__":
    main()

import time
import requests
import pandas as pd

BASE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
BASE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

def youtube_search(api_key: str, query: str, max_results=50, published_after=None, published_before=None):
    video_ids, next_page_token, fetched = [], None, 0
    while fetched < max_results:
        to_fetch = min(50, max_results - fetched)
        params = {
            "key": api_key,
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": to_fetch,
            "order": "viewCount",
            "regionCode": "ID",
        }
        if published_after:
            params["publishedAfter"] = published_after
        if published_before:
            params["publishedBefore"] = published_before
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(BASE_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        for item in items:
            video_ids.append(item["id"]["videoId"])

        fetched += len(items)
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.2)

    return list(dict.fromkeys(video_ids))

def youtube_get_videos_stats(api_key: str, video_ids):
    records = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        params = {"key": api_key, "part": "snippet,statistics,contentDetails", "id": ",".join(chunk)}
        resp = requests.get(BASE_VIDEOS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            tags = snippet.get("tags", [])
            tags_joined = "|".join(tags) if isinstance(tags, list) else ""

            records.append({
                "video_id": item["id"],
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "tags": tags_joined,
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": content.get("duration"),
            })
        time.sleep(0.2)

    return pd.DataFrame(records)

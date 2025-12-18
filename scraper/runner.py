import time
from datetime import datetime, timezone

from pipeline import main as pipeline_main

INTERVAL_SECONDS = 24 * 60 * 60

while True:
    print(f"[SCHED] start: {datetime.now(timezone.utc).isoformat()}")
    try:
        pipeline_main()
        print("[SCHED] done OK")
    except Exception as e:
        print("[SCHED] ERROR:", repr(e))
    print(f"[SCHED] sleep {INTERVAL_SECONDS}s")
    time.sleep(INTERVAL_SECONDS)

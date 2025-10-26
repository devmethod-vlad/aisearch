import os
import sys
import psutil
import redis
from app.settings.config import settings
from app.common.storages.sync_redis import SyncRedisStorage


def healthcheck():
    w_id = os.getenv("WORKER_ID")
    r = SyncRedisStorage(
        client=redis.from_url(settings.redis.dsn, decode_responses=True)
    )

    keys = []
    for key in r.scan_iter(match=f"aisearch:health:{w_id}:celery-proc:*"):
        if isinstance(key, bytes):
            keys.append(key.decode())
        else:
            keys.append(str(key))

    processes: list[tuple[int, float]] = []
    for key in keys:
        all_healthy = bool(int(r.client.hget(key, "all_healthy")))
        if not all_healthy:
            sys.exit(1)

        pid = int(r.client.hget(key, "pid"))
        ptime = float(r.client.hget(key, "proc_created_at")) % 1

        processes.append((pid, ptime))

    running_processes: set[tuple[int, float]] = set()
    for proc in psutil.process_iter(["pid"]):
        try:
            pid = proc.info["pid"]
            ptime = proc.create_time() % 1
            running_processes.add((pid, ptime)) 
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for pid, ptime in processes:
        if (pid, ptime) not in running_processes:
            sys.exit(1)


if __name__ == "__main__":
    healthcheck()

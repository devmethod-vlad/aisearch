import os
import socket
import typing as tp
from multiprocessing import current_process

from app.common.storages.sync_redis import SyncRedisStorage


def get_current_btime() -> int:
    """Получает время загрузки системы (btime) из /proc/stat."""
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("btime"):
                return int(line.split()[1])
    raise ValueError("btime not found in /proc/stat")


def get_process_absolute_starttime(pid: int) -> float:
    """Получает абсолютное время старта процесса (секунды с эпохи)."""
    with open(f"/proc/{pid}/stat") as f:
        stat = f.read()
        fields = stat.split()
        if len(fields) < 22:
            raise IndexError(f"Invalid /proc/{pid}/stat format")
        starttime_ticks = int(fields[21])

    hz = os.sysconf("SC_CLK_TCK")
    btime = get_current_btime()

    return btime + (starttime_ticks / hz)


def get_process_relative_starttime(pid: int) -> float:
    """Получает относительное время старта процесса в секундах от загрузки системы.
    Это значение не меняется для одного и того же процесса
    """
    with open(f"/proc/{pid}/stat") as f:
        stat = f.read()
        fields = stat.split()
        if len(fields) < 22:
            raise IndexError(
                f"Invalid /proc/{pid}/stat format: expected at least 22 fields, got {len(fields)}"
            )
        starttime_ticks = int(fields[21])

    hz = os.sysconf("SC_CLK_TCK")
    seconds_from_boot = starttime_ticks / hz

    return seconds_from_boot


def get_process_info(key: str) -> tuple[str, str, float]:
    """Получение информации о процессе с относительным временем создания"""
    container = socket.gethostname()
    proc_name = current_process().name
    pid = os.getpid()
    proc_create_time = get_process_relative_starttime(pid)

    return (
        f"aisearch:health:{key}:{container}:{proc_name}",
        pid,
        proc_create_time,
    )


def update_process_info(
    key: str, info: dict[str, tp.Any], sync_redis_storage: SyncRedisStorage
) -> None:
    """Обновление информации о процессе"""
    pkey, pid, ptime = get_process_info(key=key)

    sync_redis_storage.client.hset(
        pkey,
        mapping={
            "pid": pid,
            "proc_created_at": ptime,
        }
        | info,
    )


def get_worker_process_keys(
    redis_client: SyncRedisStorage, worker_id: str
) -> list[str]:
    """Получает все ключи процессов для указанного воркера из Redis."""
    process_keys = []

    for key in redis_client.scan_iter(
        match=f"aisearch:health:{worker_id}:celery-proc:*"
    ):
        if isinstance(key, bytes):
            process_keys.append(key.decode())
        else:
            process_keys.append(str(key))

    return process_keys

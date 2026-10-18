"""Fetch public pi05 JAX weights with explicit anonymous HTTP range requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import os
from pathlib import Path
import time
from urllib.parse import quote

import gcsfs
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    source = "openpi-assets/checkpoints/pi05_base/params"
    files = gcsfs.GCSFileSystem(token="anon").find(source, detail=True)
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    fds = {}
    block = 16 * 1024 * 1024
    for name, info in files.items():
        path = args.output / name[len(source) + 1 :]
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, info["size"])
        fds[name] = fd
        jobs.extend((name, start, min(start + block, info["size"]) - 1) for start in range(0, info["size"], block))

    def fetch(job):
        name, start, end = job
        url = "https://storage.googleapis.com/" + quote(name, safe="/")
        for attempt in range(4):
            try:
                response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 90))
                response.raise_for_status()
                data = response.content
                if len(data) != end - start + 1:
                    raise ValueError("Incorrect range response size")
                os.pwrite(fds[name], data, start)
                return len(data)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

        raise RuntimeError("Range download retries exhausted")

    total = 0
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, job in enumerate(as_completed([pool.submit(fetch, j) for j in jobs])):
                total += job.result()
                if i % 16 == 0:
                    print(f"{total / 1e9:.2f}GB elapsed={time.monotonic() - started:.1f}s", flush=True)
        (args.output / "DOWNLOAD_COMPLETE").write_text(str(total))
    finally:
        for fd in fds.values():
            os.close(fd)
    print("download_seconds", time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()

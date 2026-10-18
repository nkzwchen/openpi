"""Measure synthetic-only endpoint through the actual local/cloud network path."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy

from openpi.so101.contract import observation_to_input
from openpi.so101.runtime import AsyncChunkPolicy
from openpi.so101.runtime import BoundedWebsocketClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="ws://127.0.0.1:19090")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()
    client = BoundedWebsocketClient(args.uri, timeout_s=30)
    metadata = client.get_server_metadata()
    assert metadata.get("synthetic") is True
    assert metadata.get("benchmark") is True
    preparer = AsyncChunkPolicy(client)
    results = []
    rng = np.random.default_rng(0)
    try:
        for height, width in ((224, 224), (480, 640)):
            source = {
                "state": np.zeros(6, np.float32),
                "cam_main": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
                "cam_wrist": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            }
            samples = []
            for i in range(args.iterations + 3):
                start = time.monotonic()
                obs = preparer.prepare_inputs(source) if height == 224 else observation_to_input(source)
                response = client.infer(obs)
                elapsed = (time.monotonic() - start) * 1000
                if response["actions"].shape != (10, 6) or not np.isfinite(response["actions"]).all():
                    raise ValueError("Invalid benchmark response")
                if i >= 3:
                    samples.append(elapsed)
            results.append(
                {
                    "image_size": [height, width],
                    "payload_bytes": len(msgpack_numpy.packb(obs)),
                    "median_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "p99_ms": float(np.percentile(samples, 99)),
                    "mean_ms": float(np.mean(samples)),
                }
            )
            print(results[-1], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "network": "actual local -> SSH forwarded cloud websocket -> local",
                    "synthetic": True,
                    "source_image_size": [480, 640],
                    "includes_client_prepare_inputs": True,
                    "iterations_per_case": args.iterations,
                    "metadata": metadata,
                    "results": results,
                },
                indent=2,
            )
        )
    finally:
        preparer.close()


if __name__ == "__main__":
    main()

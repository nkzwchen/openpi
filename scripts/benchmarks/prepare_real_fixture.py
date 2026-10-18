"""Extract paired real images and states from a local LeRobot v3 cache for FP8 benchmarks.

Reads source data only. FFmpeg/libdav1d decodes AV1 without OpenCV's AV1 path.
Calibration/evaluation are split by episode; state quantiles use calibration episodes only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from openpi_client.image_tools import resize_with_pad
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--sample-hz", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.samples_per_split < 2:
        parser.error("Output must be new and each split must contain at least two samples")
    root = args.dataset.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if not info["codebase_version"].startswith("v3"):
        parser.error("Expected LeRobot v3")
    fps = int(info["fps"])
    if args.sample_hz < 1 or fps % args.sample_hz:
        parser.error("Source FPS must be divisible by sample Hz")
    stride = fps // args.sample_hz
    cameras = ["observation.images.cam_main", "observation.images.cam_wrist"]
    if info["features"]["observation.state"]["shape"] != [6]:
        parser.error("Expected six-dimensional measured state")
    data_files = sorted(root.glob("data/*/*.parquet"))
    table = pa.concat_tables(
        [pq.read_table(p, columns=["episode_index", "frame_index", "observation.state"]) for p in data_files]
    )
    episode_index = np.asarray(table["episode_index"])
    frame_index = np.asarray(table["frame_index"])
    states = np.asarray(table["observation.state"].to_pylist(), np.float32)
    if not np.isfinite(states).all():
        raise ValueError("Non-finite measured states")
    metadata = pa.concat_tables([pq.read_table(p) for p in sorted(root.glob("meta/episodes/*/*.parquet"))])
    episode_info = {int(r["episode_index"]): r for r in metadata.to_pylist()}
    rng = np.random.default_rng(args.seed)
    episodes = rng.permutation(np.unique(episode_index))
    if len(episodes) < 4:
        raise ValueError("At least four episodes are required for disjoint splits")
    cal_eps, eval_eps = np.split(episodes, [len(episodes) // 2])
    cal_rows = np.isin(episode_index, cal_eps)
    q01, q99 = np.quantile(states[cal_rows], [0.01, 0.99], axis=0).astype(np.float32)
    selected = {}
    for name, group in [("calibration", cal_eps), ("evaluation", eval_eps)]:
        choices = []
        for i in range(args.samples_per_split):
            ep = int(group[i % len(group)])
            rows = np.flatnonzero((episode_index == ep) & (frame_index % stride == 0))
            rows = rows[~np.isin(rows, choices)]
            if not len(rows):
                raise ValueError(f"Episode {ep} has too few unique eligible frames")
            choices.append(int(rng.choice(rows)))
        if len(set(choices)) != len(choices):
            raise ValueError("Repeated sampled frames; use fewer samples or a different seed")
        selected[name] = choices
    requests = {}
    provenance = {}
    for name, rows in selected.items():
        provenance[name] = []
        for sample_i, row in enumerate(rows):
            ep, frame = int(episode_index[row]), int(frame_index[row])
            meta = episode_info[ep]
            item = {"episode_index": ep, "frame_index": frame, "dataset_row": row, "time_s": frame / fps, "videos": {}}
            for camera in cameras:
                rel = info["video_path"].format(
                    video_key=camera,
                    chunk_index=int(meta[f"videos/{camera}/chunk_index"]),
                    file_index=int(meta[f"videos/{camera}/file_index"]),
                )
                timestamp = float(meta[f"videos/{camera}/from_timestamp"]) + frame / fps
                video_frame = round(timestamp * fps)
                requests.setdefault(rel, {}).setdefault(video_frame, []).append((name, sample_i, cameras.index(camera)))
                item["videos"][camera] = {"path": rel, "video_frame": video_frame, "timestamp_s": timestamp}
            provenance[name].append(item)
    images = {name: np.empty((len(rows), 2, 224, 224, 3), np.uint8) for name, rows in selected.items()}

    def decode(entry):
        rel, mapping = entry
        camera = next(c for c in cameras if f"/{c}/" in "/" + rel)
        height, width, _ = info["features"][camera]["shape"]
        indices = sorted(mapping)
        selection = "+".join(f"eq(n,{i})" for i in indices)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-c:v",
            "libdav1d",
            "-threads",
            "2",
            "-i",
            str(root / rel),
            "-vf",
            f"select={selection.replace(',', chr(92) + ',')}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        expected = len(indices) * height * width * 3
        if len(result.stdout) != expected:
            raise ValueError(f"Decoded byte count mismatch for {rel}: {len(result.stdout)} != {expected}")
        frames = np.frombuffer(result.stdout, np.uint8).reshape(len(indices), height, width, 3)
        resized = resize_with_pad(frames, 224, 224)
        for position, index in enumerate(indices):
            for name, sample_i, view in mapping[index]:
                images[name][sample_i, view] = resized[position]
        return {
            "path": rel,
            "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(),
            "decoded_selected_frames": len(indices),
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        video_sources = list(pool.map(decode, requests.items()))
    args.output.mkdir(parents=True)
    payload = {}
    for name, rows in selected.items():
        payload[name + "_images"] = images[name]
        payload[name + "_raw_states"] = states[rows]
        payload[name + "_states"] = (states[rows] - q01) / (q99 - q01 + 1e-6) * 2 - 1
    np.savez_compressed(args.output / "samples.npz", **payload)
    manifest = {
        "source_dataset": str(root),
        "source_fps": fps,
        "sample_hz": args.sample_hz,
        "total_episodes": len(episodes),
        "total_frames": len(states),
        "seed": args.seed,
        "calibration_episodes": cal_eps.tolist(),
        "evaluation_episodes": eval_eps.tolist(),
        "samples": provenance,
        "video_sources": video_sources,
        "source_tasks": pq.read_table(root / "meta/tasks.parquet").to_pylist(),
        "prompt_used": "pick blocks from table",
        "state_order": info["features"]["observation.state"]["names"],
        "state_normalization": {
            "method": "quantiles from calibration episodes only, no clipping",
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "epsilon": 1e-6,
            "scope": "surrogate benchmark statistics; not a trained SO101 checkpoint asset",
        },
        "image_preprocessing": "openpi_client.image_tools.resize_with_pad PIL bilinear to 224; full field of view, black letterbox",
        "samples_sha256": hashlib.sha256((args.output / "samples.npz").read_bytes()).hexdigest(),
        "scope": "Real observation calibration and BF16/FP8 numerical comparison using base pi05 weights. No training labels or task success assumptions.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "source_dataset",
                    "source_fps",
                    "sample_hz",
                    "total_episodes",
                    "total_frames",
                    "samples_sha256",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

# ruff: noqa: PLC0415
"""Reviewed expert intervals -> immutable local LeRobot v3 subset (two full-frame videos).

Only load trusted recorder pickle files. Interval endpoints are indices into the
original transitions list, half-open [start, end), never compressed frame indices.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from openpi.so101.contract import CAMERAS
from openpi.so101.contract import FPS
from openpi.so101.contract import PROMPT
from openpi.so101.contract import STATE_ORDER
from openpi.so101.contract import UNITS
from openpi.so101.contract import VERSION
from openpi.so101.contract import observation_to_input
from openpi.so101.contract import validate_action


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recording(path):
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("transitions"), list):
        raise ValueError("Expected current recorder {transitions, meta} format")
    return payload


def inspect_recording(path):
    payload = load_recording(path)
    steps, meta = payload["transitions"], payload["meta"]
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "steps": len(steps),
        "human_steps": sum(t.get("infos", {}).get("intervening") is True for t in steps),
        "mode": "mixed" if meta.get("policy") else "manual",
        "metadata": {
            k: meta.get(k)
            for k in (
                "env_hz",
                "action_units",
                "action_semantics",
                "observation_semantics",
                "operator_excluded",
                "quality_failed",
                "recording",
            )
        },
        "observation_shapes": {k: list(np.shape(v)) for k, v in steps[0]["observations"].items()} if steps else {},
    }


def review_template(path):
    payload = load_recording(path)
    return {
        "version": VERSION,
        "recording_sha256": sha256(path),
        "reviewer": "",
        "reviewed": False,
        "status": "pending",
        "intervals": [],
        "notes": "Use half-open original transition indices; exclude reset and post-task return motion.",
        "n_transitions": len(payload["transitions"]),
    }


def metadata_error(meta):
    if meta.get("operator_excluded") is not False:
        return "operator_excluded_or_missing"
    if meta.get("env_hz") != FPS:
        return "wrong_fps"
    if meta.get("action_units") != UNITS:
        return "wrong_action_units"
    if meta.get("action_semantics") != "absolute_joint_position_6d":
        return "wrong_action_semantics"
    if meta.get("observation_semantics") != "measured_joint_position_6d":
        return "wrong_state_semantics"
    if meta.get("contract") != VERSION or meta.get("task") != PROMPT or not meta.get("camera_spec"):
        return "wrong_contract_task_or_camera_spec"
    if meta.get("roi") is not False or meta.get("frozen_joints") != [] or not meta.get("calibration"):
        return "missing_full_frame_or_calibration_contract"
    return None


def step_error(t):
    info = t.get("infos", {})
    if info.get("intervening") is not True:
        return "not_expert"
    if info.get("phase") != "record":
        return "not_record_phase"
    if info.get("quality_ok") is not True or info.get("quality_problems"):
        return "bad_quality"
    control = info.get("joint_control") or {}
    if control.get("accepted") is not True or control.get("limited") is not False:
        return "rejected_or_limited_command"
    try:
        action = validate_action(t["actions"])
        if not np.allclose(action, control["requested_action"], rtol=0, atol=1e-6):
            return "request_label_mismatch"
        if not np.allclose(action, control["sent_action"], rtol=0, atol=1e-6):
            return "request_execution_mismatch"
        observation_to_input(t["observations"])
        if not np.isfinite(info["recording_enqueued_monotonic_s"]):
            return "bad_timestamp"
        int(info["recording_step"])
    except (KeyError, ValueError, TypeError):
        return "invalid_shape_range_or_audit"
    return None


def select_segments(payload, review, source_hash, horizon):
    steps, meta = payload["transitions"], payload["meta"]
    reasons = Counter()
    error = metadata_error(meta)
    if error:
        return [], Counter({error: len(steps)})
    if (
        review.get("version") != VERSION
        or review.get("recording_sha256") != source_hash
        or review.get("reviewed") is not True
        or review.get("status") != "approved"
        or not str(review.get("reviewer", "")).strip()
    ):
        return [], Counter({"unreviewed_or_hash_mismatch": len(steps)})
    intervals = review.get("intervals", [])
    selected = set()
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(steps):
            raise ValueError("Invalid expert interval")
        indices = set(range(start, end))
        if selected & indices:
            raise ValueError("Overlapping expert intervals")
        selected.update(indices)
    segments, current = [], []

    def finish():
        if current:
            if len(current) >= horizon:
                segments.append(current.copy())
                reasons["tail_without_full_horizon"] += horizon - 1
            else:
                reasons["short_interval_frames"] += len(current)
            current.clear()

    for index, step in enumerate(steps):
        error = "outside_reviewed_interval" if index not in selected else step_error(step)
        if error:
            finish()
            reasons[error] += 1
            continue
        info = step["infos"]
        if current:
            previous = steps[current[-1]]
            prev_info = previous["infos"]
            gap = float(info["recording_enqueued_monotonic_s"]) - float(prev_info["recording_enqueued_monotonic_s"])
            if (
                info.get("segment_start")
                or previous.get("dones")
                or int(info["recording_step"]) != int(prev_info["recording_step"]) + 1
                or not 0 < gap <= 0.25
            ):
                finish()
                reasons["sequence_boundary"] += 1
        current.append(index)
        # Adjacent independently reviewed intervals are distinct sequence boundaries.
        if any(index + 1 == interval["end"] for interval in intervals):
            finish()
    finish()
    return segments, reasons


def export_dataset(recordings, output, *, horizon=10, prompt=PROMPT, validation_fraction=0.2):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if horizon < 1 or not 0 < validation_fraction < 1 or not prompt.strip():
        raise ValueError("Require horizon>=1, nonempty prompt and 0<validation_fraction<1")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    if prompt != PROMPT:
        raise ValueError("Task must match the recorder contract; do not relabel existing trajectories")
    report = {"contract": VERSION, "horizon": horizon, "prompt": prompt, "sources": [], "synthetic": False}
    split_rows = {"train": [], "validation": []}
    split_eps = {"train": [], "validation": []}
    provenance = []
    seen = set()
    contract_fingerprint = None
    camera_shapes = {}
    writers = {}
    try:
        for recording in recordings:
            source = Path(recording).resolve()
            digest = sha256(source)
            if digest in seen:
                raise ValueError("Duplicate recording content")
            seen.add(digest)
            payload = load_recording(source)
            review_path = source.with_suffix(".expert.json")
            review = json.loads(review_path.read_text()) if review_path.exists() else {}
            segments, rejected = select_segments(payload, review, digest, horizon)
            meta = payload["meta"]
            # Split all episodes in a recording batch together; never split windows.
            batch_id = (
                sha256(source.parent / "meta.json") if (source.parent / "meta.json").exists() else str(source.parent)
            )
            split = (
                "validation"
                if int(hashlib.sha256(batch_id.encode()).hexdigest()[:8], 16) / 2**32 < validation_fraction
                else "train"
            )
            report["synthetic"] |= bool(meta.get("synthetic", False))
            report["sources"].append(
                {
                    "path": str(source),
                    "sha256": digest,
                    "batch_id": batch_id,
                    "split": split,
                    "review_sha256": sha256(review_path) if review_path.exists() else None,
                    "segments": len(segments),
                    "valid_starts": sum(len(s) - horizon + 1 for s in segments),
                    "reasons": dict(rejected),
                }
            )
            if segments:
                fingerprint = json.dumps(
                    {k: meta[k] for k in ("camera_spec", "calibration", "action_units")}, sort_keys=True
                )
                if contract_fingerprint is not None and fingerprint != contract_fingerprint:
                    raise ValueError("Camera/calibration differs between sources; export separately")
                contract_fingerprint = fingerprint
            for segment in segments:
                import cv2

                root = output / split
                ep_number = len(split_eps[split])
                writers = {}
                for camera in CAMERAS:
                    image = observation_to_input(payload["transitions"][segment[0]]["observations"], prompt)[
                        f"observation/{camera}"
                    ]
                    shape = image.shape[:2]
                    spec = meta["camera_spec"][camera]
                    if shape != (spec["height"], spec["width"]) or any(size % 2 for size in shape):
                        raise ValueError("Video dimensions must match metadata and be even; do not silently crop")
                    if camera in camera_shapes and camera_shapes[camera] != shape:
                        raise ValueError("Camera resolution differs across source episodes")
                    camera_shapes[camera] = shape
                    video_path = root / f"videos/observation.images.{camera}/chunk-000/file-{ep_number:03d}.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (shape[1], shape[0])
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to encode {video_path}")
                    writers[camera] = writer
                rows, episodes = split_rows[split], split_eps[split]
                ep_index, first = len(episodes), len(rows)
                for frame, source_index in enumerate(segment):
                    step = payload["transitions"][source_index]
                    obs = observation_to_input(step["observations"], prompt)
                    row = {
                        "index": len(rows),
                        "episode_index": ep_index,
                        "frame_index": frame,
                        "timestamp": frame / FPS,
                        "task_index": 0,
                        "observation.state": obs["observation/state"].tolist(),
                        "action": validate_action(step["actions"]).tolist(),
                    }
                    for camera in CAMERAS:
                        image = obs[f"observation/{camera}"]
                        if image.shape[:2] != camera_shapes[camera]:
                            raise ValueError("Camera resolution changed inside an episode")
                        writers[camera].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    rows.append(row)
                    provenance.append(
                        {
                            "split": split,
                            "index": row["index"],
                            "episode_index": ep_index,
                            "source_sha256": digest,
                            "source_frame": source_index,
                            "recording_step": step["infos"]["recording_step"],
                            "time_monotonic_s": step["infos"]["recording_enqueued_monotonic_s"],
                            "joint_control": step["infos"]["joint_control"],
                            "policy": meta.get("policy"),
                            "quality_ok": step["infos"]["quality_ok"],
                            "reviewer": review["reviewer"],
                        }
                    )
                for writer in writers.values():
                    writer.release()
                video_indices = {
                    f"videos/observation.images.{c}/{k}": v
                    for c in CAMERAS
                    for k, v in (("chunk_index", 0), ("file_index", ep_number))
                }
                episodes.append(
                    {
                        **video_indices,
                        "episode_index": ep_index,
                        "dataset_from_index": first,
                        "dataset_to_index": len(rows),
                        "length": len(segment),
                        "source_sha256": digest,
                        "source_frames": segment,
                    }
                )
        features = {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        }
        features.update(
            {
                f"observation.images.{c}": {"dtype": "video", "shape": [camera_shapes[c][0], camera_shapes[c][1], 3]}
                for c in camera_shapes
            }
        )
        for split, rows in split_rows.items():
            root = output / split
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            info = {
                "codebase_version": "v3.0",
                "robot_type": "so101",
                "fps": FPS,
                "features": features,
                "total_frames": len(rows),
                "total_episodes": len(split_eps[split]),
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "openpi_contract": VERSION,
                "full_horizon_only": True,
                "export_horizon": horizon,
                "action_units": UNITS,
                "state_order": STATE_ORDER,
                "source_contract": json.loads(contract_fingerprint) if contract_fingerprint else None,
                "prompt": prompt,
                "split": split,
                "synthetic": report["synthetic"],
            }
            (root / "meta/info.json").write_text(json.dumps(info, indent=2))
            pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": prompt}]), root / "meta/tasks.parquet")
            if rows:
                pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
                pq.write_table(
                    pa.Table.from_pylist(split_eps[split]), root / "meta/episodes/chunk-000/file-000.parquet"
                )
        report["splits"] = {
            s: {
                "frames": len(rows),
                "episodes": len(split_eps[s]),
                "valid_starts": sum(e["length"] - horizon + 1 for e in split_eps[s]),
            }
            for s, rows in split_rows.items()
        }
        report["empty_splits"] = [s for s, rows in split_rows.items() if not rows]
        (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance))
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        (output / "COMPLETE").write_text(VERSION + "\n")
    except Exception:
        (output / "INCOMPLETE").write_text("Export failed; do not train on this directory.\n")
        raise
    finally:
        for writer in writers.values():
            writer.release()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "review-template", "export"])
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.command == "export":
        if args.output is None:
            parser.error("export requires --output")
        print(
            json.dumps(
                export_dataset(
                    args.recordings,
                    args.output,
                    horizon=args.horizon,
                    prompt=args.prompt,
                    validation_fraction=args.validation_fraction,
                ),
                indent=2,
            )
        )
    else:
        for source in args.recordings:
            if args.command == "inspect":
                print(json.dumps(inspect_recording(source), ensure_ascii=False))
            else:
                destination = Path(source).with_suffix(".expert.json")
                with destination.open("x") as stream:
                    json.dump(review_template(source), stream, indent=2)
                print(destination)


if __name__ == "__main__":
    main()

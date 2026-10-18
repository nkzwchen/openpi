"""Clearly marked synthetic fixtures; never represent robot demonstrations."""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from openpi.so101.contract import CAMERAS
from openpi.so101.contract import FPS
from openpi.so101.contract import PROMPT
from openpi.so101.contract import UNITS
from openpi.so101.contract import VERSION
from openpi.so101.export import review_template


def payload(length=36, *, mixed=False, seed=0):
    meta = {
        "contract": VERSION,
        "synthetic": True,
        "task": PROMPT,
        "env_hz": FPS,
        "action_units": UNITS,
        "action_semantics": "absolute_joint_position_6d",
        "observation_semantics": "measured_joint_position_6d",
        "operator_excluded": False,
        "quality_failed": False,
        "roi": False,
        "frozen_joints": [],
        "camera_spec": {c: {"width": 64, "height": 48, "full_frame": True} for c in CAMERAS},
        "calibration": {"synthetic": "not a real calibration"},
        "seed": seed,
        "policy": {"id": "synthetic-policy"} if mixed else None,
    }
    transitions = []
    for i in range(length):
        action = np.array([i / 2, -i / 3, i / 4, -i / 5, i / 6, 30 + i / 2], np.float32)
        images = {
            c: np.full((48, 64, 3), (180, 20, 20) if j == 0 else (20, 20, 180), np.uint8) for j, c in enumerate(CAMERAS)
        }
        info = {
            "intervening": not mixed or i < 12 or i >= 18,
            "phase": "record",
            "quality_ok": True,
            "quality_problems": [],
            "segment_start": False,
            "recording_step": i,
            "recording_enqueued_monotonic_s": 100 + i / FPS,
            "joint_control": {
                "accepted": True,
                "limited": False,
                "requested_action": action.tolist(),
                "sent_action": action.tolist(),
            },
        }
        transitions.append(
            {
                "observations": {"state": action - 0.1, **images},
                "actions": action,
                "infos": info,
                "dones": i == length - 1,
            }
        )
        transitions[-1]["observations"]["state"][-1] += 0.1
    return {"meta": meta, "transitions": transitions}


def write_fixtures(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    paths = []
    # Two batches in each split, two manual + two mixed; deterministic grouping.
    for i in range(4):
        wanted = "train" if i < 2 else "validation"
        nonce = 0
        while True:
            batch_meta = json.dumps({"synthetic": True, "batch": i, "nonce": nonce})
            digest = hashlib.sha256(batch_meta.encode()).hexdigest()
            actual = "validation" if int(hashlib.sha256(digest.encode()).hexdigest()[:8], 16) / 2**32 < 0.2 else "train"
            if actual == wanted:
                break
            nonce += 1
        batch = root / f"batch_{i}"
        batch.mkdir()
        (batch / "meta.json").write_text(batch_meta)
        path = batch / "ep_0000.pkl"
        with path.open("wb") as stream:
            pickle.dump(payload(mixed=bool(i % 2), seed=i), stream)
        review = review_template(path)
        review.update(
            reviewed=True,
            status="approved",
            reviewer="synthetic-fixture-generator",
            intervals=[{"start": 0, "end": 36}],
        )
        path.with_suffix(".expert.json").write_text(json.dumps(review, indent=2))
        paths.append(path)
    return paths

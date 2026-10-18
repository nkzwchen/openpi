"""640x480 synthetic full-frame MP4 fixture, explicitly not real demonstrations."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from openpi.shared.normalize import load
from openpi.shared.normalize import save
from openpi.so101.contract import CAMERAS
from openpi.so101.export import export_dataset
from openpi.so101.export import sha256
from openpi.so101.synthetic import write_fixtures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--norm-source", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = write_fixtures(args.output / "raw")
    rng = np.random.default_rng(0)
    for path in paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        for c in CAMERAS:
            payload["meta"]["camera_spec"][c].update(width=640, height=480)
        for i, step in enumerate(payload["transitions"]):
            for c in CAMERAS:
                # Detailed texture plus coherent horizontal motion, compressed to mp4v.
                texture = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
                step["observations"][c] = np.roll(np.repeat(np.repeat(texture, 4, 0), 4, 1), i * 2, axis=1)
        with path.open("wb") as f:
            pickle.dump(payload, f)
        reviewpath = path.with_suffix(".expert.json")
        review = json.loads(reviewpath.read_text())
        review["recording_sha256"] = sha256(path)
        reviewpath.write_text(json.dumps(review))
    export_dataset(paths, args.output / "dataset")
    norm = load(args.norm_source)
    save(args.output / "assets/pi05_so101_absolute6/so101_openpi_absolute_v1", norm)
    print(args.output / "dataset/train")


if __name__ == "__main__":
    main()

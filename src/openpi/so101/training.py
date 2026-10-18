# ruff: noqa: PLC0415
"""Shared validation/binding for real absolute-action training and serving."""

import hashlib
import json
from pathlib import Path

from openpi.so101.contract import PROMPT
from openpi.so101.contract import UNITS
from openpi.so101.contract import VERSION


def dataset_contract(dataset):
    dataset = Path(dataset).resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    if (
        info.get("openpi_contract") != VERSION
        or info.get("synthetic") is not False
        or info.get("prompt") != PROMPT
        or info.get("action_units") != UNITS
        or info.get("split") != "train"
        or not info.get("full_horizon_only")
    ):
        raise ValueError("Require a real, reviewed training split with matching absolute-six task/units")
    if not (dataset.parent / "COMPLETE").exists() or (dataset.parent / "INCOMPLETE").exists():
        raise ValueError("Incomplete export")
    if not info.get("total_episodes") or not info.get("source_contract"):
        raise ValueError("Empty dataset or missing camera/calibration contract")
    report_path = dataset.parent / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("synthetic") is not False or report.get("contract") != VERSION:
        raise ValueError("Invalid or synthetic export report")
    if report.get("empty_splits"):
        raise ValueError("Require independent nonempty train and validation batches")
    return info, hashlib.sha256(report_path.read_bytes()).hexdigest()


def compute_norm_stats(config):
    """Use exactly the valid training starts; never include validation observations."""
    from openpi.shared import normalize
    from openpi.training import data_loader

    dc = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(dc, config.model.action_horizon, config.model, decode_videos=False)
    dataset = data_loader.TransformedDataset(raw, [*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for index in range(len(dataset)):
        row = dataset[index]
        for key, running in stats.items():
            running.update(row[key])
    norm = {key: running.get_statistics() for key, running in stats.items()}
    normalize.save(config.assets_dirs / dc.asset_id, norm)
    return len(dataset)


def frozen_vlm_filter():
    """Freeze SigLIP and PaliGemma, including VLM adapters; retain action expert."""
    import flax.nnx as nnx

    from openpi.shared import nnx_utils

    return nnx.Any(
        nnx_utils.PathRegex(r"PaliGemma/img/.*"),
        nnx.All(nnx_utils.PathRegex(r"PaliGemma/llm/.*"), nnx.Not(nnx_utils.PathRegex(r".*_1.*"))),
    )

"""Serve an absolute-six checkpoint with an explicit dataset contract."""

import argparse
import dataclasses
import json
from pathlib import Path
import subprocess

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.so101.contract import PROMPT
from openpi.so101.contract import UNITS
from openpi.so101.contract import VERSION
from openpi.so101.training import dataset_contract
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Training split used for this checkpoint")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    info, digest = dataset_contract(args.dataset)
    binding = json.loads((args.checkpoint.parent / "so101_contract.json").read_text())
    if binding.get("dataset_report_sha256") != digest or binding.get("dataset_info") != info:
        parser.error("Checkpoint is not bound to this dataset version")
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    cfg = config.get_config("pi05_so101_absolute6")
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(
            cfg.model,
            action_horizon=info["export_horizon"],
            paligemma_variant="gemma_2b_lora" if binding["lora"] else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if binding["lora"] else "gemma_300m",
        ),
        data=dataclasses.replace(
            cfg.data,
            repo_id=str(args.dataset.resolve()),
            assets=dataclasses.replace(cfg.data.assets, asset_id=binding["asset_id"]),
        ),
    )
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_steps})
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    metadata = {
        "contract": VERSION,
        "prompt": PROMPT,
        "action_units": UNITS,
        "synthetic": False,
        "action_horizon": info["export_horizon"],
        "checkpoint": str(args.checkpoint.resolve()),
        "code_commit": commit,
        "source_contract": info["source_contract"],
        "num_steps": args.num_steps,
    }
    WebsocketPolicyServer(policy, host=args.bind, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()

# ruff: noqa: PLC0415
"""Train a real absolute-six dataset; independent from the legacy SO101 config."""

import argparse
import dataclasses
import json
from pathlib import Path
import runpy

from openpi.so101.training import compute_norm_stats
from openpi.so101.training import dataset_contract
from openpi.training import config
from openpi.training.weight_loaders import CheckpointWeightLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Immutable export's train directory")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--weights", default="gs://openpi-assets/checkpoints/pi05_base/params")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--freeze-vlm",
        action="store_true",
        help="Freeze the complete vision-language backbone; use --no-lora to train the full action expert",
    )
    parser.add_argument("--compute-norm-only", action="store_true")
    args = parser.parse_args()
    info, digest = dataset_contract(args.dataset)
    if args.steps < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("steps/batch-size must be positive and num-workers nonnegative")
    cfg = config.get_config("pi05_so101_absolute6")
    model = dataclasses.replace(
        cfg.model,
        action_horizon=info["export_horizon"],
        paligemma_variant="gemma_2b_lora" if args.lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if args.lora else "gemma_300m",
    )
    freeze_filter = model.get_freeze_filter()
    if args.freeze_vlm:
        import flax.nnx as nnx

        from openpi.so101.training import frozen_vlm_filter

        freeze_filter = nnx.Any(freeze_filter, frozen_vlm_filter())
    asset_id = f"so101_absolute6_{digest[:16]}"
    cfg = dataclasses.replace(
        cfg,
        model=model,
        data=dataclasses.replace(
            cfg.data,
            repo_id=str(args.dataset.resolve()),
            assets=dataclasses.replace(cfg.data.assets, asset_id=asset_id),
        ),
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_root,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=False,
        weight_loader=CheckpointWeightLoader(args.weights),
        freeze_filter=freeze_filter,
        ema_decay=None if args.lora or args.freeze_vlm else cfg.ema_decay,
    )
    count = compute_norm_stats(cfg)
    print(f"Normalization computed only from {count} valid training windows; asset_id={asset_id}", flush=True)
    if args.compute_norm_only:
        return
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))
    namespace["main"](cfg)
    binding = {
        "dataset_report_sha256": digest,
        "dataset_info": info,
        "asset_id": asset_id,
        "lora": args.lora,
        "freeze_vlm": args.freeze_vlm,
        "pretrained_weights": args.weights,
    }
    (cfg.checkpoint_dir / "so101_contract.json").write_text(json.dumps(binding, indent=2))


if __name__ == "__main__":
    main()

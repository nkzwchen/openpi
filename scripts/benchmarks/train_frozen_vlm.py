# ruff: noqa: PLC0415
"""Full-size pi05 frozen-VLM throughput. Synthetic data measures speed, not quality."""

import argparse
import dataclasses
import json
from pathlib import Path
import runpy
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.so101.training import frozen_vlm_filter
from openpi.training import config as configs
from openpi.training import sharding
from openpi.training import weight_loaders


class RepeatDataset:
    # Keep a realistic number of batches per epoch, without copying video files.
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * 1000

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]


def stats(times):
    x = np.asarray(times) * 1000
    return {"median_ms": float(np.median(x)), "p95_ms": float(np.percentile(x, 95)), "mean_ms": float(np.mean(x))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--weights", type=str)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    args = p.parse_args()
    cfg = configs.get_config("pi05_so101_absolute6")
    cfg = dataclasses.replace(
        cfg,
        batch_size=args.batch_size,
        freeze_filter=frozen_vlm_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(args.weights)
        if args.weights
        else weight_loaders.NoOpWeightLoader(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    abstract = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    st = nnx.state(abstract)
    n_all = sum(x.size for x in jax.tree.leaves(st))
    n_train = sum(x.size for x in jax.tree.leaves(st.filter(cfg.trainable_filter)))
    print(
        f"Parameters total={n_all:,}, trainable={n_train:,}; VLM is frozen, action expert fully trainable", flush=True
    )
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))
    mesh = sharding.make_mesh(1)
    started = time.monotonic()
    state, _state_sharding = namespace["init_train_state"](cfg, jax.random.key(0), mesh, resume=False)
    jax.block_until_ready(state)
    print(f"Initialized in {time.monotonic() - started:.1f}s", flush=True)
    from openpi.models import model as model_lib
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.so101.contract import PROMPT

    tokens, mask = PaligemmaTokenizer(cfg.model.max_token_len).tokenize(PROMPT, np.zeros(6, np.float32))
    obs = cfg.model.fake_obs(args.batch_size)
    obs = dataclasses.replace(
        obs,
        images={
            k: jax.random.uniform(jax.random.key(i + 10), v.shape, minval=-1, maxval=1)
            for i, (k, v) in enumerate(obs.images.items())
        },
        image_masks={k: jnp.full((args.batch_size,), k != "right_wrist_0_rgb") for k in obs.images},
        tokenized_prompt=jnp.broadcast_to(tokens, (args.batch_size, len(tokens))),
        tokenized_prompt_mask=jnp.broadcast_to(mask, (args.batch_size, len(mask))),
    )
    actions = jnp.zeros((args.batch_size, 10, 32), jnp.float32)
    batch = (obs, actions)
    step = jax.jit(namespace["train_step"], static_argnums=0, donate_argnums=2)
    t = time.monotonic()
    state, info = step(cfg, jax.random.key(7), state, batch)
    jax.block_until_ready(info)
    compile_s = time.monotonic() - t
    print(f"Compile+first step {compile_s:.1f}s; loss={float(info['loss']):.4f}", flush=True)
    times = []
    for i in range(args.steps + 3):
        t = time.monotonic()
        state, info = step(cfg, jax.random.key(i + 20), state, batch)
        jax.block_until_ready(info)
        if i >= 3:
            times.append(time.monotonic() - t)
    result = {
        "batch_size": args.batch_size,
        "horizon": 10,
        "token_max_length": 200,
        "actual_prompt_tokens": int(mask.sum()),
        "real_camera_count": 2,
        "computed_image_slots": 3,
        "image_resolution": [224, 224],
        "synthetic": True,
        "pretrained_weights": args.weights,
        "total_parameters": n_all,
        "trainable_parameters": n_train,
        "frozen_parameters": n_all - n_train,
        "optimizer": "AdamW, no EMA",
        "training": "full action expert + projections; all VLM frozen",
        "compile_first_step_s": compile_s,
        "gpu_compute_step": stats(times),
        "samples_per_second": args.batch_size / np.mean(times),
        "loss": float(info["loss"]),
        "grad_norm": float(info["grad_norm"]),
        "steps_measured": args.steps,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    train_cfg = cfg
    if args.dataset:
        import itertools

        from openpi.training import data_loader

        cfg = dataclasses.replace(
            cfg,
            assets_base_dir=str(args.dataset.parents[1] / "assets"),
            data=dataclasses.replace(cfg.data, repo_id=str(args.dataset)),
        )
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        # The fixture's normalization assets can be provided alongside the dataset.
        ds = data_loader.create_torch_dataset(dc, 10, cfg.model)
        ds = data_loader.transform_dataset(ds, dc)
        idx = itertools.cycle(range(len(ds)))
        loadtimes = []
        totaltimes = []
        for i in range(min(args.steps, 20) + 3):
            t = time.monotonic()
            rows = [ds[next(idx)] for _ in range(args.batch_size)]
            tensors = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *rows)
            b = (model_lib.Observation.from_dict(tensors), tensors["actions"])
            jax.block_until_ready(b)
            loaded = time.monotonic() - t
            state, info = step(train_cfg, jax.random.key(i + 999), state, b)
            jax.block_until_ready(info)
            if i >= 3:
                loadtimes.append(loaded)
                totaltimes.append(time.monotonic() - t)
        result["video_data_prepare"] = stats(loadtimes)
        result["serial_loader_plus_train"] = stats(totaltimes)
        result["serial_loader_samples_per_second"] = args.batch_size / np.mean(totaltimes)
        pipeline_results = []
        for workers in (0, 4):
            loader = data_loader.TorchDataLoader(RepeatDataset(ds), args.batch_size, num_workers=workers, shuffle=True)
            iterator = iter(loader)

            def pull_batch(iterator=iterator):
                tensor = next(iterator)
                return model_lib.Observation.from_dict(tensor), tensor["actions"]

            start_load = time.monotonic()
            b = pull_batch()
            startup = time.monotonic() - start_load
            for i in range(3):
                state, info = step(train_cfg, jax.random.key(i + 1900), state, b)
                jax.block_until_ready(info)
                b = pull_batch()
            start_pipeline = time.monotonic()
            for i in range(args.steps):
                state, info = step(train_cfg, jax.random.key(i + 2000), state, b)
                b = pull_batch()  # Match train.py: CPU loading overlaps asynchronous GPU execution.
            jax.block_until_ready(info)
            elapsed = time.monotonic() - start_pipeline
            pipeline_results.append(
                {
                    "num_workers": workers,
                    "startup_s": startup,
                    "mean_step_ms": elapsed / args.steps * 1000,
                    "samples_per_second": args.steps * args.batch_size / elapsed,
                }
            )
            print("Pipeline", pipeline_results[-1], flush=True)
            del iterator, loader
        result["native_async_pipeline"] = pipeline_results
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

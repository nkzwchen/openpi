"""Full pi05 latency comparison: BF16 vs native-dynamic W8A8 VLM/all linears."""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax_int8 import quantized_linears
import numpy as np

from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared.nnx_utils import module_jit
from openpi.so101.contract import PROMPT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=["bf16", "vlm", "all"])
    p.add_argument("--num-steps", nargs="+", type=int, default=[5, 10])
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()
    cfg = Pi0Config(pi05=True, action_horizon=10, loss_action_dim=6)
    if args.weights:
        model = cfg.load(model_lib.restore_params(args.weights, dtype=jnp.bfloat16))
    else:

        @nnx.jit
        def initialize():
            m = cfg.create(jax.random.key(0))
            nnx.update(m, jax.tree.map(lambda v: v.astype(jnp.bfloat16), nnx.state(m)))
            return m

        model = initialize()
    model.eval()
    obs = cfg.fake_obs()
    tokens, mask = PaligemmaTokenizer(200).tokenize(PROMPT, np.zeros(6, np.float32))
    obs = dataclasses.replace(
        obs,
        images={
            k: jax.random.uniform(jax.random.key(i + 50), v.shape, minval=-1, maxval=1)
            for i, (k, v) in enumerate(obs.images.items())
        },
        image_masks={k: jnp.array([k != "right_wrist_0_rgb"]) for k in obs.images},
        tokenized_prompt=jnp.asarray(tokens)[None],
        tokenized_prompt_mask=jnp.asarray(mask)[None],
    )
    key = jax.random.key(1)
    results = []
    references = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for steps in args.num_steps:
            counts = {}
            with quantized_linears(mode, counts):
                f = module_jit(model.sample_actions)
                start = time.monotonic()
                out = f(key, obs, num_steps=steps)
                jax.block_until_ready(out)
                compile_s = time.monotonic() - start
                times = []
                for i in range(args.iterations + 3):
                    t = time.monotonic()
                    out = f(key, obs, num_steps=steps)
                    jax.block_until_ready(out)
                    if i >= 3:
                        times.append((time.monotonic() - t) * 1000)
                value = np.asarray(out)[..., :6]
            if mode == "bf16":
                references[steps] = value
            ref = references.get(steps)
            row = {
                "mode": mode,
                "num_steps": steps,
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "mean_ms": float(np.mean(times)),
                "compile_first_call_s": compile_s,
                "intercepted_calls_at_trace": counts,
                "finite": bool(np.isfinite(value).all()),
                "normalized_action_rmse": None if ref is None else float(np.sqrt(np.mean((value - ref) ** 2))),
                "normalized_action_max_abs_error": None if ref is None else float(np.max(np.abs(value - ref))),
            }
            results.append(row)
            report = {
                "pretrained_weights": args.weights,
                "architecture": "full pi05 gemma_2b + gemma_300m + SigLIP",
                "synthetic_observation": True,
                "cameras": 2,
                "computed_image_slots": 3,
                "horizon": 10,
                "results": results,
                "quantization": "experimental native dynamic INT8/FP8 linears, includes weight quantization overhead; no prepacked low-bit weight storage",
                "quality_limit": "Normalized-action differences on synthetic inputs are not task success rates.",
            }
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row, indent=2), flush=True)
            del f
            jax.clear_caches()


if __name__ == "__main__":
    main()

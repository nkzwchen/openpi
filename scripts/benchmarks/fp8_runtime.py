"""Compare JAX BF16 with FlashRT BF16/FP8 on deterministic, synthetic SO101-shaped inputs.

FlashRT is an optional external source build supplied through PYTHONPATH. Each
backend runs in its own process. This is a benchmark, not a robot policy server.
"""

# Optional backends must be imported only after CLI selection.
# ruff: noqa: PLC0415

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np


def samples(seed, count, horizon):
    import ml_dtypes

    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        images = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
        noise = rng.standard_normal((horizon, 32)).astype(ml_dtypes.bfloat16).astype(np.float32)
        noise[:, 6:] = 0
        result.append((images, noise))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["jax", "flashrt-bf16", "flashrt-fp8", "flashrt-fp8-vlm"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--evaluation-samples", type=int, default=8)
    parser.add_argument("--jax-camera-slots", type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    for name in ("num_steps", "horizon", "iterations", "calibration_samples", "evaluation_samples", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation = samples(20261018, args.evaluation_samples, args.horizon)
    calibration = samples(20261019, args.calibration_samples, args.horizon)
    prompt = "pick blocks from table"
    state = np.zeros(6, np.float32)
    start = time.perf_counter()
    metadata = {}
    temp = None
    if args.backend == "jax":
        import jax
        import jax.numpy as jnp

        from openpi.models import model as model_lib
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared.nnx_utils import module_jit

        cfg = Pi0Config(pi05=True, action_horizon=args.horizon, loss_action_dim=6)
        model = cfg.load(model_lib.restore_params(str(args.weights.resolve()), dtype=jnp.bfloat16))
        model.eval()
        # Remove only the absent, masked camera in this benchmark process. The
        # production preprocessing and model source remain unchanged.
        if args.jax_camera_slots == 2:
            original_preprocess = model_lib.preprocess_observation

            def two_view_preprocess(rng, observation, **kwargs):
                kwargs["image_keys"] = ("base_0_rgb", "left_wrist_0_rgb")
                return original_preprocess(rng, observation, **kwargs)

            model_lib.preprocess_observation = two_view_preprocess
        tokens, mask = PaligemmaTokenizer(200).tokenize(prompt, state)
        obs_list = []
        for images, _ in evaluation:
            images_dict = {
                "base_0_rgb": jnp.asarray(images[0].astype(np.float32) / 127.5 - 1.0)[None],
                "left_wrist_0_rgb": jnp.asarray(images[1].astype(np.float32) / 127.5 - 1.0)[None],
                "right_wrist_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
            }
            obs_list.append(
                dataclasses.replace(
                    cfg.fake_obs(),
                    images=images_dict,
                    image_masks={k: jnp.array([k != "right_wrist_0_rgb"]) for k in images_dict},
                    tokenized_prompt=jnp.asarray(tokens)[None],
                    tokenized_prompt_mask=jnp.asarray(mask)[None],
                )
            )
        noises = [jnp.asarray(noise)[None] for _, noise in evaluation]
        fn = module_jit(model.sample_actions)
        key = jax.random.key(1)

        def infer(index):
            result = fn(key, obs_list[index], num_steps=args.num_steps, noise=noises[index])
            jax.block_until_ready(result)
            return result

        def extract(result):
            return np.asarray(result)[0, :, :6].astype(np.float32)

        metadata.update(
            jax=jax.__version__,
            device=str(jax.devices()),
            computed_image_slots=args.jax_camera_slots,
            timing_scope="ready device observation -> synchronized device actions; excludes H2D and preprocessing",
        )
    else:
        import flash_rt.frontends.jax.pi05_rtx as frontend
        import torch

        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)

        # Fold the production loss_action_dim=6 velocity mask into the output
        # projection once. Padded noise is zero, so inactive channels stay zero
        # at EVERY Euler step, matching openpi.models.pi0.sample_actions.
        convert = frontend.convert_pi05_orbax

        def convert_with_action_mask(path):
            weights = convert(path)
            weights["decoder_action_out_proj_w"][:, 6:] = 0
            weights["decoder_action_out_proj_b"][6:] = 0
            return weights

        frontend.convert_pi05_orbax = convert_with_action_mask
        if args.backend == "flashrt-fp8-vlm":
            precision_kwargs = frontend.Pi05JaxFrontendRtx._pipeline_precision_kwargs  # noqa: SLF001

            def keep_expert_bf16(self):
                kwargs = precision_kwargs(self)
                kwargs["use_fp8_decoder"] = False
                return kwargs

            frontend.Pi05JaxFrontendRtx._pipeline_precision_kwargs = keep_expert_bf16  # noqa: SLF001
        # The base params download has no robot norm stats. Identity statistics
        # are confined to this temporary benchmark wrapper. Compare raw actions.
        temp = tempfile.TemporaryDirectory(prefix="openpi-fp8-benchmark-")
        wrapper = Path(temp.name)
        (wrapper / "params").symlink_to(args.weights.resolve(), target_is_directory=True)
        identity = {"q01": [-1.0] * 32, "q99": [1.0] * 32, "mean": [0.0] * 32, "std": [1.0] * 32}
        (wrapper / "norm_stats.json").write_text(json.dumps({"norm_stats": {"actions": identity, "state": identity}}))
        pipe = frontend.Pi05JaxFrontendRtx(
            wrapper,
            num_views=2,
            chunk_size=args.horizon,
            num_steps=args.num_steps,
            use_fp8=args.backend.startswith("flashrt-fp8"),
            use_cuda_graph=True,
            hardware="rtx_sm89",
            cache_frames=1,
        )
        pipe.set_prompt(prompt, state=state)
        # Calibration and evaluation image sets use different seeds. Both are
        # synthetic and cannot establish closed-loop task success.
        torch.manual_seed(20261019)
        pipe.calibrate([{"images": images} for images, _ in calibration], percentile=100.0)
        torch.cuda.synchronize()
        if args.backend.startswith("flashrt-fp8") and not pipe.pipeline.use_fp8:
            raise RuntimeError("FP8 requested but runtime disabled FP8")

        def infer(index):
            images, noise = evaluation[index]
            result = pipe.infer({"images": images}, noise=noise)
            torch.cuda.synchronize()
            return result

        def extract(result):
            raw = pipe._noise_out.float().cpu().numpy()  # noqa: SLF001 - raw padded actions for parity checks
            if not np.all(raw[:, 6:] == 0):
                raise AssertionError("Inactive action channels must remain exactly zero")
            return raw[:, :6].copy()

        quantized_bytes = sum(v.numel() * v.element_size() for v in pipe._fp8_store)  # noqa: SLF001
        metadata.update(
            torch=torch.__version__,
            cpu_threads=torch.get_num_threads(),
            device=torch.cuda.get_device_name(),
            computed_image_slots=2,
            cache_frames=1,
            use_cuda_graph=pipe.use_cuda_graph,
            use_fp8=pipe.pipeline.use_fp8,
            use_fp8_decoder=pipe.pipeline.use_fp8_decoder,
            fp8_tensor_bytes=quantized_bytes,
            calibration_seed=20261019,
            calibration_percentile=100.0,
            timing_scope="host uint8 images + fixed prompt -> synchronized actions; includes image prep/H2D and output copy; prompt tokenization excluded",
            external_frontend_file=str(Path(frontend.__file__).resolve()),
            frontend_sha256=hashlib.sha256(Path(frontend.__file__).read_bytes()).hexdigest(),
        )
    infer(0)
    startup_s = time.perf_counter() - start
    for i in range(args.warmup):
        infer(i % len(evaluation))
    timings = []
    for i in range(args.iterations):
        t = time.perf_counter()
        infer(i % len(evaluation))
        timings.append((time.perf_counter() - t) * 1000)
    actions = [extract(infer(i)).tolist() for i in range(len(evaluation))]
    value = np.asarray(actions, np.float32)
    if not np.isfinite(value).all():
        raise RuntimeError("Non-finite actions")
    report = {
        "backend": args.backend,
        "weights": str(args.weights.resolve()),
        "prompt": prompt,
        "horizon": args.horizon,
        "effective_action_dim": 6,
        "num_steps": args.num_steps,
        "synthetic_inputs": True,
        "evaluation_seed": 20261018,
        "state": state.tolist(),
        "evaluation_samples": len(evaluation),
        "calibration_samples": len(calibration),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "startup_compile_calibration_s": startup_s,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "mean_ms": float(np.mean(timings)),
        "finite": True,
        "metadata": metadata,
        "normalized_actions": actions,
        "quality_limit": "Synthetic-input numerical parity only, no task success evaluation; native BF16 runtime differences must be assessed separately from FP8 quantization.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "normalized_actions"}, indent=2), flush=True)
    if temp:
        temp.cleanup()


if __name__ == "__main__":
    main()

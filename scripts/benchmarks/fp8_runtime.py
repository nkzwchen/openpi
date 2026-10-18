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


def samples(seed, count, horizon, kind="random"):
    import ml_dtypes

    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        images = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
        if kind == "mixed":
            for view in range(2):
                case = (index + view) % 6
                if case == 1:
                    images[view] //= 8
                elif case == 2:
                    images[view] = 224 + images[view] // 8
                elif case == 3:
                    x = np.linspace(0, 255, 224, dtype=np.uint8)
                    images[view] = np.broadcast_to(x[None, :, None], (224, 224, 3)).copy()
                elif case == 4:
                    images[view] = np.repeat(np.repeat(images[view][:32, :32], 7, axis=0), 7, axis=1)
                elif case == 5:
                    images[view][:112, :112] = [220, 30, 20]
                    images[view][112:, 112:] = [20, 200, 40]
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
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--calibration-seed", type=int, default=20261019)
    parser.add_argument("--sample-kind", choices=["random", "mixed"], default="random")
    parser.add_argument("--vary-state", action="store_true")
    parser.add_argument("--include-prompt-update", action="store_true")
    parser.add_argument("--state-prompt-mode", choices=["exact", "fixed"], default="exact")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--request-hz", type=float, default=0, help="Pace requests; zero means continuous")
    parser.add_argument("--calibration-percentile", type=float, default=100.0)
    parser.add_argument("--jax-camera-slots", type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    for name in (
        "num_steps",
        "horizon",
        "iterations",
        "calibration_samples",
        "evaluation_samples",
        "cpu_threads",
        "rounds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.request_hz < 0:
        parser.error("--request-hz must be nonnegative")
    if args.vary_state and not args.include_prompt_update:
        parser.error("--vary-state requires --include-prompt-update so state-update cost is measured")
    if not 0 <= args.calibration_percentile <= 100:
        parser.error("--calibration-percentile must be between 0 and 100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation = samples(args.seed, args.evaluation_samples, args.horizon, args.sample_kind)
    calibration = samples(args.calibration_seed, args.calibration_samples, args.horizon, args.sample_kind)
    prompt = "pick blocks from table"
    state = np.zeros(6, np.float32)
    states = (
        np.random.default_rng(args.seed + 100).uniform(-0.95, 0.95, (len(evaluation), 6)).astype(np.float32)
        if args.vary_state
        else np.zeros((len(evaluation), 6), np.float32)
    )
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
        tokenizer = PaligemmaTokenizer(200)
        obs_list = []
        for index, (images, _) in enumerate(evaluation):
            tokens, mask = tokenizer.tokenize(prompt, states[index])
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
            obs = obs_list[index]
            if args.include_prompt_update:
                live_tokens, live_mask = tokenizer.tokenize(prompt, states[index])
                obs = dataclasses.replace(
                    obs,
                    tokenized_prompt=jnp.asarray(live_tokens)[None],
                    tokenized_prompt_mask=jnp.asarray(live_mask)[None],
                )
            result = fn(key, obs, num_steps=args.num_steps, noise=noises[index])
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
            state_prompt_mode=args.state_prompt_mode,
        )
        pipe.set_prompt(prompt, state=state)
        # Calibration and evaluation image sets use different seeds. Both are
        # synthetic and cannot establish closed-loop task success.
        torch.manual_seed(args.calibration_seed)
        pipe.calibrate([{"images": images} for images, _ in calibration], percentile=args.calibration_percentile)
        torch.cuda.synchronize()
        if args.backend.startswith("flashrt-fp8") and not pipe.pipeline.use_fp8:
            raise RuntimeError("FP8 requested but runtime disabled FP8")

        def infer(index):
            images, noise = evaluation[index]
            if args.include_prompt_update:
                pipe.set_prompt(prompt, state=states[index])
                if not pipe.calibrated:
                    pipe.calibrate([{"images": images}])
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
            calibration_seed=args.calibration_seed,
            calibration_percentile=args.calibration_percentile,
            calibration_state="zero six-dimensional state; image variation only",
            state_prompt_mode=args.state_prompt_mode,
            timing_scope="host uint8 images + fixed prompt -> synchronized actions; includes image prep/H2D and output copy; prompt tokenization excluded",
            external_frontend_file=str(Path(frontend.__file__).resolve()),
            frontend_sha256=hashlib.sha256(Path(frontend.__file__).read_bytes()).hexdigest(),
        )
    if args.include_prompt_update:
        metadata["timing_scope"] = (
            "prepared device images + host state -> tokenize, upload tokens, synchronized device actions"
            if args.backend == "jax"
            else "host uint8 images + host state -> update prompt, image prep/H2D, synchronized host actions; "
            "initial model load and calibration excluded; newly encountered exact-mode graph setup included"
        )
    metadata["includes_prompt_update"] = args.include_prompt_update
    infer(0)
    startup_s = time.perf_counter() - start
    for i in range(args.warmup):
        infer(i % len(evaluation))
    timings = []
    round_results = []
    for round_index in range(args.rounds):
        round_times = []
        next_request = time.perf_counter()
        for i in range(args.iterations):
            if args.request_hz:
                time.sleep(max(0, next_request - time.perf_counter()))
                next_request += 1 / args.request_hz
            t = time.perf_counter()
            infer(i % len(evaluation))
            round_times.append((time.perf_counter() - t) * 1000)
        timings.extend(round_times)
        round_results.append(
            {
                "round": round_index,
                "median_ms": float(np.median(round_times)),
                "p99_ms": float(np.percentile(round_times, 99)),
                "max_ms": max(round_times),
            }
        )
    actions = [extract(infer(i)).tolist() for i in range(len(evaluation))]
    value = np.asarray(actions, np.float32)
    repeat_max_abs_error = float(np.max(np.abs(extract(infer(0)) - value[0])))
    if repeat_max_abs_error != 0:
        raise AssertionError(f"Repeated input/noise changed output: {repeat_max_abs_error}")
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
        "evaluation_seed": args.seed,
        "sample_kind": args.sample_kind,
        "vary_state": args.vary_state,
        "include_prompt_update": args.include_prompt_update,
        "evaluation_states": states.tolist(),
        "round_results": round_results,
        "timing_samples": len(timings),
        "request_hz": args.request_hz,
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
        "max_ms": max(timings),
        "over_100ms": sum(t > 100 for t in timings),
        "finite": True,
        "repeat_max_abs_error": repeat_max_abs_error,
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

"""Combine measured JSON files and explicitly labeled two-GPU estimates."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(name):
        return json.loads((args.directory / name).read_text())

    train = read("train-b8-pretrained-pipeline640.json")
    one_gpu = next(x["samples_per_second"] for x in train["native_async_pipeline"] if x["num_workers"] == 4)
    estimates = []
    for length in (10, 30, 60):
        episodes = 1800 // length
        for train_fraction in (1.0, 0.8):
            selected = round(episodes * train_fraction)
            windows = selected * (length * 10 - 9)
            estimates.append(
                {
                    "episode_seconds": length,
                    "total_recording_minutes": 30,
                    "training_fraction": train_fraction,
                    "training_episodes": selected,
                    "valid_windows": windows,
                    "global_batch": 16,
                    "per_gpu_batch": 8,
                    "steps_drop_last": windows // 16,
                    "steps_keep_partial": (windows + 15) // 16,
                    "dual_gpu_efficiency_assumption": [0.7, 0.9],
                    "dual_gpu_samples_per_second_estimate": [2 * one_gpu * 0.7, 2 * one_gpu * 0.9],
                    "epoch_minutes_estimate": [windows / (2 * one_gpu * 0.9) / 60, windows / (2 * one_gpu * 0.7) / 60],
                }
            )
    quant = []
    for filename in ("inference-pretrained-int8.json", "inference-pretrained-fp8.json"):
        report = read(filename)
        baseline = {x["num_steps"]: x["median_ms"] for x in report["results"] if x["mode"] == "bf16"}
        for row in report["results"]:
            if row["mode"] == "bf16" and filename.endswith("fp8.json"):
                continue
            quant.append({**row, "speedup_vs_bf16": baseline[row["num_steps"]] / row["median_ms"]})
    result = {
        "date": "2026-10-18",
        "hardware": {
            "measured_gpus": 1,
            "gpu": "NVIDIA L40S",
            "vram_reported_mib": 46068,
            "dual_gpu": "estimated two same-model L40S, not measured",
        },
        "versions": {"jax": "0.5.3", "flax": "0.10.2"},
        "protocol": {
            "synthetic_inputs": True,
            "real_robot_started": False,
            "pretrained_weights_source": "gs://openpi-assets/checkpoints/pi05_base/params",
            "cameras": 2,
            "computed_image_slots": 3,
            "horizon": 10,
            "effective_action_dim": 6,
            "inference_iterations_per_case": 40,
            "inference_warmup_iterations": 3,
            "training_iterations_per_case": 20,
            "measurements_synchronized": True,
            "compile_excluded": True,
        },
        "training": train,
        "two_gpu_epoch_estimates": estimates,
        "quantization": quant,
        "quantization_limit": "Dynamic INT8/FP8 linear operations, weight quantization overhead included; no prepacked weight storage; no task accuracy assessment",
        "native_kernel_evidence": read("native-kernels.json"),
        "remote_latency": read("remote-latency-with-prep.json"),
        "regression_tests": {"cloud_passed": 26, "local_passed": 24, "local_skipped": 2},
        "rtc_limit": "Current AsyncChunkPolicy does not implement complete RTC prefetch and prefix guidance; recommendations require further implementation and measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

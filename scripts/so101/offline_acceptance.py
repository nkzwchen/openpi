# ruff: noqa: PLC0415, SLF001, PT018
"""A/B acceptance on marked synthetic data. Does not import robot hardware APIs."""

import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from openpi.so101.export import export_dataset
from openpi.so101.synthetic import write_fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-smoke", action="store_true", help="Needs OpenPI JAX environment/GPU; random dummy Gemma + real SigLIP"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = write_fixtures(out / "raw")
    report = export_dataset(sources, out / "dataset")
    (out / "acceptance.json").write_text(
        json.dumps({"synthetic": True, "export": report, "model_smoke": "not_run"}, indent=2)
    )
    if not args.model_smoke:
        print(out / "acceptance.json")
        return

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.shared import normalize
    from openpi.training import config as configs
    from openpi.training import data_loader
    from openpi.training.local_lerobot_v3_dataset import LocalLeRobotV3Dataset

    print("Checking real dataset loader and complete-window sampling", flush=True)
    train = LocalLeRobotV3Dataset(
        out / "dataset/train", action_horizon=10, action_sequence_keys=("action",), decode_videos=False
    )
    validation = LocalLeRobotV3Dataset(
        out / "dataset/validation", action_horizon=10, action_sequence_keys=("action",), decode_videos=False
    )
    assert len(train) == 39 and len(validation) == 39
    for dataset in (train, validation):
        for i in range(len(dataset)):
            item = dataset[i]
            ep = dataset._episode_by_index.loc[int(item["episode_index"])]
            assert int(item["index"]) + 10 <= int(ep["dataset_to_index"])
            # Labels are six absolute positions; consecutive targets are distinct.
            assert not np.array_equal(item["action"][-1], item["action"][-2])
    decoded = LocalLeRobotV3Dataset(
        out / "dataset/train",
        action_horizon=10,
        action_sequence_keys=("action",),
        decode_videos=True,
        video_backend="pyav",
    )
    image = decoded[0]
    assert image["observation.images.cam_main"][..., 0].mean() > 150
    assert image["observation.images.cam_wrist"][..., 2].mean() > 150
    print("Decoded both full-frame videos with RGB camera order intact", flush=True)
    cfg = configs.get_config("pi05_so101_absolute6")
    # Small language/action transformers; retain actual vision, loss and sampler implementation.
    model_cfg = dataclasses.replace(
        cfg.model, paligemma_variant="dummy", action_expert_variant="dummy", dtype="bfloat16"
    )
    cfg = dataclasses.replace(
        cfg,
        model=model_cfg,
        assets_base_dir=str(out / "assets"),
        data=dataclasses.replace(cfg.data, repo_id=str(out / "dataset/train")),
        policy_metadata={"synthetic": True},
    )
    from openpi.so101.training import compute_norm_stats
    from openpi.so101.training import dataset_contract

    try:
        dataset_contract(out / "dataset/train")
    except ValueError:
        pass  # Production training must reject synthetic fixtures.
    else:
        raise AssertionError("Synthetic data unexpectedly accepted for production")
    assert compute_norm_stats(cfg) == 39
    norm = normalize.load(cfg.assets_dirs / "so101_openpi_absolute_v1")
    normalize.save(out / "checkpoint/assets/so101_openpi_absolute_v1", norm)
    datacfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = data_loader.create_torch_dataset(datacfg, 10, cfg.model)
    ds = data_loader.transform_dataset(ds, datacfg)
    example = ds[0]
    assert example["state"].shape == (32,) and example["actions"].shape == (10, 32)
    raw = train[0]
    processed = {"state": raw["observation.state"], "actions": raw["action"]}
    roundtrip = transforms.Unnormalize(norm, use_quantiles=True)(
        transforms.Normalize(norm, use_quantiles=True)(processed.copy())
    )
    np.testing.assert_allclose(roundtrip["actions"], processed["actions"], atol=1e-4)
    np.testing.assert_allclose(roundtrip["state"], processed["state"], atol=1e-4)
    batch = jax.tree.map(lambda x: jnp.asarray(x)[None], example)
    observation = model_lib.Observation.from_dict(batch)
    actions = batch["actions"]
    print("Creating random dummy-Gemma / real-SigLIP model; not a trained policy", flush=True)
    started = time.monotonic()
    model = model_cfg.create(jax.random.key(0))
    model.eval()
    selected = nnx.All(nnx.Param, nnx_utils.PathRegex(r"action_out_proj/.*"))
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=selected)

    @nnx.jit
    def step(model, optimizer):
        def loss_fn(m):
            return jnp.mean(m.compute_loss(jax.random.key(4), observation, actions, train=False))

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, selected))(model)
        grad_norm = optax.global_norm(grads)
        padded_gradient = jnp.max(jnp.abs(grads["action_out_proj"]["kernel"].value[:, 6:]))
        optimizer.update(grads)
        return loss, grad_norm, padded_gradient

    loss, grad_norm, padded_gradient = step(model, optimizer)
    assert float(padded_gradient) == 0.0
    loss, grad_norm = float(loss), float(grad_norm)
    assert np.isfinite(loss) and np.isfinite(grad_norm) and grad_norm > 0
    # Loss selection must not alter checkpoint parameter shapes.
    baseline = dataclasses.replace(model_cfg, loss_action_dim=None)
    baseline_shapes = nnx.state(nnx.eval_shape(baseline.create, jax.random.key(0))).to_pure_dict()
    actual_shapes = nnx.state(model).to_pure_dict()
    assert jax.tree.structure(baseline_shapes) == jax.tree.structure(actual_shapes)
    assert all(
        a.shape == b.shape
        for a, b in zip(jax.tree.leaves(baseline_shapes), jax.tree.leaves(actual_shapes), strict=True)
    )
    print(f"Finite loss={loss:.5f}, nonzero action-head gradient={grad_norm:.5f}; saving checkpoint", flush=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(out / "checkpoint/params"), {"params": actual_shapes})
    key = jax.random.key(12)
    before = nnx_utils.module_jit(model.sample_actions)(key, observation, num_steps=2)
    before = np.asarray(before)
    assert np.all(before[..., 6:] == 0)
    restored = model_cfg.load(model_lib.restore_params(out / "checkpoint/params"))
    after = np.asarray(nnx_utils.module_jit(restored.sample_actions)(key, observation, num_steps=2))
    np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-4)
    policy = policy_config.create_trained_policy(cfg, out / "checkpoint", sample_kwargs={"num_steps": 2})
    from openpi.so101.contract import observation_to_input

    obs = {
        "state": raw["observation.state"],
        "cam_main": image["observation.images.cam_main"],
        "cam_wrist": image["observation.images.cam_wrist"],
    }
    output = policy.infer(observation_to_input(obs))
    assert output["actions"].shape == (10, 6) and np.isfinite(output["actions"]).all()
    result = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "synthetic": True,
        "export": report,
        "model_smoke": {
            "variant": "dummy Gemma, real SigLIP, random initialization",
            "gradient_scope": "action_out_proj only; not full fine-tuning",
            "loss": loss,
            "gradient_norm": grad_norm,
            "padded_action_head_gradient": float(padded_gradient),
            "inactive_sample_channels_zero": True,
            "loss_action_dim": 6,
            "parameter_shapes_compatible": True,
            "reload_max_abs_error": float(np.max(np.abs(before - after))),
            "output_shape": list(output["actions"].shape),
            "elapsed_s": time.monotonic() - started,
            "devices": [str(d) for d in jax.devices()],
        },
        "limitations": [
            "No real absolute-action demonstrations available",
            "No pretrained checkpoint validated",
            "No hardware or GUI motion validation",
        ],
    }
    (out / "acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["model_smoke"], indent=2), flush=True)


if __name__ == "__main__":
    main()

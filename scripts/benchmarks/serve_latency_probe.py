"""Synthetic-only full model endpoint for remote latency measurement; never robot control."""

import argparse

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.policies.policy import Policy
from openpi.policies.so101_absolute_policy import AbsoluteInputs
from openpi.policies.so101_absolute_policy import AbsoluteOutputs
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.so101.contract import PROMPT
from openpi.training.config import ModelTransformFactory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights")
    p.add_argument("--port", type=int, default=19090)
    p.add_argument("--num-steps", type=int, default=5)
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
    policy = Policy(
        model,
        transforms=[AbsoluteInputs(), *ModelTransformFactory()(cfg).inputs],
        output_transforms=[AbsoluteOutputs()],
        sample_kwargs={"num_steps": args.num_steps},
    )
    warm = {
        "observation/state": np.zeros(6, np.float32),
        "observation/cam_main": np.zeros((224, 224, 3), np.uint8),
        "observation/cam_wrist": np.zeros((224, 224, 3), np.uint8),
        "prompt": PROMPT,
    }
    policy.infer(warm)
    print("LATENCY_PROBE_READY", flush=True)
    WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=args.port,
        metadata={
            "synthetic": True,
            "benchmark": True,
            "num_steps": args.num_steps,
            "pretrained_weights": args.weights,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()

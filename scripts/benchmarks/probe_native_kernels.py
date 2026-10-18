"""Record StableHLO operand types and GPU backend calls for INT8/FP8 matmuls."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax_int8 import quant_dot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {}
    x = jax.random.normal(jax.random.key(0), (256, 2048), dtype=jnp.bfloat16)
    w = jax.random.normal(jax.random.key(1), (2048, 2048), dtype=jnp.bfloat16)
    for kind in ("int8", "fp8"):
        try:
            function = jax.jit(lambda a, b, kind=kind: quant_dot(a, b, kind=kind))
            lowered = function.lower(x, w)
            compiled = lowered.compile()
            value = compiled(x, w)
            jax.block_until_ready(value)
            backend = compiled.as_text()
            result[kind] = {
                "supported": True,
                "stablehlo_dots": [line.strip() for line in lowered.as_text().splitlines() if "dot_general" in line],
                "backend_gemm_lines": [
                    line.strip()[:2000]
                    for line in backend.splitlines()
                    if "custom-call" in line and ("cublas" in line or "gemm" in line)
                ],
            }
        except Exception as exc:
            result[kind] = {"supported": False, "error": repr(exc)}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

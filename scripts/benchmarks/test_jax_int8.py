"""Small arithmetic/interception checks; can run on CPU without allocating full pi05."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_int8 import quant_dot
from jax_int8 import quantized_linears

from openpi.models import lora


def main():
    x = jax.random.normal(jax.random.key(0), (2, 4, 64))
    for module in (nn.Dense(32), nn.DenseGeneral((4, 8)), lora.Einsum((64, 32)), lora.FeedForward(64, 128)):
        args = ("BTD,DF->BTF", x) if isinstance(module, lora.Einsum) else (x,)
        params = module.init(jax.random.key(1), *args)
        if isinstance(module, lora.Einsum):
            params["params"]["w"] = jax.random.normal(jax.random.key(2), (64, 32))
        reference = module.apply(params, *args)
        counts = {}
        with quantized_linears("all", counts):
            actual = module.apply(params, *args)
        relative = float(jnp.linalg.norm(actual - reference) / (jnp.linalg.norm(reference) + 1e-8))
        assert counts, (type(module), counts)
        assert relative < 0.06, (type(module), relative)
        print(type(module).__name__, relative, counts, flush=True)
    hlo = jax.jit(quant_dot).lower(jnp.ones((32, 64), jnp.bfloat16), jnp.ones((64, 32), jnp.bfloat16)).as_text()
    assert "xi8>" in hlo
    assert "xi32>" in hlo
    assert "dot_general" in hlo
    print("Native INT8 operands and INT32 dot accumulation confirmed in StableHLO")


if __name__ == "__main__":
    main()

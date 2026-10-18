"""Experimental native W8A8 dot products on JAX 0.5; not fake quantization.

Weights and activations are dynamically quantized during the compiled forward.
This intentionally includes quantization overhead; it is NOT a prepacked serving
backend. Norms, softmax, embedding lookup, conv patch embedding, attention QK/AV,
and the small NNX action projections retain floating-point arithmetic.
"""

from contextlib import contextmanager

import flax.linen as nn
import jax.numpy as jnp

from openpi.models import lora

_EINSUM = jnp.einsum


def quant_einsum(eqn, x, w, kind="int8"):
    left, right = eqn.split("->")[0].split(",")
    out = eqn.split("->")[1]
    contracted = set(left) & set(right) - set(out)
    xaxes = tuple(i for i, c in enumerate(left) if c in contracted)
    waxes = tuple(i for i, c in enumerate(right) if c in contracted)
    if not xaxes or not waxes:
        return _EINSUM(eqn, x, w)
    bound = 448 if kind == "fp8" else 127
    qtype = jnp.float8_e4m3fn if kind == "fp8" else jnp.int8
    accum = jnp.float32 if kind == "fp8" else jnp.int32
    xs = jnp.maximum(jnp.max(jnp.abs(x.astype(jnp.float32)), axis=xaxes, keepdims=True) / bound, 1e-8)
    ws = jnp.maximum(jnp.max(jnp.abs(w.astype(jnp.float32)), axis=waxes, keepdims=True) / bound, 1e-8)
    qx = jnp.clip(x / xs, -bound, bound)
    qw = jnp.clip(w / ws, -bound, bound)
    if kind == "int8":
        qx, qw = jnp.rint(qx), jnp.rint(qw)
    qx, qw = qx.astype(qtype), qw.astype(qtype)
    integer = _EINSUM(eqn, qx, qw, preferred_element_type=accum)
    scale = _EINSUM(eqn, xs, ws)
    return (integer.astype(jnp.float32) * scale).astype(x.dtype)


def quant_dot(x, w, kind="int8"):
    shape = x.shape[:-1]
    out = w.shape[1:]
    return quant_einsum("MK,KN->MN", x.reshape((-1, x.shape[-1])), w.reshape((w.shape[0], -1)), kind=kind).reshape(
        (*shape, *out)
    )


@contextmanager
def quantized_linears(mode, counts=None):
    counts = {} if counts is None else counts
    if mode == "bf16":
        yield
        return

    kind = "fp8" if mode.endswith("fp8") else "int8"

    def intercept(next_fun, args, kwargs, ctx):
        module = ctx.module
        path = "/".join(module.scope.path) if module.scope is not None else ""
        expert = any(
            part
            in {
                "q_einsum_1",
                "kv_einsum_1",
                "qkv_einsum_1",
                "attn_vec_einsum_1",
                "mlp_1",
                "final_norm_1",
                "pre_attention_norm_1",
                "pre_ffw_norm_1",
            }
            for part in path.split("/")
        )
        selected = mode.startswith("all") or not expert
        if not selected:
            return next_fun(*args, **kwargs)
        if isinstance(module, lora.Einsum) and ctx.method_name == "__call__":
            if module.lora_config is not None:
                raise ValueError("Benchmark unmerged LoRA is unsupported")
            counts["gemma_einsum"] = counts.get("gemma_einsum", 0) + 1
            eqn, x = args
            return quant_einsum(eqn, x, module.w.astype(x.dtype), kind=kind)
        if isinstance(module, lora.FeedForward) and ctx.method_name == "_dot":
            x, w, lora_weights = args
            if lora_weights is not None:
                raise ValueError("Benchmark unmerged LoRA is unsupported")
            counts["gemma_ffn"] = counts.get("gemma_ffn", 0) + 1
            return quant_dot(x, w.astype(x.dtype), kind=kind)
        if isinstance(module, (nn.Dense, nn.DenseGeneral)) and ctx.method_name == "__call__":
            x = args[0]
            dtype = jnp.dtype(module.dtype) if module.dtype else x.dtype
            x = x.astype(dtype)
            w = module.get_variable("params", "kernel").astype(dtype)
            if isinstance(module, nn.DenseGeneral):
                axes = module.axis if isinstance(module.axis, tuple) else (module.axis,)
                axes = tuple(a % x.ndim for a in axes)
                if module.batch_dims or axes != tuple(range(x.ndim - len(axes), x.ndim)):
                    return next_fun(*args, **kwargs)
                contracted = 1
                for d in x.shape[-len(axes) :]:
                    contracted *= d
                outputshape = w.shape[len(axes) :]
                y = quant_dot(x.reshape((*x.shape[: -len(axes)], contracted)), w.reshape((contracted, -1)), kind=kind)
                y = y.reshape((*x.shape[: -len(axes)], *outputshape))
            else:
                y = quant_dot(x, w, kind=kind)
            if module.use_bias:
                y = y + module.get_variable("params", "bias").astype(dtype)
            counts["linen_dense"] = counts.get("linen_dense", 0) + 1
            return y
        return next_fun(*args, **kwargs)

    with nn.intercept_methods(intercept):
        yield

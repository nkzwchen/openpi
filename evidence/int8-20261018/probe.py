"""Local diagnostic: compare dynamic/prequantized weights and inspect optimized HLO."""
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, str(Path.cwd() / 'scripts/benchmarks'))
from jax_int8 import quant_dot

outdir = Path('docs/so101/quant-audit')
outdir.mkdir(parents=True, exist_ok=True)
rows = []
for m,k,n in [(10,2048,2048), (256,2048,2048), (968,2048,2048), (968,2048,16384)]:
    x = jax.random.normal(jax.random.key(1), (m,k), dtype=jnp.bfloat16)
    w = jax.random.normal(jax.random.key(2), (k,n), dtype=jnp.bfloat16)
    ws = jnp.maximum(jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0, keepdims=True)/127, 1e-8)
    qw = jnp.rint(jnp.clip(w/ws,-127,127)).astype(jnp.int8)
    xs = jnp.maximum(jnp.max(jnp.abs(x.astype(jnp.float32)), axis=1, keepdims=True)/127, 1e-8)
    qx = jnp.rint(jnp.clip(x/xs,-127,127)).astype(jnp.int8)
    def prequant(a,b,s):
        a_scale = jnp.maximum(jnp.max(jnp.abs(a.astype(jnp.float32)), axis=1, keepdims=True)/127,1e-8)
        aq = jnp.rint(jnp.clip(a/a_scale,-127,127)).astype(jnp.int8)
        accum = jnp.dot(aq,b,preferred_element_type=jnp.int32)
        return (accum.astype(jnp.float32)*a_scale*s).astype(a.dtype)
    cases = [
        ('bf16',lambda a,b: jnp.dot(a,b), (x,w)),
        ('dynamic_int8',quant_dot,(x,w)),
        ('prequant_weight_int8',prequant,(x,qw,ws)),
        ('raw_int8_gemm',lambda a,b: jnp.dot(a,b,preferred_element_type=jnp.int32),(qx,qw)),
    ]
    for mode,fn,args in cases:
        compiled = jax.jit(fn).lower(*args).compile()
        hlo = compiled.as_text()
        (outdir/f'{m}-{k}-{n}-{mode}.hlo').write_text(hlo)
        for _ in range(5): jax.block_until_ready(compiled(*args))
        timings=[]
        for _ in range(60):
            t = time.perf_counter()
            jax.block_until_ready(compiled(*args))
            timings.append((time.perf_counter()-t)*1000)
        row = {'shape':[m,k,n], 'mode':mode,'median_ms':float(np.median(timings)),
               'backend_calls':[line.strip()[:1500] for line in hlo.splitlines() if ('custom_call_target=' in line or '__triton' in line)],
               'hlo_file':f'{m}-{k}-{n}-{mode}.hlo'}
        rows.append(row)
        print(json.dumps(row),flush=True)
(outdir/'results.json').write_text(json.dumps({'jax':jax.__version__,'device':str(jax.devices()),'results':rows},indent=2))

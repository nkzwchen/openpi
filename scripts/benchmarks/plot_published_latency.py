"""Plot the historical FP8 comparison (requires matplotlib, no model/GPU)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[2]
data=json.loads((root/'evidence/fp8-real-20261018/summary.json').read_text())
rows={(r['backend'],r['steps']):r for r in data['rows']}
fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
x=np.arange(2)
for i,(backend,label,color) in enumerate([
 ('jax','JAX BF16','#64748b'),('flashrt-bf16','FlashRT BF16','#3680b9'),
 ('flashrt-fp8-vlm','FlashRT VLM FP8 + expert BF16','#1b9e77')]):
 values=[rows[backend,n]['median_ms'] for n in (5,10)]
 bars=ax.bar(x+(i-1)*.24,values,.22,label=label,color=color)
 ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
ax.set_xticks(x,['5 denoising steps','10 denoising steps'])
ax.set_ylabel('Median latency (ms), lower is better');ax.set_ylim(0,78)
ax.set_title('SO101 recorded observations / pi0.5 base / NVIDIA L40S')
ax.legend(loc='upper left',fontsize=9);ax.spines[['top','right']].set_visible(False)
fig.supxlabel('2026-10-18 | 600 timings per configuration | backend timing scopes differ\nNo closed-loop robot success evaluation',fontsize=9)
out=root/'evidence/fp8-latency.png';fig.savefig(out,dpi=160);print(out)

"""Verify published historical numbers on CPU; does not run model inference."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    evidence = root / 'evidence'
    manifest = json.loads((evidence / 'manifest.json').read_text())
    for item in manifest['files']:
        path = evidence / item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['published_sha256']:
            raise ValueError(f'Artifact changed: {path.name}')
    folder = evidence / 'fp8-real-20261018'
    summary = json.loads((folder / 'summary.json').read_text())
    reports = {}
    for path in sorted(folder.glob('*-s*.json')):
        report = json.loads(path.read_text())
        if report['synthetic_inputs'] is not False:
            raise ValueError('Expected recorded observations')
        reports[report['backend'], report['num_steps']] = report
    if len(reports) != 6:
        raise ValueError('Expected three backends at two denoising lengths')
    for row in summary['rows']:
        data = reports[row['backend'], row['steps']]
        for key in ('median_ms', 'p99_ms', 'max_ms', 'timing_samples'):
            np.testing.assert_allclose(row[key], data[key], rtol=1e-10)
        for key, backend in [('speedup_vs_jax', 'jax'), ('speedup_vs_native_bf16', 'flashrt-bf16')]:
            np.testing.assert_allclose(row[key], reports[backend, row['steps']]['median_ms'] / data['median_ms'])
        fixture = data['metadata']['real_fixture']
        if set(fixture['calibration_episodes']) & set(fixture['evaluation_episodes']):
            raise ValueError('Calibration/evaluation episode overlap')
        print(f"{row['backend']:20s} steps={row['steps']:2d} median={row['median_ms']:.3f} ms p99={row['p99_ms']:.3f} ms")
    for error in summary['errors']:
        data = reports[error['backend'], error['steps']]
        reference = reports['jax' if error['reference'] == 'jax' else 'flashrt-bf16', error['steps']]
        if data['evaluation_states'] != reference['evaluation_states']:
            raise ValueError('State inputs differ across backends')
        if data['metadata']['real_fixture']['samples_sha256'] != reference['metadata']['real_fixture']['samples_sha256']:
            raise ValueError('Fixture inputs differ across backends')
        delta = np.asarray(data['normalized_actions'], dtype=np.float64) - np.asarray(reference['normalized_actions'], dtype=np.float64)
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite model output')
        np.testing.assert_allclose(error['rmse'], np.sqrt(np.mean(delta**2)), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(error['max_abs_error'], np.abs(delta).max(), rtol=1e-10, atol=1e-12)
    print('PASS: hashes, latency summaries, input alignment, episode split and action error recomputation.')
    print('This verifies saved outputs only; it does not reproduce GPU timings or robot success rates.')


if __name__ == '__main__':
    main()

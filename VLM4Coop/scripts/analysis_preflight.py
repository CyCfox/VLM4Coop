"""Seeded numerical and PDF-export witness; no experiment result is read."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

root = Path('VLM4Coop')
spec = json.loads((root / 'configs/compute-analysis.json').read_text())
spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8]
torch.manual_seed(0)
x = torch.randn(8, 8)
y = x @ x.T
assert y.shape == (8, 8) and torch.isfinite(y).all()
assert torch.equal(y, y.T)
fig, ax = plt.subplots(figsize=(3.5, 2))
ax.plot(np.arange(8), y.diag().numpy())
ax.set_title('Dependency witness only; no experiment data', fontsize=8)
fig.tight_layout()
output = root / 'reports/analysis_dependency_witness.pdf'
fig.savefig(output)
plt.close(fig)
assert output.read_bytes().startswith(b'%PDF')
result = dict(status='passed', spec_hash=spec_hash, device='cpu', shape=list(y.shape),
              witness_trace=float(y.trace()), pdf_bytes=output.stat().st_size,
              packages={name: importlib.metadata.version(name) for name in
                        ('matplotlib', 'numpy', 'Pillow', 'torch')},
              experiment_results_read=False)
(root / 'reports/analysis_preflight.json').write_text(json.dumps(result, indent=2))
print('WITNESS ' + json.dumps(result))

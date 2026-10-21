"""Seeded kernel witness and actual dependency/device visibility evidence."""
import json
import importlib.metadata as metadata
import torch
import numpy as np
import transformers
import torchvision
from pathlib import Path
from vlm4coop.training.configuration import PROJECT

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, 'Only authorized GPU 0 should be visible'
assert torch.__version__.startswith('2.6.0')
assert np.__version__ == '1.26.4'
assert transformers.__version__ == '5.3.0'
assert torchvision.__version__.startswith('0.21.0')
torch.manual_seed(7)
x = torch.randn(64, 64, device='cuda')
y = x @ x.T
assert torch.isfinite(y).all()
assert torch.allclose(y.diag(), (x*x).sum(-1), rtol=1e-5, atol=1e-5)
result = {'status': 'passed', 'torch': torch.__version__, 'transformers': transformers.__version__,
          'torchvision': torchvision.__version__, 'numpy': np.__version__,
          'gpu': torch.cuda.get_device_name(0), 'visible_gpu_count': torch.cuda.device_count(),
          'bf16_supported': torch.cuda.is_bf16_supported(), 'kernel_diagonal_sum': y.diag().sum().item()}
(PROJECT / 'reports/gpu_preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))

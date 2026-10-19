"""Small real GPU numerical parity check for optional official inference kernels."""
import json
import time
from pathlib import Path
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule
from vlm4coop.config import PROJECT

torch.manual_seed(19)
shape = (1, 64, 2, 128)
q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
g = -torch.rand((1, 64, 2), device='cuda')
beta = torch.rand((1, 64, 2), device='cuda', dtype=torch.bfloat16)
started = time.monotonic()
reference, state_reference = torch_chunk_gated_delta_rule(q,k,v,g,beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
actual, state_actual = chunk_gated_delta_rule(q,k,v,g,beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
result = {'max_abs_output_error': (actual-reference).abs().max().item(),
          'relative_output_l2_error': ((actual-reference).float().norm()/reference.float().norm()).item(),
          'relative_state_l2_error': ((state_actual-state_reference).float().norm()/state_reference.float().norm()).item(),
          'first_compile_and_run_seconds': time.monotonic()-started}
assert torch.isfinite(actual).all()
assert result['relative_output_l2_error'] < .02, result
assert result['relative_state_l2_error'] < .02, result
result['status'] = 'passed'
(PROJECT / 'reports/fla_kernel_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

"""MPE preview entry; all Python execution must be inside Docker."""
from runpy import run_module
if __name__=='__main__': run_module('vlm4coop.cli.probes.render_mpe_preview',run_name='__main__')

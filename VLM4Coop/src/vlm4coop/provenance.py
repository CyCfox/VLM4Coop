"""Record exactly the source/config/dependency versions used by each run."""
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import json
import platform
import tarfile
from vlm4coop.config import PROJECT, ROOT


def capture(output):
    output = Path(output)
    paths = sorted(p for name in ('src', 'configs') for p in (PROJECT / name).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts)
    records = {str(p.relative_to(PROJECT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in ('AGENTS.md', 'VLM4Coop_Experiment_Plan.md'):
        p = ROOT / name
        records['../' + name] = sha256(p.read_bytes()).hexdigest()
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'gymnasium', 'lbforaging', 'Pillow',
                 'transformers', 'accelerate', 'smaclite', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (output / 'provenance.json').write_text(json.dumps({"files": records,
        "packages": packages, "python": platform.python_version(),
        "epymarl_commit": "cbc38c09588064eab978501d0f12c2cf58fa7fc2",
        "lvlm2p_commit": "bbad541b029ba3eff941299d3c937312676aed73"}, indent=2))
    with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(PROJECT)))
        archive.add(ROOT / 'VLM4Coop_Experiment_Plan.md', arcname='protocol/Experiment_Plan.md')
        archive.add(PROJECT / 'reports/USER_PROTOCOL_DECISION.md', arcname='protocol/USER_PROTOCOL_DECISION.md')

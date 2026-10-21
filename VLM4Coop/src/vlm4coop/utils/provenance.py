"""Record exactly the source/config/dependency versions used by each run."""
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import json
import platform
import tarfile
from vlm4coop.utils.paths import PROJECT, ROOT


def capture(output):
    output = Path(output)
    paths = sorted(p for name in ('src', 'configs') for p in (PROJECT / name).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts
                   and not p.is_relative_to(PROJECT / 'src/old'))
    records = {str(p.relative_to(PROJECT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol_path=None
    frozen=output/'config.json'
    if frozen.exists():
        protocol=json.loads(frozen.read_text()).get('experiment_config',{}).get('protocol_document')
        if protocol:
            protocol_path=(PROJECT/protocol).resolve()
            if not protocol_path.is_relative_to(PROJECT.resolve()): raise ValueError('Protocol must stay inside VLM4Coop')
            records[str(protocol_path.relative_to(PROJECT.resolve()))]=sha256(protocol_path.read_bytes()).hexdigest()
    for name in ('AGENTS.md', 'VLM4Coop_Experiment_Plan.md'):
        p = ROOT / name
        records['../' + name] = sha256(p.read_bytes()).hexdigest()
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'gymnasium', 'lbforaging', 'Pillow',
                 'transformers', 'accelerate', 'smaclite', 'PyYAML', 'pettingzoo', 'pygame'):
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
        if protocol_path is not None:
            archive.add(protocol_path,arcname='protocol/'+protocol_path.name)

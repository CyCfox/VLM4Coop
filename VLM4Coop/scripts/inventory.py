"""Index reviewable source/config/test/report artifacts (runtime stays out)."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
root = Path('/workspace/my_project/VLM4Coop')
now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
paths = []
for name in ('src', 'configs', 'tests', 'scripts', 'reports', 'references', 'paper'):
    paths.extend(p for p in (root / name).rglob('*') if p.is_file() and '__pycache__' not in str(p))
paths.extend([root / 'README.md', root / 'THIRD_PARTY_NOTICES.md'])
lines = ['# Research Output Manifest', '', '| Timestamp | Skill | File | Stage | Description |',
         '|-----------|-------|------|-------|-------------|']
for p in sorted(paths):
    stage = 'implementation' if p.suffix in {'.py', '.sh', '.yaml', '.txt'} else 'review'
    lines.append(f'| {now} | /research-pipeline | {p.relative_to(root)} | {stage} | {p.stat().st_size} bytes |')
(root / 'MANIFEST.md').write_text('\n'.join(lines) + '\n')

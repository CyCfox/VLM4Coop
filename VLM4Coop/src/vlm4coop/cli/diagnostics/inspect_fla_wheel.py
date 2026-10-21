from pathlib import Path
import zipfile
wheel = next(Path('VLM4Coop/cache/wheels').glob('flash_linear_attention-0.3.2-*.whl'))
with zipfile.ZipFile(wheel) as archive:
    metadata = archive.read(next(name for name in archive.namelist() if name.endswith('METADATA'))).decode()
print('\n'.join(line for line in metadata.splitlines() if line.startswith(('Requires-Dist:', 'Requires-Python:'))))

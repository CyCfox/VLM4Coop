from pathlib import Path
import zipfile
wheel = next(Path('VLM4Coop/cache/wheels').glob('fla_core-0.3.2-*.whl'))
with zipfile.ZipFile(wheel) as archive:
    metadata = archive.read(next(name for name in archive.namelist() if name.endswith('METADATA'))).decode()
    for line in metadata.splitlines():
        if line.startswith(('Requires-Dist:', 'Requires-Python:')):
            print(line)
    print(archive.read('fla/__init__.py').decode()[:2000])

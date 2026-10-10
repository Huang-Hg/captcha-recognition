"""Save the code actually available at experiment startup with its hashes."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import shutil
from .evaluation import dump


def capture_source(destination):
    root = Path(__file__).resolve().parents[2]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    sources = [source for name in ('src', 'scripts') for source in (root/name).rglob('*.py')]
    sources.append(root/'pyproject.toml')
    for source in sorted(sources):
        relative = source.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    dependencies = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python-headless', 'PyYAML'):
        dependencies[name] = importlib.metadata.version(name)
    dump(destination/'manifest.json', dict(role='Automatic source snapshot at experiment startup',
        python=platform.python_version(), dependencies=dependencies, files=files))

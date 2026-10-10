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
    distribution = importlib.metadata.distribution('captcha-recognition')
    if (root/'pyproject.toml').is_file() and Path(__file__).resolve().parent.parent.name == 'src':
        sources = [source for name in ('src', 'scripts') for source in (root/name).rglob('*.py')]
        sources.extend((root/'src/captcha_recognition/backbones').glob('LICENSE'))
        sources.extend((root/'src/captcha_recognition/backbones').glob('NOTICE'))
        sources.append(root/'pyproject.toml')
        entries = [(source.relative_to(root), source) for source in sources]
    else:
        entries = []
        for relative in distribution.files or ():
            path = Path(relative)
            if '..' in path.parts or path.is_absolute():
                continue
            if path.suffix == '.py' or (path.parent.as_posix() == 'captcha_recognition/backbones'
                                      and path.name in ('LICENSE', 'NOTICE')):
                entries.append((Path('src')/path, Path(distribution.locate_file(relative))))
        if not entries:
            raise RuntimeError('Installed distribution does not expose source files for provenance')
    for relative, source in sorted(entries):
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    dependencies = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python-headless'):
        dependencies[name] = importlib.metadata.version(name)
    dump(destination/'manifest.json', dict(role='Automatic source snapshot at experiment startup',
        python=platform.python_version(), package_version=distribution.version,
        dependencies=dependencies, files=files))

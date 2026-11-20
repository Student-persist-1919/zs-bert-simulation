"""Read immutable project dataset snapshots; verify compressed and raw bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_snapshot(root, dataset, destination):
    base = Path(root) / 'resources/datasets'
    metadata = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))[dataset]
    archive = base / metadata['filename']
    if digest(archive) != metadata['archive_sha256']:
        raise ValueError(f'Corrupt dataset archive: {archive}')
    destination = Path(destination)
    if destination.exists() and digest(destination) == metadata['raw_sha256']:
        return metadata
    temporary = destination.with_suffix('.download.tmp')
    try:
        with gzip.open(archive, 'rb') as source, open(temporary, 'wb') as target:
            shutil.copyfileobj(source, target)
        if temporary.stat().st_size != metadata['raw_bytes'] or digest(temporary) != metadata['raw_sha256']:
            raise ValueError(f'Decompressed dataset checksum mismatch: {dataset}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata

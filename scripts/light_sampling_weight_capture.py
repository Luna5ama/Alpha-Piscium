"""Verify and preserve a completed Vibris action job before artifact deletion."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('I:/code/mcshaders/Alpha-Piscium-4')
RUN = ROOT / 'builds/light-sampling/20261031-weights'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preserve(job_file, destination, revision, preset):
    job = json.loads(job_file.read_text(encoding='utf-8-sig'))
    assert job['success']
    result = job['result']
    provenance = result['provenance']
    assert provenance['resolved_revision'] == revision
    assert provenance['preset_id'] == preset
    assert not provenance['stale'] and not provenance['head_changed']
    assert not provenance.get('dirty_shader_delta_sha256')
    assert all(r['status'] == 'RECEIPT_STATUS_OK' for r in result['action_receipts'])
    assert not destination.exists(), destination
    destination.mkdir(parents=True)
    entries, names = [], set()
    for artifact in result['artifacts']:
        source = Path(artifact['relative_path'])
        if not source.is_absolute():
            source = ROOT / source
        name = source.name
        assert name not in names, name
        names.add(name)
        target = destination / name
        shutil.copy2(source, target)
        actual = digest(target)
        assert actual == artifact['sha256'] and digest(source) == actual, source
        entries.append(dict(original=str(source), relative_path=name, byte_size=target.stat().st_size, sha256=actual))
    shutil.copy2(job_file, destination / 'job-result.json')
    manifest = dict(revision=revision, preset=preset, files=entries)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--preset', required=True)
    args = parser.parse_args()
    ready = json.loads((RUN / 'source-ready.json').read_text(encoding='utf-8-sig'))
    assert args.revision in ready.values(), 'Revision must match source-ready.json'
    assert args.destination.resolve().is_relative_to(RUN.resolve())
    print(json.dumps(preserve(args.job, args.destination, args.revision, args.preset), indent=2))

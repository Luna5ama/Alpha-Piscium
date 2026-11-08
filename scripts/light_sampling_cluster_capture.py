"""Archive and validate fresh mode-3 point windows for the cluster summary experiment."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from light_sampling_moving_analysis import leaves, native_node
from light_sampling_point_moments import ROOT, POINTS, capture_request, extract, input_identity, normal_bits, read, receiver_camera, sha
from light_sampling_point_screening import tree_variants
from light_sampling_weight_reference import load_tree

RUN = ROOT / 'builds/light-sampling/20261108-cluster-summaries'
GROUPS = {
    'sides': ('light-sampling-failure-moving-4', '1b9b62c599cb373e0740f162a3748a3b65346472'),
    'side-top': ('light-sampling-failure-moving-4', 'f06dcbe3026c3a15cc8896dc75f4af553b414c76'),
    'ls1': ('light-sampling-1', 'c091fc402f91f457037819fd1b88b657a1646687'),
}
SETTINGS = dict(
    DEBUG_LIGHT_ARITY='4', DEBUG_LIGHT_AUDIT='true', DEBUG_LIGHT_CUBE='true',
    DEBUG_LIGHT_WEIGHT_MODEL='1', DENOISER_ACCUM='false', DENOISER_ANTI_FIREFLY='false',
    DENOISER_SPATIAL='false', GI_DECORRELATE='false', GI_PATH_GUIDING='false',
    GI_SPATIAL_REUSE='false', GI_USE_REFERENCE='0', NORMAL_MAPPING='false',
    PBR_MATERIAL='0', RC_ENABLE='false', TAA='false', TAA_JITTER='false', VOXEL_GRID_SIZE='16',
)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def request(group, revision=None):
    preset, default_revision = GROUPS[group]
    revision = revision or default_revision
    job = capture_request(revision, {'SETTING_' + key: value for key, value in SETTINGS.items()}, preset)
    job['restore_state'] = dict(on_error=True, on_success=True)
    return job


def fixture_for(group):
    if group == 'ls1':
        data = read(ROOT / 'experiments/light-sampling/20261031-ls1-point-fixture.json')
        for row, point in zip(data['receivers'], ('ls1-pillar', 'ls1-wall')):
            row['id'] = point
        data['light_count'] = data['leaf_count']
        return data
    points = POINTS[:2] if group == 'sides' else POINTS[1:]
    texture = f'shaders/textures/light_sampling_moving4_{group}.bin'
    return dict(texture=texture, texture_sha256=sha(ROOT / texture),
                receivers=points, light_count=(ROOT / texture).stat().st_size // 16 - 4)


def archive(group, job_id, revision=None):
    job = ROOT / '.vibris/jobs' / job_id
    public = read(job / 'request.json')['arguments']
    result = read(job / 'result.json')
    native = result['result']
    provenance = native['provenance']
    preset, default_revision = GROUPS[group]
    revision = revision or default_revision
    assert result['success'] and provenance['preset_id'] == preset and provenance['resolved_revision'] == revision
    assert not any(provenance[key] for key in ('head_changed', 'stale', 'dirty_shader_delta_sha256', 'patch_sha256'))
    assert len(native['action_receipts']) == len(public['actions'])
    assert all(row['status'] == 'RECEIPT_STATUS_OK' for row in native['action_receipts'])
    settings = {row['name']: row['value'] for row in provenance['effective_settings']['settings']}
    for key, value in request(group)['configs'][0]['values'].items():
        assert settings[key] == value, (key, settings[key], value)
    manifest_ref = next(row for row in native['artifacts'] if row['kind'] == 'ARTIFACT_KIND_MANIFEST')
    manifest = Path(manifest_ref['relative_path'])
    assert sha(manifest) == manifest_ref['sha256']
    metadata = read(manifest)
    assert metadata['manifest_id'] == native['result_manifest_id']
    out = RUN / 'captures' / group
    out.mkdir(parents=True, exist_ok=True)
    for record in metadata['files']:
        filename = Path(record['relative_path'])
        assert not filename.is_absolute() and '..' not in filename.parts
        destination = out / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha(destination) != record['sha256']:
            shutil.copyfile(manifest.parent / filename, destination)
        assert destination.stat().st_size == record['byte_size'] and sha(destination) == record['sha256'], filename
    shutil.copyfile(manifest, out / 'manifest.json')
    for name in ('request.json', 'result.json', 'state.json', 'events.jsonl'):
        shutil.copyfile(job / name, out / ('vibris-' + name))
    return dict(folder=out.relative_to(ROOT).as_posix(), job_id=job_id, manifest_id=metadata['manifest_id'],
                manifest_sha256=manifest_ref['sha256'], file_count=len(metadata['files']),
                preset_id=preset, revision=revision, provenance=provenance)


def ingest(group, job_id, revision=None, fixture_path=None):
    evidence = archive(group, job_id, revision)
    out = ROOT / evidence['folder']
    fixture = read(fixture_path) if fixture_path else fixture_for(group)
    assert fixture['texture_sha256'] == sha(ROOT / fixture['texture'])
    for alias in ('tree', 'camera', 'voxel-meta', 'voxel-materials'):
        for ext in ('.bin', '.json'):
            path = out / (alias + ext)
            if not path.exists():
                os.link(out / ('before_' + alias + ext), path)
    camera = receiver_camera(out, fixture)
    context = read(out / 'vibris-request.json')['arguments']['__vibris_scene_context']
    assert context['resolution'] == dict(width=1920, height=1080)
    phases = [input_identity(out, 16, camera['camera_brick'], phase, context)
              for phase in ('before', 'between', 'after')]
    assert phases[0][0] == phases[1][0] == phases[2][0], phases
    assert all(stats['trace_exhausted'] == 0 for _, stats in phases), phases
    fixture_records = np.fromfile(ROOT / fixture['texture'], '<i4').reshape(-1, 4)[4:]
    trees, _ = tree_variants(out, fixture, 'world-prefix')
    prefix = trees['world-prefix']
    assert len(fixture_records) == fixture['light_count'] == len(prefix.keys)
    assert set(map(tuple, fixture_records)) == set(prefix.keys)
    first = np.fromfile(out / 'audit.bin', '<f4')[:32].reshape(2, 4, 4)
    gbuffer = np.memmap(out / 'colortex8.bin', '<u4', mode='r', shape=(1080, 1920, 4))
    rgb = np.memmap(out / 'colortex9.bin', 'u1', mode='r', shape=(1080, 1920, 4))
    depth = np.memmap(out / 'colortex10.bin', '<f4', mode='r', shape=(1080, 1920))
    for point, receiver in enumerate(fixture['receivers']):
        x, y = receiver['raw_xy']
        assert np.allclose(first[point, 0, :3], receiver['origin'], rtol=0, atol=.005)
        assert first[point, 1, :3] @ receiver['normal'] > .999
        assert int(gbuffer[y, x, 3] >> 16) == receiver.get('material', receiver.get('material_id'))
        if 'rgb_bytes' in receiver:
            assert rgb[y, x, :3].tolist() == receiver['rgb_bytes']
        if 'view_z' in receiver:
            assert abs(depth[y, x] - receiver['view_z']) < .005
    for phase in ('between', 'after'):
        later = np.fromfile(out / f'audit_{phase}.bin', '<f4')[:32].reshape(2, 4, 4)
        np.testing.assert_array_equal(first, later)
    moments = RUN / 'moments' / group
    summary = extract(out, fixture, moments)
    assert all(not row['missing_pixel_groups'] and not row['nonfinite_rgb_samples'] and
               not row['unavailable_lights'] for row in summary)
    evidence.update(camera=camera, identity=phases[0][0], phase_stats=[row[1] for row in phases],
                    fixture=fixture, context=context, summary=summary,
                    native_capacity=len(load_tree(out / 'tree.bin', 1)[1]))
    write(out / 'accepted.json', evidence)
    return {k: evidence[k] for k in ('folder', 'manifest_id', 'manifest_sha256', 'file_count', 'identity')}


def rebind_ls1(source, texture):
    """Keep the exact two existing pixels; rebuild only the emitter table."""
    original = fixture_for('ls1')
    audit = np.fromfile(source / 'audit.bin', '<f4')[:32].reshape(2, 4, 4)
    header, nodes, links = load_tree(source / 'tree.bin', 1)
    records = sorted(leaves(native_node(nodes, links, 1)))
    assert len(records) == int(header[1]) and not header[2]
    output = np.empty((len(records) + 4, 4), dtype='<i4')
    output[4:] = [row[:4] for row in records]
    for i, receiver in enumerate(original['receivers']):
        x, y = receiver['raw_xy']
        assert audit[i, 3, :2].tolist() == [x, y]
        assert int(audit[i, 0, 3]) == receiver['material_id']
        assert np.linalg.norm(audit[i, 0, :3] - receiver['origin']) < .005
        assert audit[i, 1, :3] @ receiver['normal'] > .999
        output[i * 2] = [x, y, receiver['material_id'], len(records)]
        output[i * 2 + 1, :3] = audit[i, 0, :3].view('<i4')
        output.view('<u4')[i * 2 + 1, 3] = normal_bits(audit[i, 1, :3])
    texture.write_bytes(output.tobytes())
    data = dict(texture=texture.resolve().relative_to(ROOT).as_posix(), texture_sha256=sha(texture),
                source_tree_sha256=sha(source / 'tree.bin'), light_count=len(records),
                receivers=original['receivers'])
    write(texture.with_suffix('.json'), data)
    return dict(texture=data['texture'], fixture=str(texture.with_suffix('.json')), width=len(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('request', 'ingest', 'rebind'))
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--job-id')
    parser.add_argument('--revision')
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--rejected', type=Path)
    parser.add_argument('--texture', type=Path)
    args = parser.parse_args()
    if args.operation == 'request':
        result = request(args.group, args.revision)
    elif args.operation == 'ingest':
        result = ingest(args.group, args.job_id, args.revision, args.fixture)
    else:
        assert args.group == 'ls1'
        result = rebind_ls1(args.rejected, args.texture)
    print(json.dumps(result))

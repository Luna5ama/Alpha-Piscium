"""Two-receiver mode-3 fixtures and raw conditional RGB moments (never image masks)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_moving_analysis import leaves, native_node, ordered_signature, world_voxel_identity, morton_decode, morton_encode
from light_sampling_weight_reference import load_tree
from light_sampling_weight_capture import bracketed_quality_request

ROOT = Path(__file__).resolve().parents[1]
POINTS = [
    dict(id='side-26', slot=26, raw_xy=[600, 472], origin=[109.156212, 76.138733, 362.002991],
         normal=[0, 0, 1], material=4651, rgb_bytes=[121, 85, 58]),
    dict(id='side-27', slot=27, raw_xy=[840, 472], origin=[112.391624, 76.430634, 364.002991],
         normal=[0, 0, 1], material=4651, rgb_bytes=[89, 61, 41]),
    dict(id='top-28', slot=28, raw_xy=[1080, 472], origin=[115.779617, 76.003006, 365.398621],
         normal=[0, 1, 0], material=4651, rgb_bytes=[57, 113, 36]),
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normal_bits(normal):
    vector = np.asarray(normal, float)
    vector /= np.abs(vector).sum()
    xy = vector[:2].copy()
    if vector[2] < 0:
        xy = (1 - np.abs(xy[::-1])) * np.where(xy >= 0, 1, -1)
    snorm = np.rint(np.clip(xy, -1, 1) * 32767).astype('<i2').view('<u2')
    return int(snorm[0]) | (int(snorm[1]) << 16)


def prepare(pilot, output):
    header, nodes, links = load_tree(pilot / 'tree.bin', 1)
    leaf_list = sorted(leaves(native_node(nodes, links, 1)))
    assert len(leaf_list) == int(header[1]) and header[2] == 0
    assert len(set(leaf_list)) == len(leaf_list)
    records = np.asarray([row[:4] for row in leaf_list], dtype='<i4')
    masses = np.asarray([row[4] for row in leaf_list], dtype='<f4')
    meta = read(pilot / 'colortex8.json')
    width, height = meta['width'], meta['height']
    assert (width, height) == (1920, 1080)
    gbuffer = np.memmap(pilot / 'colortex8.bin', '<u4', mode='r', shape=(height, width, 4))
    rgb = np.memmap(pilot / 'colortex9.bin', 'u1', mode='r', shape=(height, width, 4))
    audit = np.fromfile(pilot / 'audit.bin', '<f4')[:1024].reshape(64, 4, 4)
    receivers = []
    for expected in POINTS:
        point = audit[expected['slot']]
        x, y = expected['raw_xy']
        assert np.linalg.norm(point[0, :3] - expected['origin']) < .005
        assert point[1, :3] @ expected['normal'] > .999
        assert int(gbuffer[y, x, 3] >> 16) == expected['material']
        assert rgb[y, x, :3].tolist() == expected['rgb_bytes']
        receivers.append(dict(**expected, origin_actual=point[0, :3].tolist(), normal_actual=point[1, :3].tolist()))
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for name, indices in [('sides', (0, 1)), ('side-top', (1, 2))]:
        fixture = np.zeros((len(records) + 4, 4), dtype='<i4')
        chosen = [receivers[i] for i in indices]
        for i, receiver in enumerate(chosen):
            fixture[2 * i] = [*receiver['raw_xy'], receiver['material'], len(records)]
            fixture[2 * i + 1, :3] = np.asarray(receiver['origin_actual'], '<f4').view('<i4')
            fixture.view('<u4')[2 * i + 1, 3] = normal_bits(receiver['normal_actual'])
        fixture[4:] = records
        path = ROOT / f'shaders/textures/light_sampling_moving4_{name}.bin'
        path.write_bytes(fixture.tobytes())
        group = dict(id=name, texture=path.relative_to(ROOT).as_posix(), texture_sha256=sha(path),
                     width=len(fixture), light_count=len(records), receivers=chosen,
                     source_tree=str(pilot / 'tree.bin'), source_tree_sha256=sha(pilot / 'tree.bin'),
                     leaf_identity_sha256=hashlib.sha256(records.tobytes() + masses.tobytes()).hexdigest(),
                     samples_per_light_per_frame=width * height // (2 * len(records)))
        np.savez_compressed(output / f'{name}-leaves.npz', records=records, masses=masses)
        (output / f'{name}.json').write_text(json.dumps(group, indent=2) + '\n', encoding='utf-8')
        groups.append(group)
    (output / 'groups.json').write_text(json.dumps(groups, indent=2) + '\n', encoding='utf-8')
    return dict(groups=[g['id'] for g in groups], light_count=len(records))


def input_identity(folder, grid, camera_brick, phase, context):
    """Trace exhaustion is an output defect, not an emitter/occluder identity change."""
    header, nodes, links = load_tree(folder / f'{phase}_tree.bin', 1)
    root = native_node(nodes, links, 1)
    light_list = sorted(leaves(root))
    assert len(light_list) == int(header[1]) and header[2] == 0
    camera = np.fromfile(folder / f'{phase}_camera.bin', '<u4')
    voxels = world_voxel_identity(folder, grid, camera_brick, prefix=phase + '_', context=context)
    return dict(camera_sha256=hashlib.sha256(np.r_[camera[16:32], camera[80:112]].tobytes()).hexdigest(),
                leaf_sha256=hashlib.sha256(np.asarray(light_list, '<f8').tobytes()).hexdigest(),
                ordered_topology_sha256=ordered_signature(root),
                world_voxel_sha256=voxels['world_blocks_material_sha256']), dict(
                    trace_exhausted=int(header[3]), emitters=int(header[1]), allocated_bricks=voxels['allocated_bricks'])


def capture_request(revision, values, preset='light-sampling-failure-moving-4'):
    values = dict(values, SETTING_DEBUG_LIGHT_SAMPLING='3')
    request = bracketed_quality_request(revision, preset, values)
    actions = []
    for action in request['actions']:
        if action['type'] == 'accumulate_texture':
            actions.append(dict(action, artifact_name='window1'))
            for name, logical in [('camera', 'iris_ssbo_0'), ('tree', 'iris_ssbo_12'),
                                  ('voxel-meta', 'iris_ssbo_3'), ('voxel-materials', 'iris_ssbo_4')]:
                actions.append(dict(type='dump_buffer', resource=dict(logical_name=logical), artifact_name=f'between_{name}'))
            actions.append(dict(type='dump_buffer', resource=dict(logical_name='iris_ssbo_14'), artifact_name='audit_between'))
            actions.append(dict(action, artifact_name='window2'))
        else:
            actions.append(action)
    actions.append(dict(type='dump_buffer', resource=dict(logical_name='iris_ssbo_14'), artifact_name='audit_after'))
    request['actions'] = actions
    return request


def receiver_camera(folder, fixture):
    """Infer the preset camera from two validated mode-3 receiver origins and primary depth."""
    camera_path = folder / 'iris_ssbo_0.bin'
    depth_path = folder / 'colortex10.bin'
    if not camera_path.exists():
        camera_path = folder / 'camera.bin'
    if not depth_path.exists():
        depth_path = folder / 'depth.bin'
    metadata = read(depth_path.with_suffix('.json'))
    width, height = metadata['width'], metadata['height']
    data = np.fromfile(camera_path, '<f4')
    view_inverse = data[16:32].reshape(4, 4).T.astype(float)
    projection_inverse = data[96:112].reshape(4, 4).T.astype(float)
    depth = np.memmap(depth_path, '<f4', mode='r', shape=(height, width))
    audit = np.fromfile(folder / 'audit.bin', '<f4')[:32].reshape(2, 4, 4).astype(float)
    estimates = []
    for i, receiver in enumerate(fixture['receivers']):
        assert audit[i, 2, 3] == 1
        x, y = receiver['raw_xy']
        z = float(depth[y, x])
        view = [((x + .5) / width * 2 - 1) * -z * projection_inverse[0, 0],
                ((y + .5) / height * 2 - 1) * -z * projection_inverse[1, 1], z, 1]
        estimates.append(audit[i, 0, :3] - (view_inverse @ view)[:3] - .003 * audit[i, 1, :3])
    spread = np.ptp(estimates, axis=0)
    assert spread.max() < .002
    position = np.median(estimates, axis=0)
    assert np.min(np.abs(position - np.rint(position))) > max(float(spread.max()), 2e-5)
    return dict(position=position.tolist(), camera_brick=(np.floor(position).astype(np.int64) >> 4).tolist(),
                max_axis_spread=spread.tolist(), method='Two validated mode3 origins and same-frame primary depth')


def nonemissive_identity(folder, camera_brick, grid=16):
    """Canonical world bricks after removing exactly the root-reachable emissive voxels."""
    header, nodes, links = load_tree(folder / 'tree.bin', 1)
    light_list = leaves(native_node(nodes, links, 1))
    assert len(light_list) == int(header[1]) and header[2] == 0
    by_brick = {}
    for light in light_list:
        position = np.asarray(light[:3], dtype=np.int64)
        by_brick.setdefault(tuple(position >> 4), []).append((int(morton_encode(position & 15)), int(light[3])))
    meta = np.fromfile(folder / 'voxel-meta.bin', '<u4')
    count, slots = int(meta[0]), grid ** 3
    allocated = meta[1 + 2 * slots + 1024:1 + 2 * slots + 1024 + count]
    material = np.memmap(folder / 'voxel-materials.bin', '<u4', mode='r')
    origin = np.asarray(camera_brick) - grid // 2
    rows, removed = [], 0
    for allocation, key in enumerate(allocated):
        brick = tuple(origin + morton_decode(key))
        values = np.array(material[allocation * 4096:(allocation + 1) * 4096])
        for local, material_id in by_brick.get(brick, []):
            assert int(values[local]) >> 1 == material_id
            values[local] = 0
            removed += 1
        nonempty = int(np.count_nonzero(values))
        if nonempty:
            rows.append((brick, values, nonempty))
    assert removed == len(light_list)
    digest = hashlib.sha256()
    for brick, values, _ in sorted(rows, key=lambda row: row[0]):
        digest.update(np.asarray(brick, '<i8').tobytes())
        digest.update(values.astype('<u4', copy=False).tobytes())
    return dict(sha256=digest.hexdigest(), removed_emitters=removed,
                nonempty_remaining_voxels=sum(row[2] for row in rows), nonempty_remaining_bricks=len(rows),
                definition='Sorted signed world-brick xyz int64 plus all4096 packed uint32 material words; root-reachable emitter words zeroed, empty bricks omitted.')


def extract(folder, fixture, output, prefix='window', points=None):
    """Equal 32-sample complete pixel groups; keep zero samples and report missingness."""
    light_count = int(fixture['light_count'])
    summary = []
    audit_path = folder / 'audit.bin'
    audit = np.fromfile(audit_path, '<f4')[:32].reshape(2, 4, 4).astype(float)
    selected = np.arange(len(fixture['receivers'])) if points is None else np.asarray(points,dtype=int)
    assert len(selected) and len(set(selected))==len(selected) and ((0<=selected)&(selected<len(audit))).all()
    for point in selected:
        receiver = fixture['receivers'][point]
        assert audit[point, 2, 3] == 1, 'Mode3 receiver invalid'
        assert np.linalg.norm(audit[point, 0, :3] - receiver.get('origin_actual', receiver.get('origin'))) < .005
        assert audit[point, 1, :3] @ receiver['normal'] > .999
        assert int(audit[point, 0, 3]) == receiver.get('material', receiver.get('material_id'))
        assert audit[point, 3, :2].tolist() == receiver['raw_xy']
        assert int(audit[point, 3, 2]) == light_count and audit[point, 3, 3] == 3
    for path in (folder / 'audit_between.bin', folder / 'audit_after.bin', folder / 'audit-after.bin'):
        if not path.exists():
            continue
        later = np.fromfile(path, '<f4')[:32].reshape(2, 4, 4).astype(float)[selected]
        reference = audit[selected]
        assert np.all(later[:, 2, 3] == 1)
        assert np.all(later[:, 0, 3] == reference[:, 0, 3])
        assert np.allclose(later[:, 0, :3], reference[:, 0, :3], rtol=0, atol=.005)
        assert np.all(np.sum(later[:, 1, :3] * reference[:, 1, :3], axis=1) > .999)
        assert np.array_equal(later[:, 1, 3], reference[:, 1, 3])
        assert np.allclose(later[:, 2, :3], reference[:, 2, :3], rtol=1e-6, atol=1e-7)
        assert np.array_equal(later[:, 3], reference[:, 3])
    previous_end = -1
    output.mkdir(parents=True, exist_ok=True)
    for window in (1, 2):
        name = f'{prefix}{window}'
        metadata = read(folder / f'{name}.mean.json')
        width, height = int(metadata['width']), int(metadata['height'])
        samples = width * height // (2 * light_count)
        assert np.all(audit[:, 1, 3] == samples)
        accumulation = metadata['accumulation']
        assert accumulation['frames'] == 32 and accumulation['batches'] == 8
        assert accumulation['batch_lengths'] == [4] * 8
        assert accumulation['end_frame'] - accumulation['start_frame'] == 31
        assert accumulation['start_frame'] > previous_end
        previous_end = accumulation['end_frame']
        def image(kind):
            return np.memmap(folder / f'{name}.{kind}.bin', '<f4', mode='r', shape=(width * height, 4))[
                :samples * 2 * light_count].reshape(samples, 2, light_count, 4)
        mean, variance = image('mean'), image('variance')
        counts, nonfinite = image('valid_count'), image('nonfinite')
        incomplete, valid_batches = image('incomplete_batches'), image('valid_batches')
        for point in selected:
            receiver = fixture['receivers'][point]
            means, variances = mean[:, point], variance[:, point]
            good = np.isfinite(means).all(axis=2) & np.isfinite(variances).all(axis=2)
            good &= (counts[:, point] == 32).all(axis=2) & (nonfinite[:, point] == 0).all(axis=2)
            good &= (incomplete[:, point] == 0).all(axis=2) & (valid_batches[:, point] == 8).all(axis=2)
            mu_squared = np.sum(means[:, :, :3].astype(float) ** 2, axis=2)
            variance_sum = np.sum(variances[:, :, :3], axis=2, dtype=float)
            residual = means[:, :, 3] - mu_squared - (31 / 32) * variance_sum
            scale = np.maximum(np.abs(means[:, :, 3]) + mu_squared + np.abs(variance_sum), 1e-20)
            error = float(np.max(np.abs(residual[good]) / scale[good])) if good.any() else None
            assert error is None or error < 1e-4, error
            complete = good.sum(axis=0)
            moments = np.divide(np.where(good[:, :, None], means, 0).sum(axis=0, dtype=float),
                                complete[:, None], out=np.full((light_count, 4), np.nan), where=complete[:, None] > 0)
            destination = output / f"{receiver['id']}-{name}.npz"
            np.savez_compressed(destination, mean_rgb=moments[:, :3], m2=moments[:, 3],
                                complete_pixel_groups=complete, complete_samples=complete * 32,
                                potential_samples=np.full(light_count, samples * 32),
                                valid_samples=counts[:, point].sum(axis=0, dtype=float),
                                nonfinite_samples=nonfinite[:, point].sum(axis=0, dtype=float),
                                incomplete_pixel_groups=samples - complete)
            summary.append(dict(point=receiver['id'], window=window, path=str(destination),
                                light_count=light_count, samples_per_frame=samples,
                                complete_pixel_groups=int(complete.sum()), missing_pixel_groups=int(good.size - good.sum()),
                                unavailable_lights=np.flatnonzero(complete == 0).tolist(),
                                positive_m2_lights=int(np.count_nonzero(moments[:, 3] > 0)),
                                nonfinite_rgb_samples=int(nonfinite[:, point, :, :3].sum()),
                                second_moment_identity_max_relative_error=error,
                                accumulation=accumulation, receiver_audit=audit[point].tolist()))
    (output / 'moments.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'extract'])
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.folder, args.output) if args.operation == 'prepare' else extract(args.folder, read(args.fixture), args.output)
    print(json.dumps(result, allow_nan=False))

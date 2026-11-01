"""World-space identities and full proposal diagnostics for the moving-pose experiment."""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from light_sampling_weight_reference import audit_capture, load_tree, probabilities

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'builds/light-sampling/20261031-moving-stability'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def morton_encode(xyz):
    xyz = np.asarray(xyz, dtype=np.int64)
    result = np.zeros(xyz.shape[:-1], dtype=np.int64)
    for bit in range(10):
        for axis in range(3):
            result |= ((xyz[..., axis] >> bit) & 1) << (bit * 3 + axis)
    return result


def morton_decode(values):
    values = np.asarray(values, dtype=np.int64)
    result = np.zeros(values.shape + (3,), dtype=np.int64)
    for bit in range(10):
        for axis in range(3):
            result[..., axis] |= ((values >> (bit * 3 + axis)) & 1) << bit
    return result


def camera_from_audit(folder):
    meta = read(folder / 'single.json')
    width, height = meta['width'], meta['height']
    globals_ = np.fromfile(folder / 'iris_ssbo_0.bin', '<f4')
    view_inverse = globals_[16:32].reshape(4, 4).T.astype(float)
    projection_inverse = globals_[96:112].reshape(4, 4).T.astype(float)
    depth = np.memmap(folder / 'colortex10.bin', '<f4', mode='r', shape=(height, width))
    geometry = np.memmap(folder / 'colortex8.bin', '<u4', mode='r', shape=(height, width, 4))
    albedo = np.memmap(folder / 'colortex9.bin', 'u1', mode='r', shape=(height, width, 4))
    audit = np.fromfile(folder / 'audit.bin', '<f4')[:1024].reshape(64, 4, 4).astype(float)
    estimates, points = [], []
    for slot, record in enumerate(audit):
        if record[3, 3] == 0:
            continue
        x, y = (2 * (slot % 8) + 1) * width // 16, (2 * (slot // 8) + 1) * height // 16
        z = float(depth[y, x])
        view = [((x + .5) / width * 2 - 1) * -z * projection_inverse[0, 0],
                ((y + .5) / height * 2 - 1) * -z * projection_inverse[1, 1], z, 1]
        estimates.append(record[0, :3] - (view_inverse @ view)[:3] - .003 * record[1, :3])
        points.append(dict(slot=slot, pixel=[x, y], origin=record[0, :3].tolist(), normal=record[1, :3].tolist(),
                           material=int(geometry[y, x, 3] >> 16), albedo=albedo[y, x, :3].tolist()))
    estimates = np.asarray(estimates)
    assert len(estimates) >= 3, 'Need receiving audit points to reconstruct the actual camera'
    spread = np.ptp(estimates, axis=0)
    assert spread.max() < .002, spread
    position = np.median(estimates, axis=0)
    # Iris CameraUniforms floors the unshifted camera, including negative positions.
    # A value within the reconstruction uncertainty of an integer needs external pose evidence.
    uncertainty = max(float(spread.max()), 2e-5)
    assert np.min(np.abs(position - np.rint(position))) > uncertainty, position
    integer = np.floor(position).astype(np.int64)
    return dict(position=position.tolist(), integer=integer.tolist(), camera_brick=(integer >> 4).tolist(),
                method='64 GPU audit origins minus same-frame unprojected view positions; static previous view inverse',
                max_axis_spread=spread.tolist(), points=points,
                view_inverse=view_inverse.tolist(), projection_inverse=projection_inverse.tolist())


def world_voxel_identity(folder, grid, camera_brick):
    """Canonical signed world blocks plus the complete packed uint material word."""
    meta = np.fromfile(folder / 'voxel-meta.bin', '<u4')
    material = np.memmap(folder / 'voxel-materials.bin', '<u4', mode='r')
    count, slots = int(meta[0]), grid ** 3
    bricks = meta[1 + 2 * slots + 1024:1 + 2 * slots + 1024 + count]
    assert len(bricks) == count and len(np.unique(bricks)) == count
    assert count * 4096 <= len(material)
    window_origin = np.asarray(camera_brick) - grid // 2
    local = morton_decode(np.arange(4096))
    parts, brick_records = [], []
    for allocation, key in enumerate(bricks):
        assert key < slots and meta[1 + slots + int(key)] == allocation
        world_brick = window_origin + morton_decode(key)
        values = material[allocation * 4096:(allocation + 1) * 4096]
        present = np.flatnonzero(values)
        if not len(present):
            continue
        packed = np.empty((len(present), 4), dtype='<i8')
        packed[:, :3] = world_brick * 16 + local[present]
        packed[:, 3] = values[present]
        parts.append(packed)
        brick_records.append(dict(world_brick=world_brick.tolist(), nonempty=len(present),
                                  packed_material_sha256=digest(values.tobytes())))
    records = np.concatenate(parts) if parts else np.empty((0, 4), dtype='<i8')
    order = np.lexsort((records[:, 2], records[:, 1], records[:, 0]))
    return dict(grid=grid, camera_brick=list(camera_brick), window_brick_origin=window_origin.tolist(),
                allocated_bricks=count, nonempty_voxels=len(records),
                world_blocks_material_sha256=digest(records[order].tobytes()),
                bricks=sorted(brick_records, key=lambda row: row['world_brick']))


@dataclass
class Node:
    data: np.ndarray
    material: int | None
    children: tuple


def native_node(nodes, links, index):
    if nodes[index, 3] == 0:
        return None
    link = int(links[index])
    if link & 0x80000000:
        return Node(nodes[index, :7].copy(), link & 0x7fffffff, ())
    base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
    children = tuple(filter(None, (native_node(nodes, links, i) for i in range(base, base + count))))
    return Node(nodes[index, :7].copy(), None, children)


def leaf_key(node):
    return (*map(float, node.data[:3]), node.material, float(node.data[3]))


def leaves(node):
    if node is None:
        return []
    if node.material is not None:
        return [leaf_key(node)]
    return [key for child in node.children for key in leaves(child)]


def signature(node):
    if node is None:
        return digest(b'empty')
    if node.material is not None:
        return digest(b'L' + np.asarray(leaf_key(node), '<f8').tobytes())
    return digest(b'N' + b''.join(sorted(bytes.fromhex(signature(child)) for child in node.children)))


def ordered_signature(node):
    if node is None or node.material is not None:
        return signature(node)
    return digest(b'N' + b''.join(bytes.fromhex(ordered_signature(child)) for child in node.children))


def combine_world_children(children):
    return children[0] if len(children) == 1 else merged(children) if children else None


def captured_brick_roots(folder, nodes, links, grid, camera_brick, layout, arity=4):
    if layout == 'world-prefix':
        from light_sampling_world_layout import WorldLayout
        storage = WorldLayout(grid, arity, camera_brick)
        meta = np.fromfile(folder / 'voxel-meta.bin', '<u4')
        keys = meta[1 + 2 * grid ** 3 + 1024:1 + 2 * grid ** 3 + 1024 + int(meta[0])]
        origin = np.asarray(camera_brick) - grid // 2
        bricks = origin + morton_decode(keys)
        entries = ((tuple(map(int, brick)), storage.brick_address(brick)) for brick in bricks)
    else:
        assert layout in ('camera-relative', 'periodic'), layout
        entries = ((tuple((nodes[i, :3].astype(np.int64) >> 4).tolist()), i)
                   for i in range(grid ** 3, 2 * grid ** 3) if nodes[i, 3] > 0)
    roots = {}
    for brick, index in entries:
        if nodes[index, 3] <= 0:
            continue
        assert np.all(nodes[index, :3] >= np.asarray(brick) * 16)
        assert np.all(nodes[index, 4:7] <= (np.asarray(brick) + 1) * 16)
        assert brick not in roots
        roots[brick] = native_node(nodes, links, index)
    return roots


def rebuild_grouping(bricks, grid, camera_brick, layout, arity=4):
    if layout == 'world-prefix':
        from light_sampling_world_layout import WorldLayout, oracle
        result, _ = WorldLayout(grid, arity, camera_brick).build(bricks, combine_world_children)
        reference = oracle(bricks, combine_world_children, arity)
        assert ordered_signature(result) == ordered_signature(reference)
        return result
    assert layout in ('camera-relative', 'periodic'), layout
    return rebuild_outer(bricks, grid, camera_brick, layout == 'periodic', arity)


def full_pmf(node, origin, normal):
    result, zero = {}, 0.
    pending = [(node, 1.)]
    while pending:
        current, p = pending.pop()
        if current is None:
            zero += p
        elif current.material is not None:
            result[leaf_key(current)] = p
        else:
            weights = probabilities(np.stack([c.data for c in current.children]), 1, origin, normal)
            if weights.sum() == 0:
                zero += p
            else:
                pending.extend((child, p * float(q)) for child, q in zip(current.children, weights) if q > 0)
    assert abs(sum(result.values()) + zero - 1) < 1e-9
    return result, zero


def compare_pmf(a, b):
    pa, za = a
    pb, zb = b
    keys = set(pa) | set(pb)
    important = sorted(keys, key=lambda k: abs(pa.get(k, 0) - pb.get(k, 0)), reverse=True)[:5]
    return dict(total_variation=.5 * (sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in keys) + abs(za - zb)),
                maximum_absolute_difference=max((abs(pa.get(k, 0) - pb.get(k, 0)) for k in keys), default=0),
                support_changes=sum((pa.get(k, 0) > 0) != (pb.get(k, 0) > 0) for k in keys),
                zero_result_probabilities=[za, zb],
                largest_light_changes=[dict(world_position=list(k[:3]), material=k[3], mass=k[4],
                                            left_probability=pa.get(k, 0), right_probability=pb.get(k, 0)) for k in important])


def merged(children):
    children = tuple(children)
    data = np.zeros(7)
    data[:3] = np.min([c.data[:3] for c in children], axis=0)
    data[4:7] = np.max([c.data[4:7] for c in children], axis=0)
    mass = np.float32(0)
    for child in children:
        mass = np.float32(mass + np.float32(child.data[3]))
    data[3] = mass
    return Node(data, None, children)


def rebuild_outer(bricks, grid, camera_brick, world, arity):
    bits = {2: 1, 4: 2, 8: 3}[arity]
    origin = np.asarray(camera_brick) - grid // 2
    entries = {}
    for xyz, node in bricks.items():
        relative = np.asarray(xyz) - origin
        assert np.all(relative >= 0) and np.all(relative < grid)
        slot = (np.asarray(xyz) & (grid - 1)) if world else relative
        key = int(morton_encode(slot))
        assert key not in entries
        entries[key] = node
    level_bits = 3 * (grid.bit_length() - 1)
    while level_bits:
        width = min(bits, level_bits)
        grouped = {}
        for key in sorted(entries):
            grouped.setdefault(key >> width, []).append(entries[key])
        entries = {key: merged(children) for key, children in grouped.items()}
        level_bits -= width
    return entries.get(0)


def spans(node, depth=0, output=None):
    output = {} if output is None else output
    if node is None or node.material is not None:
        return output
    row = output.setdefault(depth, dict(nodes=0, maximum_diagonal=0., mass=0., mass_times_diagonal=0.))
    diagonal = float(np.linalg.norm(node.data[4:7] - node.data[:3]))
    row['nodes'] += 1
    row['maximum_diagonal'] = max(row['maximum_diagonal'], diagonal)
    row['mass'] += float(node.data[3])
    row['mass_times_diagonal'] += float(node.data[3]) * diagonal
    for child in node.children:
        spans(child, depth + 1, output)
    return output


def mathematical_checks():
    output = []
    for grid in (16, 32, 64):
        for arity in (2, 4, 8):
            for start in (-grid - 1, -1, 0, 1, grid * 10):
                origin = np.array([start, start, start])
                coords = [origin + [1, 3, 3], origin + [2, 3, 3], origin + [grid - 1, 3, 3],
                          origin + [3, 4, 3], origin + [3, 5, 3]]
                bricks = {}
                for i, xyz in enumerate(coords):
                    low = xyz * 16
                    bricks[tuple(xyz)] = Node(np.r_[low, float(i + 1), low + 1], 2 + i, ())
                cameras = [origin + grid // 2, origin + grid // 2 + [0, 1, 0]]
                old = [rebuild_outer(bricks, grid, c, False, arity) for c in cameras]
                new = [rebuild_outer(bricks, grid, c, True, arity) for c in cameras]
                assert signature(new[0]) == signature(new[1])
                for position in (origin * 16 + [24, 40, 40], origin * 16 + [40, 48, 48]):
                    for normal in (np.array([1, 0, 0]), np.array([0, 1, 0])):
                        difference = compare_pmf(*(full_pmf(n, position, normal) for n in new))
                        assert difference['maximum_absolute_difference'] < 1e-12
                # Explicit bijection, including negative world coordinates and every grid slot.
                world_axis = np.arange(start, start + grid)
                assert len(np.unique(world_axis & (grid - 1))) == grid
                single = dict(list(bricks.items())[:1])
                assert len(leaves(rebuild_outer(single, grid, cameras[0], True, arity))) == 1
                assert rebuild_outer({}, grid, cameras[0], True, arity) is None
                output.append(dict(grid=grid, arity=arity, origin=origin.tolist(),
                                   old_topology_changed=signature(old[0]) != signature(old[1]),
                                   new_topology_stable=True, full_pmf_tolerance=1e-12))
    return output


def wrap_counterexample():
    bricks = {}
    for i, xyz in enumerate(((1, 3, 3), (16, 3, 3))):
        low = np.asarray(xyz) * 16
        bricks[xyz] = Node(np.r_[low, 1., low + 1], 2 + i, ())
    output = []
    for world in (False, True):
        node = rebuild_outer(bricks, 16, [9, 8, 8], world, 4)
        depth = 0
        while len(node.children) == 1:
            node = node.children[0]
            depth += 1
        output.append(dict(world_slots=world, first_joint_cluster_depth=depth,
                           cluster_extent=(node.data[4:7] - node.data[:3]).tolist()))
    return output


def decode_normal(packed):
    xy = np.array([packed & 255, (packed >> 8) & 255], dtype='u1').view('i1').astype(float)
    xy = np.maximum(xy / 127., -1.)
    n = np.r_[xy, 1 - abs(xy).sum()]
    t = np.clip(-n[2], 0, 1)
    n[:2] += np.where(n[:2] >= 0, -t, t)
    return n / np.linalg.norm(n)


def match_receiver(query, folder, pose):
    """Reproject a fixed world point; never equate pixels across different cameras."""
    meta = read(folder / 'single.json')
    width, height = meta['width'], meta['height']
    inverse = np.asarray(pose['view_inverse'])
    projection_inverse = np.asarray(pose['projection_inverse'])
    camera = np.asarray(pose['position'])
    target = np.asarray(query['origin']) - .003 * np.asarray(query['normal'])
    view = np.linalg.inv(inverse) @ np.r_[target - camera, 1]
    if view[2] >= 0:
        return dict(matched=False, reason='behind camera')
    ndc = view[:2] / (-view[2] * np.diag(projection_inverse)[:2])
    pixel = np.rint((ndc * .5 + .5) * [width, height] - .5).astype(int)
    geometry = np.memmap(folder / 'colortex8.bin', '<u4', mode='r', shape=(height, width, 4))
    albedo = np.memmap(folder / 'colortex9.bin', 'u1', mode='r', shape=(height, width, 4))
    depth = np.memmap(folder / 'colortex10.bin', '<f4', mode='r', shape=(height, width))
    best = None
    for y in range(max(0, pixel[1] - 1), min(height, pixel[1] + 2)):
        for x in range(max(0, pixel[0] - 1), min(width, pixel[0] + 2)):
            z = float(depth[y, x])
            if not -65536 < z < 0:
                continue
            normal = decode_normal(int(geometry[y, x, 0]))
            cosine = float(normal @ query['normal'])
            if int(geometry[y, x, 3] >> 16) != query['material'] or cosine < .999:
                continue
            if not np.array_equal(albedo[y, x, :3], query['albedo']):
                continue
            v = [((x + .5) / width * 2 - 1) * -z * projection_inverse[0, 0],
                 ((y + .5) / height * 2 - 1) * -z * projection_inverse[1, 1], z, 1]
            actual = (inverse @ v)[:3] + camera
            error = float(np.linalg.norm(actual - target))
            if best is None or error < best['world_position_error']:
                best = dict(pixel=[x, y], world_position=actual.tolist(), world_position_error=error,
                            normal_cosine=cosine, material=query['material'], albedo=query['albedo'])
    if best is None or best['world_position_error'] > .15:
        return dict(matched=False, reason='no same material/normal/albedo surface within 0.15 world blocks', nearest=best)
    x, y = best['pixel']
    count_metadata = read(folder / 'quality.valid_count.json')
    assert count_metadata['scalar_type'] == 'FLOAT32'
    counts = np.memmap(folder / 'quality.valid_count.bin', '<f4', mode='r', shape=(height, width, 4))
    complete = bool(np.all(counts[y, x, :3] == 32))
    mean = np.memmap(folder / 'quality.mean.bin', '<f4', mode='r', shape=(height, width, 4))
    variance = np.memmap(folder / 'quality.variance.bin', '<f4', mode='r', shape=(height, width, 4))
    return dict(matched=True, complete_32_frames=complete, **best,
                rgb_mean=mean[y, x, :3].tolist(), sum_rgb_sample_variance=float(variance[y, x, :3].sum()),
                limitation='Nearest projected pixel of the same surface; finite pixel footprints differ between poses.')


def analyze(run):
    rows = sorted(read(run / 'cases.json'), key=lambda row: (int(row['grid']), row['preset_id'], row['method']))
    assert len({(row['grid'], row['preset_id'], row['method']) for row in rows}) == len(rows)
    report = dict(scope='Static-pose and fixed-input diagnostics; not continuous-motion or first-frame evidence.',
                  mathematical_checks=mathematical_checks(), wrap_counterexample=wrap_counterexample(),
                  cases=[], pose_pairs=[], fixed_input=[], pmf_pairs=[], matched_world_receivers=[])
    trees, roots, poses, voxel_summaries = {}, {}, {}, {}
    (run / 'pmfs').mkdir(exist_ok=True)
    queries_by_grid, layouts = {}, {}
    for row in rows:
        if row['method'] == 'brdf':
            continue
        path = run / row['path']
        folder = path.parent if path.is_file() else path
        grid = int(row['grid'])
        label = f'g{grid}/' + row['preset_id'] + '/' + row['method']
        pose = camera_from_audit(folder)
        if grid not in queries_by_grid:
            queries_by_grid[grid] = pose['points'][::max(1, len(pose['points']) // 3)][:3]
        queries = queries_by_grid[grid]
        report['fixed_world_queries'] = queries_by_grid
        header, nodes, links = load_tree(folder / 'tree.bin', 1)
        root = native_node(nodes, links, 1)
        trees[label], poses[label] = root, pose
        layout = row.get('layout', 'periodic' if row['method'] == 'world-a' else 'camera-relative')
        layouts[label] = layout
        brick_roots = captured_brick_roots(folder, nodes, links, grid, pose['camera_brick'], layout)
        roots[label] = brick_roots
        replay = rebuild_grouping(brick_roots, grid, pose['camera_brick'], layout)
        assert ordered_signature(root) == ordered_signature(replay), label
        voxel = world_voxel_identity(folder, grid, pose['camera_brick'])
        voxel_summaries[label] = voxel
        (folder / 'world-voxel-identity.json').write_text(json.dumps(voxel, indent=2) + '\n', encoding='utf-8')
        leaf_list = sorted(leaves(root))
        assert len(leaf_list) == int(header[1])
        audit = audit_capture(folder, 1, 4)
        item = dict(label=label, layout=layout, grid=grid, camera=pose, emitters=len(leaf_list), root_mass=float(root.data[3]),
                    leaf_identity_sha256=digest(np.asarray(leaf_list, '<f8').tobytes()),
                    semantic_topology_sha256=signature(root), outer_replay_matches_native=True,
                    ordered_topology_sha256=ordered_signature(root),
                    world_voxel_sha256=voxel['world_blocks_material_sha256'], allocated_bricks=voxel['allocated_bricks'],
                    nonempty_voxels=voxel['nonempty_voxels'], tree_audit=audit, spans_by_depth=spans(root))
        report['cases'].append(item)
        for i, query in enumerate(queries):
            report['matched_world_receivers'].append(dict(label=label, point=i, **match_receiver(query, folder, pose)))
        for i, query in enumerate(queries):
            pmf, zero = full_pmf(root, np.asarray(query['origin']), np.asarray(query['normal']))
            np.savez_compressed(run / 'pmfs' / (label.replace('/', '-') + f'-point{i}.npz'),
                                leaf_keys=np.asarray(leaf_list), probabilities=np.array([pmf.get(k, 0) for k in leaf_list]),
                                zero_result_probability=zero)
    labels = list(trees)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            left_grid, left_preset, left_method = left.split('/')
            right_grid, right_preset, right_method = right.split('/')
            if left_grid != right_grid:
                continue
            if left_preset != right_preset and left_method != right_method:
                continue
            same_lamps = sorted(leaves(trees[left])) == sorted(leaves(trees[right]))
            same_voxels = voxel_summaries[left]['world_blocks_material_sha256'] == voxel_summaries[right]['world_blocks_material_sha256']
            pair = dict(left=left, right=right, same_world_lights_and_mass=same_lamps,
                        same_world_voxels=same_voxels, same_semantic_topology=signature(trees[left]) == signature(trees[right]),
                        same_ordered_topology=ordered_signature(trees[left]) == ordered_signature(trees[right]),
                        camera_brick_delta=(np.array(poses[right]['camera_brick']) - poses[left]['camera_brick']).tolist())
            report['pose_pairs'].append(pair)
            for i, query in enumerate(queries_by_grid[int(left_grid[1:])]):
                comparison = compare_pmf(*(full_pmf(trees[label], np.asarray(query['origin']), np.asarray(query['normal'])) for label in (left, right)))
                report['pmf_pairs'].append(dict(left=left, right=right, point=i, **comparison))
                if same_lamps and layouts[left] == layouts[right] and layouts[left] in ('periodic', 'world-prefix'):
                    assert pair['same_ordered_topology'] and comparison['maximum_absolute_difference'] < 1e-9
    # A real reachable subset, fully contained in both shifted windows. Explicitly offline attribution.
    for grid in queries_by_grid:
        label = next(value for value in labels if value.startswith(f'g{grid}/'))
        camera = np.asarray(poses[label]['camera_brick'])
        fixed = {k: n for k, n in roots[label].items() if k[1] >= camera[1] - grid // 2 + 1}
        cameras = [camera, camera + [0, 1, 0]]
        prefix_run = any(layout == 'world-prefix' for layout in layouts.values())
        for layout in ('camera-relative', 'world-prefix' if prefix_run else 'periodic'):
            replay = [rebuild_grouping(fixed, grid, c, layout) for c in cameras]
            result = dict(source=label, evidence='Offline rebuild of a fixed subset of runtime brick subtrees; not a runtime equal-input pair.',
                          layout=layout, grid=grid, bricks=len(fixed), emitters=len(leaves(replay[0])),
                          excluded_bricks=len(roots[label]) - len(fixed),
                          camera_bricks=[c.tolist() for c in cameras], same_semantic_topology=signature(replay[0]) == signature(replay[1]), queries=[])
            for query in queries_by_grid[grid]:
                comparison = compare_pmf(*(full_pmf(n, np.asarray(query['origin']), np.asarray(query['normal'])) for n in replay))
                result['queries'].append(comparison)
                if layout in ('periodic', 'world-prefix'):
                    assert result['same_semantic_topology'] and comparison['maximum_absolute_difference'] < 1e-12
            report['fixed_input'].append(result)
    (run / 'moving-analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(cases=len(report['cases']), mathematical_cases=len(report['mathematical_checks']),
                          runtime_equal_input_pairs=sum(p['same_world_lights_and_mass'] and p['same_world_voxels'] for p in report['pose_pairs']),
                          output=str(run / 'moving-analysis.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--math-only', action='store_true')
    args = parser.parse_args()
    if args.math_only:
        print(json.dumps(dict(cases=len(mathematical_checks()), status='passed')))
    else:
        analyze(args.run)

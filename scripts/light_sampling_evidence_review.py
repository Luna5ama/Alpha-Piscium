"""Re-examine saved captures; no shader changes or runtime calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from light_sampling_geometry import cube_integral


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / 'builds/light-sampling/20261029-geometry-matrix'
OUTPUT = ROOT / 'builds/light-sampling/20261031-evidence-review'
SCENES = ['room1', 'room2', 'basilica', 'gishadow', 'tunnel']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(directory, names):
    manifest = read_json(directory / 'manifest.json')
    entries = {Path(item['relative_path']).name: item for item in manifest['files']}
    hashes = {}
    for name in names:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert digest == entries[name]['sha256'], (directory, name)
        hashes[name] = digest
    return hashes


def historical_audit_addressing(width, height):
    slots = [(y - 45) // 90 * 8 + (x - 80) // 160
             for y in range(45, height, 90) for x in range(80, width, 160)]
    counts = Counter(slots)
    return dict(width=width, height=height, writers=len(slots), unique_slots=len(counts),
                max_slot=max(slots), out_of_range_writers=sum(i >= 64 for i in slots),
                colliding_valid_slots=sum(i < 64 and n > 1 for i, n in counts.items()))


def compact_binary_tree(path):
    words = np.fromfile(path, '<u4')
    header = words[:8]
    assert header[2] == header[3] == 0
    words = words[8:].reshape(-1, 8)
    values = words.view('<f4')
    pending, ids, depths = [(1, 0)], [], []
    while pending:
        index, depth = pending.pop()
        if values[index, 3] == 0:
            continue
        ids.append(index)
        depths.append(depth)
        link = int(words[index, 7])
        if not link & 0x80000000:
            pending.extend([(link + 1, depth + 1), (link, depth + 1)])
    assert len(ids) == len(set(ids))
    remap = {index: i for i, index in enumerate(ids)}
    nodes = np.vstack([values[ids, :7], np.zeros(7)]).astype(np.float64)
    leaves, internal, children = [], [], []
    for i, index in enumerate(ids):
        link = int(words[index, 7])
        if link & 0x80000000:
            leaves.append(i)
        else:
            internal.append(i)
            children.append([remap.get(link, len(ids)), remap.get(link + 1, len(ids))])
    internal = np.array(internal)
    children = np.array(children)
    depths = np.array(depths)
    leaves = np.array(leaves)
    assert len(leaves) == header[1]
    assert np.allclose(nodes[internal, 3], nodes[children, 3].sum(1), rtol=2e-6)
    assert np.all(nodes[leaves, 4:7] - nodes[leaves, :3] == 1)
    return nodes, internal, children, depths, leaves, header


def evaluate_capture(scene, case, projection_cache):
    directory = GEOMETRY / 'raw' / scene / case
    names = ['tree.bin', 'tree.json', 'audit.bin', 'audit.json', 'single.json']
    hashes = verify(directory, names)
    provenance = read_json(directory / 'job-result.json')['result']['provenance']
    settings = {s['name']: s['value'] for s in provenance['effective_settings']['settings']}
    assert not provenance['stale'] and not provenance['head_changed']
    assert settings['SETTING_DEBUG_LIGHT_CUBE'] == settings['SETTING_DEBUG_LIGHT_AUDIT'] == 'true'
    assert settings['SETTING_DEBUG_LIGHT_WEIGHT'] == 'false'
    assert settings['SETTING_DEBUG_LIGHT_SAMPLING'] == '2'
    assert settings['SETTING_DEBUG_LIGHT_LOOKAHEAD'] == ('true' if case == 'cube-ahead' else 'false')
    image_meta = read_json(directory / 'single.json')
    assert (image_meta['width'], image_meta['height']) == (1280, 720)
    frame = read_json(directory / 'tree.json')['frame_id']
    assert frame == read_json(directory / 'audit.json')['frame_id']
    nodes, internal, children, depths, leaves, header = compact_binary_tree(directory / 'tree.bin')
    records = np.fromfile(directory / 'audit.bin', '<f4').reshape(64, 4, 4).astype(np.float64)
    slots = np.flatnonzero(records[:, 3, 3] != 0)
    records = records[slots]
    origin, normal = records[:, 0, :3], records[:, 1, :3]
    assert np.allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-5)
    half = (nodes[:, 4:7] - nodes[:, :3]) * .5
    delta = nodes[None, :, :3] + half[None, :, :] - origin[:, None, :]
    r2 = (half * half).sum(1)
    d2 = (delta * delta).sum(2)
    dot = (delta * normal[:, None, :]).sum(2)
    cosine = np.clip((dot + np.sqrt(r2)) / np.sqrt(np.maximum(d2, .25)), .001, 1)
    weights = nodes[None, :, 3] * cosine / np.maximum(np.maximum(d2, r2), .25)
    scores = weights.copy()
    if case == 'cube-ahead':
        for _ in range(2):
            refined = scores.copy()
            refined[:, internal] = scores[:, children].sum(2)
            scores = refined
    totals = scores[:, children].sum(2)
    assert np.all(totals > 0)
    conditionals = scores[:, children] / totals[:, :, None]
    probabilities = np.zeros_like(weights)
    probabilities[:, 0] = 1
    for depth in range(int(depths.max())):
        rows = np.flatnonzero(depths[internal] == depth)
        for child_slot in range(2):
            child_ids = children[rows, child_slot]
            real = child_ids < len(depths)
            probabilities[:, child_ids[real]] = (
                probabilities[:, internal[rows[real]]] * conditionals[:, rows[real], child_slot])
    q = probabilities[:, leaves]
    assert np.all(q > 0) and np.allclose(q.sum(1), 1, atol=1e-12)
    leaf_lookup = {tuple(nodes[i, :3]): j for j, i in enumerate(leaves)}
    selected = np.array([leaf_lookup[tuple(record[3, :3])] for record in records])
    pdf_errors = np.abs(records[:, 1, 3] / q[np.arange(len(records)), selected] - 1)
    assert pdf_errors.max() < .005

    leaf_score = weights[:, leaves]
    p = leaf_score / leaf_score.sum(1)[:, None]
    subtree_sum = np.zeros_like(weights)
    subtree_sum[:, leaves] = leaf_score
    for depth in range(int(depths.max()) - 1, -1, -1):
        rows = np.flatnonzero(depths[internal] == depth)
        subtree_sum[:, internal[rows]] = subtree_sum[:, children[rows]].sum(2)
    assert np.allclose(subtree_sum[:, 0], leaf_score.sum(1))
    ideal_branch = subtree_sum[:, children] / subtree_sum[:, internal, None]
    positive = ideal_branch > 0
    log_ratio = np.zeros_like(ideal_branch)
    log_ratio[positive] = np.log(ideal_branch[positive] / conditionals[positive])
    kl_nodes = (subtree_sum[:, internal] / subtree_sum[:, 0, None]
                * (ideal_branch * log_ratio).sum(2))
    kl_depth = np.array([kl_nodes[:, depths[internal] == d].sum(1)
                         for d in range(int(depths.max()))]).T
    kl_total = (p * np.log(p / q)).sum(1)
    assert np.allclose(kl_depth.sum(1), kl_total, atol=1e-10)
    horizon_max = dot[:, leaves] + (np.abs(normal[:, None, :]) * half[None, leaves, :]).sum(2)
    horizon_zero = horizon_max <= 0
    projected = np.zeros_like(q)
    leaf_positions = [tuple(nodes[index, :3]) for index in leaves]
    for i, slot in enumerate(slots):
        cache_key = (scene, int(slot), tuple(origin[i]), tuple(normal[i]))
        if cache_key not in projection_cache:
            projection_cache[cache_key] = {
                low: 0.0 if horizon_zero[i, j] else cube_integral(np.array(low), origin[i], normal[i])
                for j, low in enumerate(leaf_positions)}
        projected[i] = [projection_cache[cache_key][low] for low in leaf_positions]
    assert np.all(projected >= 0) and np.all(projected.sum(1) > 0)
    projected *= nodes[leaves, 3]
    projected /= projected.sum(1)[:, None]
    rows = []
    for i, slot in enumerate(slots):
        importance_order = np.argsort(-p[i])
        important = importance_order[:np.searchsorted(np.cumsum(p[i, importance_order]), .9) + 1]
        most = int(importance_order[0])
        rows.append(dict(slot=int(slot), pixel_raw=[int(80 + slot % 8 * 160), int(45 + slot // 8 * 90)],
                         origin=origin[i].tolist(), normal=normal[i].tolist(),
                         total_variation=float(np.abs(p[i] - q[i]).sum() * .5),
                         leaf_score_second_moment_ratio=float(np.sum(p[i] ** 2 / q[i])),
                         flat_top90_tree_probability=float(q[i, important].sum()),
                         below_horizon_tree_probability=float(q[i, horizon_zero[i]].sum()),
                         below_horizon_flat_probability=float(p[i, horizon_zero[i]].sum()),
                         cube_proxy_second_moment_tree=float(np.sum(projected[i] ** 2 / q[i])),
                         cube_proxy_second_moment_flat=float(np.sum(projected[i] ** 2 / p[i])),
                         kl_depth=kl_depth[i].tolist(),
                         largest_leaf_score=dict(origin=nodes[leaves[most], :3].tolist(),
                                                 flat_probability=float(p[i, most]),
                                                 tree_probability=float(q[i, most])),
                         selected_pdf_relative_error=float(pdf_errors[i])))
    keys = ['total_variation', 'leaf_score_second_moment_ratio', 'flat_top90_tree_probability',
            'below_horizon_tree_probability', 'below_horizon_flat_probability',
            'cube_proxy_second_moment_tree', 'cube_proxy_second_moment_flat']
    summary = {key: dict(zip(['min', 'median', 'p90', 'max'],
                            map(float, np.quantile([r[key] for r in rows], [0, .5, .9, 1]))))
               for key in keys}
    counters = dict(zip(['allocated_nodes', 'emitters', 'overflow', 'exhausted', 'traces', 'visible',
                         'receivers', 'reserved'], map(int, header)))
    source = {key: provenance[key] for key in ['resolved_revision', 'source_snapshot_sha256',
                                               'scene_sha256', 'preset_sha256', 'preset_id']}
    depth_fraction = kl_depth.sum(0) / kl_depth.sum()
    return dict(scene=scene, case=case, frame=frame, source=source, source_artifacts=hashes, counters=counters,
                receivers=len(rows), max_selected_pdf_relative_error=float(pdf_errors.max()),
                leaf_score_kl_fraction_by_depth=depth_fraction.tolist(), summary=summary, points=rows)


def main():
    result = dict(
        scope='Offline reanalysis of original 720p captures; no new GPU captures or timing.',
        target='Normalize the exact same original per-leaf geometric score. This is an unoccluded proxy, not ground-truth lighting.',
        second_moment_ratio='sum(p_leaf_score^2/q_tree); equals one for exact sampling of this proxy. It is not a measured rendered-variance ratio.',
        cube_proxy='Analytically integrate the hemisphere-clipped cube projected solid angle, multiplied by the same fixed leaf mass. Ignore occlusion, texture variation, and conditional sampling error; report sum(p_cube^2/q) for the tree and flat leaf-score proposals. This is not measured rendered variance.',
        limitations=['64 fixed audit pixels at most; not a uniform scene survey.',
                     'No per-ray rejection reason or all-light visibility is captured.',
                     'Thirty-two-frame sample variance is not converged-reference MSE.',
                     '1080p audit indexing collides and writes outside the 64-record buffer; use 720p for this reconstruction.'],
        addressing=[historical_audit_addressing(1280, 720), historical_audit_addressing(1920, 1080)],
        captures=[])
    projection_cache = {}
    for scene in SCENES:
        for case in ['cube-plain', 'cube-ahead']:
            capture = evaluate_capture(scene, case, projection_cache)
            result['captures'].append(capture)
            print(scene, case, capture['receivers'],
                  {k: round(v['median'], 4) for k, v in capture['summary'].items()}, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'leaf-score-review.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

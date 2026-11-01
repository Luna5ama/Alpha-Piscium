"""Inspect current four-way light-tree captures without modifying their originals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_geometry import cube_integral
from light_sampling_runtime import LUMA, read_image, render
from light_sampling_weight_reference import audit_capture, distribution, load_tree

ROOT = Path(__file__).resolve().parents[1]
FIXED_ROIS = {
    'room1': {'wall': (410,280,560,350), 'dark': (410,360,560,500), 'near_light': (590,280,650,345)},
    'room2': {'wall': (900,150,1040,230), 'dark': (760,300,900,430), 'near_light': (950,70,1070,150)},
    'gishadow': {'near_light': (320,175,550,360)},
    'tunnel': {'floor': (180,520,800,700), 'wall': (950,220,1250,550)},
}


def leaf_identity(path, model):
    _, nodes, links = load_tree(path, model)
    leaves, pending = [], [1]
    while pending:
        i = pending.pop()
        if nodes[i, 3] == 0:
            continue
        link = int(links[i])
        if link & 0x80000000:
            leaves.append((tuple(nodes[i, :3]), link & 0x7fffffff, nodes[i, 3]))
        else:
            base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
            pending.extend(range(base, base + count))
    return sorted(leaves)


def topology_identity(path, model, ordered=False):
    _, nodes, links = load_tree(path, model)
    def visit(i):
        if nodes[i, 3] == 0:
            return b''
        link = int(links[i])
        if link & 0x80000000:
            return hashlib.sha256(b'L' + np.asarray(nodes[i, :3], dtype='<f4').tobytes()
                                  + (link & 0x7fffffff).to_bytes(4, 'little')).digest()
        base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
        children = list(filter(None, (visit(j) for j in range(base, base+count))))
        return hashlib.sha256(b'N' + b''.join(children if ordered else sorted(children))).digest()
    return visit(1).hex()


def voxel_identity(directory, prefix='', suffix='', grid_size=16):
    meta = np.fromfile(directory / f'{prefix}voxel-meta{suffix}.bin', '<u4')
    material = np.memmap(directory / f'{prefix}voxel-materials{suffix}.bin', dtype='<u4', mode='r')
    count = int(meta[0]); grid = grid_size ** 3
    bricks = meta[1 + 2 * grid + 1024:1 + 2 * grid + 1024 + count]
    assert len(bricks) == count and len(np.unique(bricks)) == count
    assert count * 4096 <= len(material)
    parts = []
    for alloc, brick in sorted(enumerate(bricks), key=lambda item: item[1]):
        assert brick < grid and meta[1 + grid + int(brick)] == alloc
        values = material[alloc*4096:(alloc+1)*4096]
        present = np.flatnonzero(values)
        packed = np.empty((len(present), 3), dtype='<u4')
        packed[:, 0], packed[:, 1], packed[:, 2] = brick, present, values[present]
        parts.append(packed)
    records = np.concatenate(parts) if parts else np.empty((0, 3), dtype='<u4')
    return records, dict(allocated_bricks=count, nonempty_voxels=len(records), sha256=hashlib.sha256(records.tobytes()).hexdigest())


def baseline_review(directory, cube_points, target_origin=None, target_normal=None):
    header, nodes, links = load_tree(directory / 'tree.bin', 0)
    children, depths, leaves = {}, {}, []
    pending = [(1, 0)]
    while pending:
        index, depth = pending.pop()
        depths[index] = depth
        if nodes[index, 3] == 0:
            continue
        link = int(links[index])
        if link & 0x80000000:
            leaves.append(index)
        else:
            base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
            children[index] = list(range(base, base + count))
            pending.extend((child, depth + 1) for child in children[index])
    leaf_ids = np.array(leaves)
    records = np.fromfile(directory / 'audit.bin', '<f4')[:1024].reshape(64, 4, 4)
    slots = np.flatnonzero(records[:, 3, 3] != 0)
    if target_origin is not None:
        distance = np.linalg.norm(records[slots, 0, :3] - target_origin, axis=1)
        mismatch = 1 - records[slots, 1, :3] @ target_normal
        slots = slots[np.argsort(distance + 10 * np.maximum(mismatch, 0))]
    cube_slots = set(slots[:cube_points])
    rows = []
    for slot, record in enumerate(records):
        if record[3, 3] == 0:
            continue
        origin, normal = record[0, :3].astype(float), record[1, :3].astype(float)
        q_map, zero = distribution(nodes, links, origin, normal, 0)
        q = np.array([q_map.get(i, 0) for i in leaves])
        half = (nodes[leaf_ids, 4:7] - nodes[leaf_ids, :3]) * .5
        delta = nodes[leaf_ids, :3] + half - origin
        d2 = np.sum(delta * delta, axis=1)
        r2 = np.sum(half * half, axis=1)
        cosine = np.clip((delta @ normal + np.sqrt(r2)) / np.sqrt(np.maximum(d2, .25)), .001, 1)
        scores = nodes[leaf_ids, 3] * cosine / np.maximum.reduce([d2, r2, np.full_like(d2, .25)])
        p = scores / scores.sum()
        subtree = {i: float(s) for i, s in zip(leaves, scores)}
        proposal_subtree = dict(q_map)
        for i in sorted(children, key=depths.get, reverse=True):
            subtree[i] = sum(subtree.get(j, 0) for j in children[i])
            proposal_subtree[i] = sum(proposal_subtree.get(j, 0) for j in children[i])
        by_depth = {}
        branches = []
        for i, kids in children.items():
            ideal = np.array([subtree.get(j, 0) for j in kids]) / subtree[i]
            actual_mass = np.array([proposal_subtree.get(j, 0) for j in kids])
            actual = actual_mass / actual_mass.sum()
            positive = ideal > 0
            contribution = subtree[i] / subtree[1] * np.sum(ideal[positive] * np.log(ideal[positive] / actual[positive]))
            depth = depths[i]
            by_depth[depth] = by_depth.get(depth, 0) + float(contribution)
            branches.append(dict(node=i, depth=depth, kl=float(contribution), ideal=ideal.tolist(), actual=actual.tolist()))
        kl = float(np.sum(p * np.log(p / q)))
        assert np.isclose(sum(by_depth.values()), kl, atol=1e-9)
        behind = delta @ normal + np.sum(np.abs(normal) * half, axis=1) <= 0
        row = dict(slot=slot, origin=origin.tolist(), normal=normal.tolist(), zero_probability=zero,
                   below_horizon_probability=float(q[behind].sum()), total_variation=float(np.abs(p-q).sum()*.5),
                   kl=kl, kl_by_depth=by_depth, largest_branches=sorted(branches, key=lambda x: x['kl'], reverse=True)[:10])
        if slot in cube_slots:
            weights = np.array([cube_integral(nodes[i, :3], origin, normal) * nodes[i, 3] for i in leaves])
            if weights.sum() > 0:
                target = weights / weights.sum()
                row['unoccluded_cube_proxy'] = dict(tree_second_moment=float(np.sum(target**2 / q)),
                                                    flat_second_moment=float(np.sum(target**2 / p)))
        rows.append(row)
    return dict(audit=audit_capture(directory, 0), receiver_points=rows,
                cube_selection=dict(target_origin=target_origin, target_normal=target_normal, slots=sorted(map(int, cube_slots))),
                cube_scope='Fixed mass, unoccluded projected cube integral; excludes texture and visibility.')


def quality_review(run):
    from scipy.ndimage import gaussian_filter
    rows, images = [], []
    for scene in ('room1', 'room2', 'basilica', 'gishadow', 'tunnel'):
        captures = []
        for name, model in [('original', 0), ('halfspace-a', 1), ('cosine-b', 2), ('near-far', 3), ('moment-sphere', 4), ('diffuse-sg', 5), ('brdf', 0)]:
            suffix = '-fixed' if scene == 'basilica' and name in ('halfspace-a', 'cosine-b') else ''
            if scene in ('room1','gishadow'):
                directory = run / 'matched-cohort' / scene / name
            elif scene == 'room2':
                directory = run / 'near-cohort' / scene / name
            else:
                directory = run / ('final-raw' if scene == 'basilica' else 'cohort-final') / scene / (name + suffix)
            job = json.loads((directory / 'job-result.json').read_text())['result']
            p = job['provenance']
            old = '44d13f2115520ea8170aff79c196bade84a47ac2'
            moment_fix = 'ca93e3b340aa42b7cca9002d0ba6d31c464682e7'
            support_fix = '4c5b33c04ab93a2de7b2b4b58ff63cfa6ac8e675'
            accepted = (support_fix,) if model in (1,2) else ((moment_fix,support_fix) if model in (4,5) else (old,moment_fix,support_fix))
            assert p['resolved_revision'] in accepted, (scene,name,p['resolved_revision'])
            assert not p['head_changed'] and not p['stale']
            settings = {s['name']:s['value'] for s in job['action_receipts'][0]['runtime_mutation']['effective_settings']['settings']}
            required = {'SETTING_DEBUG_LIGHT_ARITY':'4', 'SETTING_DEBUG_LIGHT_AUDIT':'true', 'SETTING_DEBUG_LIGHT_CUBE':'true',
                        'SETTING_DEBUG_LIGHT_SAMPLING':'1' if name=='brdf' else '2', 'SETTING_DEBUG_LIGHT_WEIGHT_MODEL':str(model),
                        'SETTING_RC_ENABLE':'false','SETTING_GI_PATH_GUIDING':'false','SETTING_GI_SPATIAL_REUSE':'false',
                        'SETTING_TAA':'false','SETTING_DENOISER_ACCUM':'false','SETTING_DENOISER_SPATIAL':'false','SETTING_VOXEL_GRID_SIZE':'16'}
            assert all(settings[k] == v for k,v in required.items())
            assert 'SETTING_DEBUG_LIGHT_WEIGHT' not in settings and 'SETTING_DEBUG_LIGHT_LOOKAHEAD' not in settings
            tree = leaf_identity(directory / 'tree.bin', model)
            if captures:
                assert tree == captures[0]['tree']
                assert p['scene_sha256'] == captures[0]['scene_sha256']
            header, _, _ = load_tree(directory / 'tree.bin', model)
            reasons = np.fromfile(directory / 'audit.bin', '<u4')[1024:1040]
            assert int(sum(map(int, reasons[:10]))) == int(header[6])
            assert reasons[10] == 0
            assert json.loads((directory/'tree.json').read_text())['frame_id'] == json.loads((directory/'audit.json').read_text())['frame_id']
            single, sm = read_image(directory / 'single.bin')
            assert (sm['width'], sm['height']) == (1280,720)
            mean, mm = read_image(directory / 'mean32.mean.bin')
            variance, _ = read_image(directory / 'mean32.variance.bin')
            count, _ = read_image(directory / 'mean32.valid_count.bin')
            incomplete, _ = read_image(directory / 'mean32.incomplete_batches.bin')
            acc = mm['accumulation']
            assert acc['frames'] == 32 and acc['end_frame'] - acc['start_frame'] == 31
            assert sm['frame_id'] < acc['start_frame'] and np.all(variance >= 0)
            complete = np.all(count == 32, axis=2) & np.all(incomplete == 0, axis=2)
            row = dict(scene=scene, sampler=name, revision=p['resolved_revision'], source=str(directory), scene_sha256=p['scene_sha256'], emitters=len(tree),
                       reasons=list(map(int, reasons)), incomplete_pixels=int((~complete).sum()),
                       missing_samples_by_channel=[int(np.maximum(32-count[:,:,i],0).sum()) for i in range(4)])
            captures.append(dict(row=row, tree=tree, scene_sha256=p['scene_sha256'], mean=mean, variance=variance, complete=complete))
            for accumulation, data, tag in [('Single', single, 'single'), ('32 frames', mean, 'mean32')]:
                image = run / 'images' / f'{scene}-{name}-{tag}.png'
                image.parent.mkdir(exist_ok=True)
                render(data, image, 6)
                images.append(dict(path=str(image.resolve()), dimensions={'Scene':scene, 'Sampler':name, 'Accumulation':accumulation}))
        mask = np.logical_and.reduce([c['complete'] & (c['mean'][:,:,3] > 0) for c in captures])
        pooled = gaussian_filter(np.mean([c['mean'][:,:,:3] @ LUMA for c in captures], axis=0), 2)
        lit = mask & (pooled > 1e-6)
        q50, q99 = np.quantile(pooled[lit], [.5,.99])
        masks = {'receiver':mask, 'dark_half':lit & (pooled <= q50), 'brightest_1':lit & (pooled >= q99)}
        for label, (x0,y0,x1,y1) in FIXED_ROIS.get(scene, {}).items():
            region = np.zeros_like(mask); region[y0:y1,x0:x1] = True
            masks['fixed_'+label] = region[::-1] & mask
        for capture in captures:
            row, mean, variance = capture['row'], capture['mean'], capture['variance']
            row['regions'] = {k:dict(pixels=int(m.sum()), mean_luminance=float((mean[:,:,:3] @ LUMA)[m].mean()),
                                     mean_rgb=np.mean(mean[:,:,:3][m],axis=0).tolist(), variance_rgb=np.mean(variance[:,:,:3][m],axis=0).tolist(),
                                     rgb_variance=float(variance[:,:,:3][m].sum(axis=1).mean()),
                                     observed_zero_mean_pixels=int(np.sum(np.all(mean[:,:,:3][m] == 0,axis=1)))) for k,m in masks.items() if m.any()}
            row['fixed_roi_background_pixels'] = {}
            for label, (x0,y0,x1,y1) in FIXED_ROIS.get(scene, {}).items():
                region = np.zeros_like(mask); region[y0:y1,x0:x1] = True; region = region[::-1]
                row['fixed_roi_background_pixels'][label] = int(np.sum(region & (mean[:,:,3] <= 0)))
            rows.append(row)
    return dict(scope='32-frame equal-sample diagnostics; pooled quantile masks are exploratory and exclude pooled luminance <= 1e-6; fixed ROIs include all complete receiver pixels', fixed_rois_display_xyxy=FIXED_ROIS, exposure_ev=6, rows=rows, images=images)


def timing_review(run):
    output = []
    for scene in ('room1', 'room2', 'basilica', 'gishadow', 'tunnel'):
        folder = {'room1':'room1-matched','room2':'room2-near','gishadow':'gishadow-matched'}.get(scene,scene)
        job = json.loads((run / 'perf-final' / folder / 'profile' / 'profile.json').read_text())
        assert job['success'] and job['passed'] == 14
        required = {'prepare2_compute', 'prepare3_compute', 'composite41_compute', 'composite42_compute'} | {f'prepare{i}_compute' for i in range(6,20)}
        assert not any(c['metric_id'] in required for c in job['metric_conflicts'])
        cases = job['cases']
        assert [c['config_id'] for c in cases[:7]] == list(reversed([c['config_id'] for c in cases[7:]]))
        for order, case in enumerate(cases, 1):
            assert case['status'] == 'passed' and case['frames'] == case['warmup_frames'] == 128
            p = case['provenance']
            assert not p['stale'] and not p['head_changed']
            metrics = {m['metric_id']: m for m in case['metrics']['metrics'] if m.get('main_source')}
            build = [m for m in metrics.values() if m['main_source'] == '/pass/prepare/LightSamplingBuild.comp.glsl']
            initial = [m for m in metrics.values() if m['main_source'] == '/pass/composite/LightSamplingInitial.comp.glsl']
            assert len(build) == 14 and len(initial) == 1
            assert all(len(m['samples_ns']) == 128 for m in build + initial)
            build_ms = sum(m['average_ns'] for m in build) / 1e6
            initial_ms = initial[0]['average_ns'] / 1e6
            siblings = {name:metrics[name]['average_ns']/1e6 for name in ('prepare2_compute','prepare3_compute','composite42_compute') if name in metrics}
            output.append(dict(scene=scene, sampler=case['config_id'], order=order, revision=p['resolved_revision'],
                               scene_sha256=p['scene_sha256'], initial_avg_ms=initial_ms,
                               build_sum_avg_ms=build_ms, required_avg_ms=initial_ms + (0 if case['config_id']=='brdf' else build_ms),
                               actual_diagnostic_avg_ms=initial_ms + build_ms, siblings_avg_ms=siblings,
                               stages={m['metric_id']:m['average_ns']/1e6 for m in build}))
    return output


def reference_review(run):
    from scipy.ndimage import gaussian_filter
    quality = json.loads((run / 'quality-final.json').read_text())['rows']
    output = []
    for scene in ('room1', 'room2', 'basilica', 'gishadow', 'tunnel'):
        rows = [r for r in quality if r['scene'] == scene]
        if scene in ('room1','room2','gishadow'):
            folder = run / 'reference' / scene / ('near-light' if scene == 'room2' else 'matched')
            files = [folder / 'window1.mean.bin', folder / 'window2.mean.bin']
            baseline_files = []
            ref_tree = folder / 'tree.bin'
            refs = [folder / 'tree-before.bin', ref_tree]
        else:
            folder = run / 'reference' / scene / 'combined'
            files = [folder / 'brdf-window1.mean.bin', folder / 'brdf-window2.mean.bin']
            baseline_files = [folder / f'original-window{i}.mean.bin' for i in (1,2)]
            ref_tree = folder / 'brdf-after.bin'
            refs = [folder / f'{method}-{stage}.bin' for method in ('original', 'brdf') for stage in ('before', 'after')]
        quality_tree = Path(rows[0]['source']) / 'tree.bin'
        reference_identity = leaf_identity(ref_tree,0)
        if any(leaf_identity(p,0) != reference_identity or topology_identity(p,0) != topology_identity(ref_tree,0) for p in refs):
            output.append(dict(scene=scene, status='reference tree changed across windows'))
            continue
        if leaf_identity(quality_tree,0) != reference_identity or topology_identity(quality_tree,0) != topology_identity(ref_tree,0):
            output.append(dict(scene=scene,status='quality and reference tree differ',quality_emitters=len(leaf_identity(quality_tree,0)),reference_emitters=len(reference_identity)))
            continue
        def read_reference(path):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['scalar_type'] == 'FLOAT32' and meta['channel_count'] == 4
            return np.fromfile(path,dtype='<f4').reshape(meta['height'],meta['width'],4), meta
        windows, metadata = zip(*(read_reference(p) for p in files))
        assert all(m['accumulation']['frames'] == 1024 for m in metadata)
        region = metadata[0]['region']; x,y,w,h = (region[k] for k in ('x','y','width','height'))
        assert metadata[1]['region'] == region
        reference_complete = np.ones((h,w), dtype=bool)
        reference_windows = []
        for path, window in zip(files, windows):
            stem = str(path).removesuffix('.mean.bin')
            count,_ = read_reference(Path(stem+'.valid_count.bin'))
            incomplete,_ = read_reference(Path(stem+'.incomplete_batches.bin'))
            nonfinite,_ = read_reference(Path(stem+'.nonfinite.bin'))
            finite = np.all(np.isfinite(window) & np.isfinite(count) & np.isfinite(incomplete) & np.isfinite(nonfinite),axis=2)
            complete = finite & np.all(count == 1024,axis=2) & np.all(incomplete == 0,axis=2) & np.all(nonfinite == 0,axis=2)
            reference_complete &= complete
            reference_windows.append(dict(incomplete_pixels=int((~complete).sum()),
                missing_samples_by_channel=[int(np.maximum(1024-np.nan_to_num(count[:,:,i],nan=0),0).sum()) for i in range(4)],
                nonfinite_samples_by_channel=[int(np.nan_to_num(nonfinite[:,:,i],nan=0).sum()) for i in range(4)]))
        windows = [np.nan_to_num(a,nan=0,posinf=0,neginf=0) for a in windows]
        mean = (windows[0][:,:,:3] + windows[1][:,:,:3]) * .5
        uncertainty = (windows[0][:,:,:3] - windows[1][:,:,:3])**2 * .25
        low_ref = gaussian_filter(mean, sigma=(8,8,0))
        low_uncertainty = ((gaussian_filter(windows[0][:,:,:3], sigma=(8,8,0)) - gaussian_filter(windows[1][:,:,:3], sigma=(8,8,0)))**2)*.25
        captures = []
        for row in rows:
            directory = Path(row['source'])
            values, _ = read_image(directory / 'mean32.mean.bin')
            count, _ = read_image(directory / 'mean32.valid_count.bin')
            incomplete, _ = read_image(directory / 'mean32.incomplete_batches.bin')
            samples = values[y:y+h,x:x+w,:3]
            valid = np.all(count[y:y+h,x:x+w,:] == 32, axis=2) & np.all(incomplete[y:y+h,x:x+w,:] == 0,axis=2) & (values[y:y+h,x:x+w,3] > 0)
            captures.append((row,samples,valid))
        quality_complete = np.logical_and.reduce([c[2] for c in captures])
        mask = quality_complete & reference_complete
        assert mask.any()
        ref_unc = float(np.mean(uncertainty[mask])); ref_low_unc = float(np.mean(low_uncertainty[mask]))
        baseline_windows = [read_image(path)[0][:,:,:3] for path in baseline_files]
        disagreement = float(np.mean(((((baseline_windows[0]+baseline_windows[1])*.5)-mean)**2)[mask])) if baseline_windows else None
        result = dict(scene=scene,status='matched',region=region,reference='BRDF two disjoint consecutive 1024-frame windows; independence not established',
                      reference_error_estimate_rgb=ref_unc,reference_low_frequency_error_estimate_rgb=ref_low_unc,pixels=int(mask.sum()),
                      reference_windows=reference_windows,quality_eligible_pixels=int(quality_complete.sum()),
                      removed_by_reference_gate=int((quality_complete & ~reference_complete).sum()),methods=[])
        result['baseline_vs_brdf_reference_mse_rgb'] = disagreement
        for row,samples,_ in captures:
            mse = float(np.mean(((samples-mean)**2)[mask]))
            low_mse = float(np.mean(((gaussian_filter(samples,sigma=(8,8,0))-low_ref)**2)[mask]))
            result['methods'].append(dict(sampler=row['sampler'],mse_rgb=mse,low_frequency_mse_rgb=low_mse,
                                          mse_over_reference_uncertainty=mse/ref_unc if ref_unc else None,
                                          low_mse_over_reference_uncertainty=low_mse/ref_low_unc if ref_low_unc else None))
            result['methods'][-1]['fixed_rois'] = {}
            for label, (x0,y0,x1,y1) in FIXED_ROIS.get(scene,{}).items():
                roi = np.zeros((h,w),dtype=bool)
                left,right=max(x0,x)-x,min(x1,x+w)-x
                bottom,top=max(720-y1,y)-y,min(720-y0,y+h)-y
                if left < right and bottom < top:
                    roi[bottom:top,left:right] = True
                roi &= mask
                if roi.any():
                    result['methods'][-1]['fixed_rois'][label] = dict(pixels=int(roi.sum()),
                        mse_rgb=float(np.mean(((samples-mean)**2)[roi])),
                        low_frequency_mse_rgb=float(np.mean(((gaussian_filter(samples,sigma=(8,8,0))-low_ref)**2)[roi])),
                        reference_error_estimate_rgb=float(np.mean(uncertainty[roi])),
                        reference_low_frequency_error_estimate_rgb=float(np.mean(low_uncertainty[roi])))
        output.append(result)
    return output


def report(run):
    quality = json.loads((run/'quality-final.json').read_text())
    timing = json.loads((run/'timing-final.json').read_text())
    reference = json.loads((run/'reference-final.json').read_text())
    for row in quality['rows']:
        t = [x for x in timing if x['scene']==row['scene'] and x['sampler']==row['sampler']]
        assert len(t)==2
        row['timing'] = dict(runs=t, required_avg_ms=float(np.mean([x['required_avg_ms'] for x in t])),
                             build_sum_avg_ms=float(np.mean([x['build_sum_avg_ms'] for x in t])),
                             initial_avg_ms=float(np.mean([x['initial_avg_ms'] for x in t])),
                             forward_reverse_ratio=max(x['required_avg_ms'] for x in t)/min(x['required_avg_ms'] for x in t),
                             sentinel_forward_reverse_ratio=max(x['siblings_avg_ms']['prepare2_compute'] for x in t)/min(x['siblings_avg_ms']['prepare2_compute'] for x in t))
        source = Path(row['source']); audit = json.loads((source/'analysis.json').read_text()) if (source/'analysis.json').exists() else None
        if audit and 'audit' in audit: audit = audit['audit']
        row['audit'] = audit
        row['node_bytes'] = 48 if row['sampler'] in ('moment-sphere','diffuse-sg') else 32
        row['tree_buffer_bytes'] = (source/'tree.bin').stat().st_size
        mm = json.loads((source/'mean32.mean.json').read_text())['accumulation']
        batches = np.asarray(mm['screen_batch_means'])[:,:3] @ LUMA
        row['screen_mean_luminance'] = float(batches.mean())
        row['screen_mean_luminance_se'] = float(batches.std(ddof=1)/np.sqrt(len(batches)))
    for scene in ('room1','room2','basilica','gishadow','tunnel'):
        rows=[r for r in quality['rows'] if r['scene']==scene]
        base=next(r for r in rows if r['sampler']=='original')
        sentinel_values = [t['siblings_avg_ms']['prepare2_compute'] for t in timing if t['scene']==scene]
        scene_sentinel_ratio = max(sentinel_values)/min(sentinel_values)
        for r in rows:
            r['timing']['scene_sentinel_range_ratio'] = scene_sentinel_ratio
            r['timing']['comparison_caution'] = bool(scene_sentinel_ratio > 1.25 or r['timing']['forward_reverse_ratio'] > 1.25 or r['timing']['sentinel_forward_reverse_ratio'] > 1.25)
            r['relative_screen_mean_to_original'] = r['screen_mean_luminance']/base['screen_mean_luminance']-1 if base['screen_mean_luminance'] else None
            for region, stats in r['regions'].items():
                basevar = base['regions'][region]['rgb_variance']
                stats['variance_ratio_to_original'] = stats['rgb_variance']/basevar if basevar else None
                stats['equal_time_variance_ratio_to_original'] = (stats['variance_ratio_to_original']*r['timing']['required_avg_ms']/base['timing']['required_avg_ms']) if basevar else None
    return dict(quality=quality['rows'], timing=timing, references=reference,
                limitations=['Equal-time ratios extrapolate 32-frame variance linearly with total required GPU time; not a directly measured equal-time image.',
                             'Older room1, room2, and gishadow groups with different light trees remain separate and are not used for the matched reference comparisons.',
                             'The first room2 dark ROI is nearly black and inconclusive; the near-light reference uses a separate matching 853-light cohort.',
                             'Fixed ROI low-frequency error is reported only for the intersection with the reference capture region.',
                             'Reference-window difference is an empirical uncertainty indicator, not a confidence interval when temporal correlation is unknown.',
                             'Actual running Vibris jar was not proven byte-identical to the locally inspected time-freezing controller.',
                             'Cross-scene voxel input drift remains unexplained; same-scene matching does not prove every frame had identical non-emissive visibility inputs.'])


def point_reference(run):
    directory = run/'point-reference/main'
    fixture = np.fromfile(ROOT/'shaders/textures/light_sampling_room2_reference.bin', '<i4').reshape(-1,4)
    nlights = len(fixture); lanes = (1280*720)//(2*nlights); used = lanes*2*nlights
    assert nlights == 853 and lanes == 540 and used == 921240
    records = np.fromfile(directory/'audit-before.bin','<f4')[:32].reshape(2,4,4)
    after = np.fromfile(directory/'audit-after.bin','<f4')[:32].reshape(2,4,4)
    assert np.array_equal(records,after) and np.all(records[:,2,3] == 1)
    assert np.array_equal(records[:,0,3],[54,2]) and np.all(records[:,1,3] == lanes)
    assert np.all(records[:,3,2] == nlights) and np.all(records[:,3,3] == 3)
    expected_tree = run/'near-cohort/room2/original/tree.bin'
    for name in ('tree-before.bin','tree-after.bin'):
        tree = directory/name
        assert leaf_identity(tree,0) == leaf_identity(expected_tree,0)
        assert topology_identity(tree,0) == topology_identity(expected_tree,0)
    original = np.fromfile(run/'near-cohort/room2/original/audit.bin','<f4')[:1024].reshape(64,4,4)
    assert np.array_equal(records[:,:2,:3], original[[54,2],:2,:3])
    voxel = json.loads((run/'point-reference/voxel-identity.json').read_text())
    assert voxel['equal']
    windows, summaries = [], []
    for index in (1,2):
        stem = directory/f'window{index}'
        def read(suffix):
            return np.fromfile(Path(str(stem)+suffix),'<f4').reshape(-1,4)[:used].reshape(lanes,2,nlights,4).astype(np.float64)
        mean, variance = read('.mean.bin'), read('.variance.bin')
        count, incomplete, nonfinite = read('.valid_count.bin'), read('.incomplete_batches.bin'), read('.nonfinite.bin')
        complete = np.all(np.isfinite(mean)&np.isfinite(variance)&np.isfinite(count)&np.isfinite(incomplete)&np.isfinite(nonfinite),axis=3)
        complete &= np.all(count == 32,axis=3)&np.all(incomplete == 0,axis=3)&np.all(nonfinite == 0,axis=3)
        missing = int((~complete).sum())
        summaries.append(dict(incomplete_lanes=missing,missing_samples_by_channel=[int(np.maximum(32-np.nan_to_num(count[:,:,:,c],nan=0),0).sum()) for c in range(4)],nonfinite_samples_by_channel=[int(np.nan_to_num(nonfinite[:,:,:,c],nan=0).sum()) for c in range(4)]))
        assert missing == 0, summaries[-1]
        # The native accumulator reports sample variance with divisor n-1.
        rgb_m2 = np.mean(variance[:,:,:,:3]*(31/32)+mean[:,:,:,:3]**2,axis=0)
        rgb_mean = np.mean(mean[:,:,:,:3],axis=0)
        y2_mean = np.mean(mean[:,:,:,3],axis=0)
        iid_y2_se = np.sqrt(np.sum(variance[:,:,:,3]/32,axis=0))/lanes
        windows.append(dict(rgb_mean=rgb_mean,rgb_m2=rgb_m2,y2=y2_mean,iid_y2_se=iid_y2_se))
    metadata = [json.loads((directory/f'window{i}.mean.json').read_text())['accumulation'] for i in (1,2)]
    assert all(m['frames']==32 and m['end_frame']-m['start_frame']==31 for m in metadata)
    gap = metadata[1]['start_frame']-metadata[0]['end_frame']-1
    rgb = (windows[0]['rgb_mean']+windows[1]['rgb_mean'])*.5
    rgb_m2 = (windows[0]['rgb_m2']+windows[1]['rgb_m2'])*.5
    y2 = (windows[0]['y2']+windows[1]['y2'])*.5
    ymean = rgb @ LUMA
    empirical_y2_error = np.abs(windows[0]['y2']-windows[1]['y2'])*.5
    ymean_error = np.abs((windows[0]['rgb_mean']-windows[1]['rgb_mean']) @ LUMA)*.5
    qs = np.zeros((6,2,nlights),dtype=float); qzero = np.zeros((6,2),dtype=float)
    key_to_index = {tuple(map(int,row)):i for i,row in enumerate(fixture)}
    for model,name in enumerate(('original','halfspace-a','cosine-b','near-far','moment-sphere','diffuse-sg')):
        _,nodes,links = load_tree(run/'near-cohort/room2'/name/'tree.bin',model)
        for point in range(2):
            dist,zero = distribution(nodes,links,records[point,0,:3],records[point,1,:3],model)
            for node,prob in dist.items():
                key = tuple(map(int,nodes[node,:3]))+(int(links[node]&0x7fffffff),)
                qs[model,point,key_to_index[key]]=prob
            qzero[model,point]=zero
            assert abs(qs[model,point].sum()+zero-1)<1e-8
            assert not np.any((y2[point]>0)&(qs[model,point]<=0)), (name,point)
    ideal_weights = np.sqrt(y2); ideal_weights /= np.maximum(ideal_weights.sum(axis=1,keepdims=True),1e-300)
    np.savez_compressed(run/'point-reference/per-light.npz',fixture=fixture,receiver=records,rgb_mean=rgb,rgb_second_moment=rgb_m2,
                        luminance_mean=ymean,luminance_second_moment=y2,window1_luminance_second_moment=windows[0]['y2'],
                        window2_luminance_second_moment=windows[1]['y2'],luminance_second_moment_half_window_difference=empirical_y2_error,
                        window1_rgb_mean=windows[0]['rgb_mean'],window2_rgb_mean=windows[1]['rgb_mean'],
                        window1_iid_only_y2_standard_error=windows[0]['iid_y2_se'],window2_iid_only_y2_standard_error=windows[1]['iid_y2_se'],
                        luminance_mean_half_window_difference=ymean_error,proposal_probabilities=qs,proposal_zero=qzero,ideal_sqrt_m2_probabilities=ideal_weights)
    result=[]
    mass_by_key = {tuple(map(int,pos))+(int(material),):mass for pos,material,mass in leaf_identity(expected_tree,0)}
    for point,slot in enumerate((54,2)):
        total=float(ymean[point].sum()); moments=y2[point]; ideal=np.sqrt(moments); ideal=ideal/ideal.sum() if ideal.sum() else ideal
        origin=records[point,0,:3].astype(float); normal=records[point,1,:3].astype(float)
        proxy=np.array([cube_integral(row[:3].astype(float),origin,normal)*mass_by_key[tuple(map(int,row))] for row in fixture])
        proxy = proxy/proxy.sum() if proxy.sum() else proxy
        scores=[]
        for model,name in enumerate(('original','halfspace-a','cosine-b','near-far','moment-sphere','diffuse-sg')):
            q=qs[model,point]; positive=moments>0
            second=float(np.sum(moments[positive]/q[positive]));proxy_positive=proxy>0
            proxy_m2=float(np.sum(proxy[proxy_positive]**2/q[proxy_positive])) if np.all(q[proxy_positive]>0) else None
            window_seconds=[float(np.sum(win['y2'][point,positive]/q[positive])) for win in windows]
            window_means=[float((win['rgb_mean'][point] @ LUMA).sum()) for win in windows]
            window_variances=[max(0.,second_value-mean_value**2) for second_value,mean_value in zip(window_seconds,window_means)]
            iid_se=float(np.sqrt(sum(np.sum((win['iid_y2_se'][point,positive]/q[positive])**2) for win in windows))/2)
            scores.append(dict(model=name,zero_probability=float(qzero[model,point]),conditional_second_moment=second,variance_estimate=max(0.,second-total*total),
                               conditional_second_moment_window_values=window_seconds,conditional_second_moment_half_window_difference=abs(window_seconds[0]-window_seconds[1])*.5,
                               variance_window_values=window_variances,variance_half_window_difference=abs(window_variances[0]-window_variances[1])*.5,
                               conditional_second_moment_iid_only_standard_error=iid_se,unoccluded_cube_proxy_second_moment=proxy_m2))
        ideal_second=float(np.sum(moments[ideal>0]/ideal[ideal>0])) if ideal.any() else 0.
        result.append(dict(slot=slot,origin=records[point,0,:3].tolist(),normal=records[point,1,:3].tolist(),total_mean_luminance=total,
                           sum_per_light_absolute_half_window_difference=float(ymean_error[point].sum()),lights_with_observed_nonzero_m2=int(np.sum(moments>0)),
                           lights_with_only_zero_samples=int(np.sum(moments==0)),ideal_sqrt_m2_second_moment=ideal_second,
                           ideal_sqrt_m2_variance_estimate=max(0.,ideal_second-total*total),models=scores,
                           top_lights=sorted([dict(fixture_index=i,position=fixture[i,:3].tolist(),material=int(fixture[i,3]),mean=float(ymean[point,i]),m2=float(moments[i]),m2_half_window_difference=float(empirical_y2_error[point,i])) for i in range(nlights)],key=lambda x:x['m2'],reverse=True)[:20]))
    return dict(revision='f5f59c1b785404555998f1f81ef6406520ffb4ab',fixture_lights=nlights,lanes_per_light_per_frame=lanes,frames_per_window=32,
                samples_per_light_per_window=lanes*32,window_frames=[[m['start_frame'],m['end_frame']] for m in metadata],inter_window_gap_frames=gap,
                zero_observed_iid_only_approx_95_percent_probability_upper=3/(lanes*32*2),
                completeness=summaries,voxel=voxel,receivers=result,per_light_npz=str((run/'point-reference/per-light.npz').resolve()),
                caveats=['Actual fixed-light conditional contribution includes texture and visibility; zeros are retained but cannot be labeled occluded from this output alone.',
                         'Window gaps reflect artifact export latency; each window is consecutive, but the two windows are not adjacent.',
                         'Window differences are empirical uncertainty indicators, not confidence intervals; across-lane and temporal independence is unproven.',
                         'The approximate 3/N zero-hit probability bound assumes IID draws; zero-observed lights are not proven invisible and have no strict upper M2 bound.',
                         'Finite-sample ideal sqrt(M2) is a diagnostic, not an attainable unbiased oracle.',
                         'This fixed fixture diagnostic is not a substitute for the full image matrix.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--cube-points', type=int, default=0)
    parser.add_argument('--model', type=int, default=0, choices=range(6))
    parser.add_argument('--arity', type=int, default=4, choices=(2,4,8))
    parser.add_argument('--target-origin', type=float, nargs=3)
    parser.add_argument('--target-normal', type=float, nargs=3)
    parser.add_argument('--compare-tree', type=Path)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--brdf', action='store_true')
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--point-reference', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (args.target_origin is None) == (args.target_normal is None)
    if args.point_reference:
        result = point_reference(args.directory)
    elif args.report:
        result = report(args.directory)
    elif args.reference:
        result = reference_review(args.directory)
    elif args.timing:
        result = timing_review(args.directory)
    elif args.brdf:
        header, _, _ = load_tree(args.directory / 'tree.bin', 0)
        reasons = np.fromfile(args.directory / 'audit.bin', '<u4')[1024:1040]
        assert header[1] > 0 and header[2] == header[3] == 0
        assert sum(map(int, reasons[:10])) == int(header[6]) and reasons[10] == 0
        result = dict(receivers=int(header[6]), reasons=list(map(int, reasons)))
    else:
        result = quality_review(args.directory) if args.quality else (baseline_review(args.directory, args.cube_points, args.target_origin, args.target_normal) if args.model == 0 else audit_capture(args.directory, args.model, args.arity))
    if args.compare_tree:
        actual = leaf_identity(args.directory / 'tree.bin', args.model)
        expected = leaf_identity(args.compare_tree, 0)
        assert actual == expected, f'Leaf identity mismatch: actual={len(actual)}, expected={len(expected)}'
        assert topology_identity(args.directory / 'tree.bin', args.model) == topology_identity(args.compare_tree, 0), 'Topology mismatch'
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

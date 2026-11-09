"""Red-pillar moment decomposition and archived voxel bounds, without a new Iris capture."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from numpy.polynomial.legendre import leggauss

from light_sampling_clustering_hierarchy import brick_roots
from light_sampling_clustering_repair import EVIDENCE, build_primitives, control_root, score
from light_sampling_clustering_rule_screen import RULES
from light_sampling_geometry import cube_faces
from light_sampling_moving_analysis import morton_encode
from light_sampling_point_moments import read, sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import probabilities

RUN = EVIDENCE.parents[1] / 'builds/light-sampling/20261108-clustering-repair'
RAW = RUN / 'cases/light-sampling-1/m2-1f1d84ea'
SOURCE = EVIDENCE / '20261108-clustering-repair-reference.npz'
KEYS = ((154,146,78,10254), (150,146,78,10254), (157,145,76,682), (146,153,88,682))
METHODS = ('prefix', 'upper/flux-diagonal', 'flat')
EXPECTED_LEAVES = 'ce3347600e2d804aeaea14082eeb84a971dce2f2154797c7f1ba52f0795d0631'
EXPECTED_VOXEL = '4dfca2b6ba4714126b4962a206a4bda14e358bf9a33eaa67ac85bfb99e9d61df'


def signed(values):
    return dict(positive=float(values[values > 0].sum()), negative=float(values[values < 0].sum()),
                net=float(values.sum()))


def inv(p):
    return np.divide(1., p, out=np.zeros_like(p), where=p > 0)


def split_moments(moments, probabilities, cross_mean2):
    mu = moments[:, :3]
    m2 = moments[:, 3]
    mean2 = np.einsum('ij,ij->i', mu, mu)
    mean_sum2 = float(mu.sum(axis=0) @ mu.sum(axis=0))
    assert np.all(m2-mean2 >= -1e-12)
    inverses = [inv(p) for p in probabilities]
    matrices = {}
    results = {}
    for name, p, inverse in zip(METHODS, probabilities, inverses, strict=True):
        assert not np.any((m2 > 0) & (p == 0))
        difference = inverse - inverses[0]
        mean_delta, residual_delta = mean2 * difference, (m2-mean2) * difference
        cross_delta = cross_mean2 * difference
        total = m2 * difference
        np.testing.assert_allclose(mean_delta+residual_delta, total, rtol=1e-12, atol=1e-15)
        variance = float(m2 @ inverse - mean_sum2)
        between, within = float(mean2 @ inverse - mean_sum2), float((m2-mean2) @ inverse)
        np.testing.assert_allclose(between+within, variance, rtol=1e-12, atol=1e-15)
        matrices[name] = np.column_stack((mean_delta, residual_delta, total, cross_delta, total-cross_delta))
        results[name] = dict(
            variance=variance, mean_only_variance=between, within_lamp_variance=within,
            variance_over_prefix=variance/results['prefix']['variance'] if name != 'prefix' else 1.,
            delta={field: signed(matrix) for field, matrix in zip(
                ('mean', 'within', 'total', 'cross_mean', 'cross_within'), matrices[name].T, strict=True)},
        )
    return mean2, matrices, results


class ArchivedVoxels:
    """Archived packed occupancy and block quads, evaluated by an independent CPU DDA."""
    def __init__(self, accepted, lamp_keys):
        meta = np.fromfile(RAW / 'before_voxel-meta.bin', '<u4')
        self.materials = np.memmap(RAW / 'before_voxel-materials.bin', '<u4', mode='r')
        self.grid = 16
        self.origin = (np.array(accepted['pose']['camera_brick']) - self.grid // 2) * 16
        count = int(meta[0])
        slots = self.grid**3
        bricks = meta[1 + 2*slots + 1024:1 + 2*slots + 1024 + count]
        self.allocation = np.full(slots, -1, dtype=int)
        self.allocation[bricks] = np.arange(count)
        np.testing.assert_array_equal(meta[1+slots+bricks], np.arange(count))
        self.emitter_materials = np.unique(lamp_keys[:, 3])
        textures = RUN/'m2-source/shaders/textures'
        self.model_files = [textures/'pbr_lut_2.bin', textures/'block_model_quads.bin']
        self.models = np.memmap(self.model_files[0], '<u4', mode='r', shape=(12741,64))
        self.quads = np.memmap(self.model_files[1], 'u1', mode='r', shape=(4,32768,4))

    def model_hits(self, materials, cells, origin, directions):
        mask = 42 - (directions < 0) @ np.array([1,4,16])
        found = np.zeros(len(cells), dtype=bool)
        unknown = materials >= len(self.models)
        ids = np.flatnonzero(~unknown & (materials != 1))
        metadata = self.models[materials[ids], mask[ids]]
        for code in np.unique(metadata[metadata != 0]):
            group = ids[metadata == code]
            word = int(code)
            transform = np.array([word,word >> 3,word >> 6])
            axis = transform & 3
            assert sorted(axis.tolist()) == [0,1,2]
            signs = np.where(transform & 4,-1.,1.)
            local = (origin-cells[group]-.5)[:,axis]*signs+.5
            direction = directions[group][:,axis]*signs
            first = ((word >> 9) & 0x3fff) * 2
            row = (word >> 23) & 3
            count = (word >> 25) & 63
            texels = self.quads[row,first:first+2*count].astype(float).reshape(count,2,4)/255.
            for a,b in texels:
                offset = a[:3]-local
                if word & 0x80000000:
                    orthogonal = int(round(a[3]*255))-253
                    assert 0 <= orthogonal <= 2
                    n = np.eye(3)[orthogonal]
                    basis = np.eye(3)[np.arange(3) != orthogonal]
                else:
                    n = np.array([a[3],b[0],b[1]])*(255./126.)-1.
                    reference = [0,0,1] if abs(n[2]) < .999 else [0,1,0]
                    u = np.cross(reference,n)
                    basis = np.array([u,np.cross(n,u)])
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = (offset @ n)/(direction @ n)
                    uv = np.abs((direction*t[:,None]-offset) @ basis.T)
                intersects = (t >= 0) & np.all(uv <= b[2:]+1e-6,axis=1)
                found[group] |= intersects & (t < 1.)
                # A partial model beyond the segment is an ambiguous CPU replay.
                unknown[group] |= intersects & (t >= 1.)
        return found, unknown

    def lookup(self, cells):
        local = cells - self.origin
        inside = ((local >= 0) & (local < self.grid*16)).all(axis=1)
        values = np.zeros(len(cells), dtype='<u4')
        ids = np.flatnonzero(inside)
        allocated = self.allocation[morton_encode(local[ids] // 16)]
        ids, allocated = ids[allocated >= 0], allocated[allocated >= 0]
        values[ids] = self.materials[allocated*4096 + morton_encode(local[ids] % 16)]
        return values, ~inside

    def occlusion(self, origin, points):
        directions = points - origin
        steps = np.sign(directions).astype(int)
        cells = np.broadcast_to(np.floor(origin).astype(int), directions.shape).copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            boundary = cells + (steps > 0)
            crossing = np.where(steps != 0, (boundary-origin)/directions, np.inf)
            interval = np.where(steps != 0, 1/np.abs(directions), np.inf)
        size = len(points)
        active = np.ones(size, bool)
        uncertain = np.zeros(size, bool)
        blocked = np.zeros(size, bool)
        blocker = np.zeros(size, dtype='<u4')
        blocker_cell = np.zeros((size, 3), dtype=int)
        for _ in range(3*self.grid*16 + 3):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            word, outside = self.lookup(cells[ids])
            # The shader treats the fixture's emitting materials as cubes.
            hit = ((word & 1) != 0) | np.isin(word >> 1, self.emitter_materials)
            candidate = (word != 0) & ~hit
            hits, model_unknown = self.model_hits(word[candidate] >> 1, cells[ids[candidate]],
                                                  origin, directions[ids[candidate]])
            hit[candidate] = hits
            uncertain[ids[candidate]] |= model_unknown
            if np.any(hit):
                hit_ids = ids[hit]
                blocked[hit_ids] = True
                blocker[hit_ids] = word[hit]
                blocker_cell[hit_ids] = cells[hit_ids]
            uncertain[ids] |= outside
            axis = crossing[ids].argmin(axis=1)
            t = crossing[ids, axis]
            # Trace open segment only. Target faces are at t=1.
            done = hit | (t >= 1.-1e-12)
            active[ids[done]] = False
            moving, axis = ids[~done], axis[~done]
            cells[moving, axis] += steps[moving, axis]
            crossing[moving, axis] += interval[moving, axis]
        assert not active.any()
        return blocked, uncertain & ~blocked, blocker, blocker_cell


def quadrature(low, origin, normal, voxels, order):
    """Independent area integral. Visibility is bounded on these quadrature rays."""
    # Receiver normals in the two audited pixels are axis aligned. For these
    # voxel-aligned faces, integrating the full square with max(cos,0) is exact
    # except for quadrature error and the visibility discontinuities.
    x, w = leggauss(order)
    u, v = np.meshgrid((x+1)*.5, (x+1)*.5, indexing='ij')
    weights = np.outer(w*.5, w*.5).ravel()
    out = []
    faces = cube_faces(low.copy(), origin, normal)
    weight_total = sum(f['weight'] for f in faces)
    for face in faces:
        axis = face['axis']
        a, b = [k for k in range(3) if k != axis]
        points = np.broadcast_to(low.astype(float), (order**2, 3)).copy()
        points[:, axis] += origin[axis] > low[axis]
        points[:, a] += u.ravel()
        points[:, b] += v.ravel()
        delta = points-origin
        d2 = np.einsum('ij,ij->i', delta, delta)
        area_to_solid_angle = np.abs(delta[:, axis]) / d2**1.5
        projected = np.maximum(delta @ normal, 0.) / np.sqrt(d2)
        measure = weights * area_to_solid_angle * projected
        blocked, unknown, blocker, blocker_cell = voxels.occlusion(origin, points)
        face_probability = face['weight'] / weight_total if weight_total else 0.
        active = (projected > 0) & (face_probability > 0)
        pdf = np.zeros_like(projected)
        if face['tiny']:
            pdf[active] = face_probability / (face['area']*area_to_solid_angle[active])
        else:
            pdf[active] = face_probability * projected[active] / face['h']
        assert np.isfinite(pdf).all() and np.all(pdf[active] > 0)
        proposal_weight = weights * area_to_solid_angle * pdf
        geometry = projected * inv(pdf) / np.pi
        supported = measure[active].sum()
        np.testing.assert_allclose(supported, measure.sum(), rtol=1e-10, atol=1e-14)
        geometry_m2 = proposal_weight * geometry**2
        # Interior quadrature points on the clipping boundary have zero weight.
        cell = Counter(tuple(map(int, [*pos, word >> 1]))
                       for pos, word, amount in zip(blocker_cell, blocker, measure, strict=True)
                       if word and amount > 0)
        out.append(dict(
            **dict(face, tiny=bool(face['tiny'])), projected_integral=float(measure.sum()),
            face_probability=float(face_probability), quadrature_pdf_integral=float(proposal_weight.sum()),
            geometry_m2_unoccluded=float(geometry_m2.sum()),
            geometry_m2_visible_bounds=[float(geometry_m2[~(blocked|unknown)].sum()),
                                        float(geometry_m2[~blocked].sum())],
            blocked_integral=float(measure[blocked].sum()),
            unknown_integral=float(measure[unknown].sum()),
            blocker_counts=[dict(key=k, rays=n) for k, n in cell.most_common(4)],
        ))
    total = sum(f['projected_integral'] for f in out)
    blocked = sum(f['blocked_integral'] for f in out)
    unknown = sum(f['unknown_integral'] for f in out)
    analytic = sum(f['h'] for f in out)
    if total == 0:
        assert analytic == 0.
        visibility = None
    else:
        np.testing.assert_allclose(total, analytic, rtol=5e-3, atol=5e-8)
        visibility = [max(0., 1.-(blocked+unknown)/total), max(0., 1.-blocked/total)]
    # In the ordinary face mode p(face)=H_face/sum(H), p(w|face)=cos/H_face.
    # The shader's cosine/(pi*p(w)) is then constant over the entire lamp.
    constant_geometry = total > 0 and all(not f['tiny'] for f in out)
    pdf_integral = sum(f['quadrature_pdf_integral'] for f in out)
    if total > 0:
        np.testing.assert_allclose(pdf_integral, 1., rtol=0, atol=.02)
    unoccluded_m2 = sum(f['geometry_m2_unoccluded'] for f in out)
    visible_m2 = np.sum([f['geometry_m2_visible_bounds'] for f in out], axis=0).tolist() if out else [0.,0.]
    if constant_geometry:
        np.testing.assert_allclose(unoccluded_m2, (analytic/np.pi)**2, rtol=5e-3, atol=5e-12)
    return dict(order=order, faces=out, projected_integral=total, analytic_integral=analytic,
                quadrature_pdf_integral=float(pdf_integral), geometry_m2_unoccluded=unoccluded_m2,
                geometry_m2_visible_bounds=visible_m2,
                geometry_variance_unoccluded=unoccluded_m2-(total/np.pi)**2,
                visibility_interval=visibility, ordinary_face_mode=constant_geometry,
                ideal_geometry_factor=analytic/np.pi if constant_geometry else None)


def summarize_samples(samples):
    assert np.isfinite(samples).all()
    squared = np.einsum('ij,ij->i', samples[:, :3], samples[:, :3])
    np.testing.assert_allclose(squared, samples[:, 3], rtol=4e-7, atol=1e-15)
    directions = samples[squared > 0, :3] / np.sqrt(squared[squared > 0, None])
    color_span = np.diff(np.quantile(directions, [.05,.95], axis=0), axis=0)[0].tolist() if len(directions) else None
    ordered = np.sort(squared)
    upper_count = max(1, int(np.ceil(len(ordered)*.05)))
    return dict(samples=len(samples), zero_rgb=int(np.count_nonzero(squared == 0)),
                normalized_rgb_p95_minus_p05=color_span,
                squared_rgb_quantiles=np.quantile(squared,[.5,.9,.95,.99,1.]).tolist(),
                largest_5_percent_m2_share=float(ordered[-upper_count:].sum()/ordered.sum())
                if ordered.sum() else None)


def choose_controls(keys, point, moments, pmfs, matrices, reasons):
    def add(index, description):
        reasons.setdefault(int(index), []).append(f'{point}: {description}')
    mean_m2 = np.mean([m[:, 3] for m in moments], axis=0)
    for name, p in zip(METHODS[1:], pmfs[1:], strict=True):
        gain = p-pmfs[0]
        delta = np.mean([m[name][:, 2] for m in matrices], axis=0)
        add(np.argmax(delta), name+' highest added M2/p')
        add(np.argmin(delta), name+' highest reduced M2/p')
        zero = (mean_m2 == 0) & (gain > 0)
        if np.any(zero):
            add(np.argmax(np.where(zero, gain, -np.inf)), name+' largest probability gain among zero-observed M2')
    if point == 'left-red-pillar':
        for primary in KEYS[:3]:
            same = (keys[:, 3] == primary[3]) & np.any(keys[:, :3] != primary[:3], axis=1)
            distance2 = ((keys[:, :3] - primary[:3])**2).sum(axis=1)
            near = np.argsort(np.where(same, distance2, np.inf), kind='stable')
            add(near[0], 'nearest same-material comparison to '+str(primary))


def neutral_receiver_output(audit, settings):
    """One achromatic unit of mass at the archived gain, through receiver albedo."""
    assert settings['SETTING_WORKING_COLOR_SPACE'] == '8'
    source = RUN/'m2-source/shaders/util/colors/ColorSpaces.glsl'
    contents = source.read_text(encoding='utf8')
    body = re.search(r'_mat_COLOR_MCSPACEFACE_to_SRGB = mat3\(([^)]+)\)', contents).group(1)
    # GLSL lists matrix columns. The shader multiplies the RGB row vector on the left.
    columns = np.fromstring(body.replace(',', ' '), sep=' ').reshape(3,3)
    gain = .5 * 2**float(settings['SETTING_EMISSIVE_STRENGTH'])
    return (columns @ audit[:,2,:3].T).T * gain, sha(source)


def neutral_moments(fine, rgb):
    integral = fine['projected_integral']
    visible = fine['visibility_interval'] or [0.,0.]
    m2_scale = float(rgb @ rgb)
    return dict(no_blockers=dict(mean_rgb=(rgb*integral/np.pi).tolist(),
                                 m2=m2_scale*fine['geometry_m2_unoccluded']),
                known_blockers=dict(mean_rgb_at_visibility_bounds=[(rgb*integral*v/np.pi).tolist()
                                                                    for v in visible],
                                    m2_interval=(m2_scale*np.array(fine['geometry_m2_visible_bounds'])).tolist()))


def neutral_delta(lights):
    """Covered lamps only. For proposal differences the common mean term cancels."""
    output = {}
    for method in METHODS[1:]:
        no_blockers = 0.
        with_blockers = np.zeros(2)
        for light in lights:
            q0, q1 = light['pmf']['prefix'], light['pmf'][method]
            control = light['offline_neutral']
            if min(q0, q1) == 0:
                assert control['no_blockers']['m2'] <= 1e-15
                continue
            difference = 1/q1-1/q0
            no_blockers += difference*control['no_blockers']['m2']
            with_blockers += sorted(difference*np.asarray(control['known_blockers']['m2_interval']))
        output[method] = dict(no_blockers_m2_over_q_delta=no_blockers,
                              with_blockers_m2_over_q_delta_interval=with_blockers.tolist())
    return output


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    assert sha(SOURCE) == '3317ac116bc89c055528f9fb8038fbf7bd791ffe27821aced8bd75a9d92167e0'
    accepted = read(RAW/'accepted.json')
    manifest = read(RAW/'manifest.json')
    assert sha(RAW/'manifest.json') == accepted['manifest_sha256']
    artifacts = {item['relative_path']: item['sha256'] for item in manifest['files']}
    originals = {name: sha(RAW/name) for name in ('before_voxel-materials.bin', 'before_voxel-meta.bin',
                                                  'before_tree.bin', 'audit.bin', 'single.bin', 'single.json')}
    assert all(digest == artifacts[name] for name, digest in originals.items())
    assert accepted['identity']['world_voxel_sha256'] == EXPECTED_VOXEL
    assert accepted['identity']['leaf_sha256'] == EXPECTED_LEAVES
    assert accepted['settings']['SETTING_DEBUG_LIGHT_CUBE'] == 'true'
    old = read(EVIDENCE/'20261108-clustering-repair.json')
    previous = {(r['point'], r['window']): r for r in old['clustering_rule_screen']['windows'] if r['cohort']=='new/1'}
    audited_flat = {(r['point'], r['window']): r for r in read(RUN/'review/flat-reference-independent.json')['windows']
                    if r['cohort']=='new/1'}
    with np.load(SOURCE) as ref:
        cohort = json.loads(str(ref['metadata']))['1']
        assert cohort['identity'] == accepted['identity']
        keys = ref['1/keys']
        root = control_root(ref, '1/prefix/')
        inputs = {(w['point'], w['window']): (w, ref[f'1/{i}/moments'])
                  for i,w in enumerate(cohort['windows'])}
    all_keys = [tuple(map(int, key)) for key in keys]
    atoms = brick_roots(root)
    upper_root = build_primitives(np.asarray(list(atoms)), list(atoms.values()), True, partition=RULES['flux-diagonal'])
    rebuilt_atoms = brick_roots(upper_root)
    assert rebuilt_atoms.keys() == atoms.keys()
    assert all(rebuilt_atoms[k] is v for k,v in atoms.items())
    trees = {'prefix': Tree(root), 'upper/flux-diagonal': Tree(upper_root)}
    voxels = ArchivedVoxels(accepted, keys)
    packed, outside = voxels.lookup(keys[:, :3])
    assert not outside.any()
    np.testing.assert_array_equal(packed >> 1, keys[:, 3])
    audit = np.fromfile(RAW/'audit.bin', '<f4')[:32].reshape(2,4,4)
    receiver_neutral, color_source_sha = neutral_receiver_output(audit, accepted['settings'])
    reasons = {all_keys.index(k): ['named red-pillar reference'] for k in KEYS}
    analysis = {}
    raw_arrays = dict(world_keys=keys)
    moment_hashes = {}
    for point, entries in ((p, sorted((i, data) for (name,i),data in inputs.items() if name==p))
                           for p in dict.fromkeys(p for p,_ in inputs)):
        moments = [data[1] for _, data in entries]
        (origin, normal) = np.asarray(entries[0][1][0]['receiver_audit'])[:2, :3]
        cross = (moments[0][:,:3] * moments[1][:,:3]).sum(axis=1)
        evals = {n: score(t, all_keys, origin, normal, moments[0]) for n,t in trees.items()}
        node_ids = evals['prefix'][-1]
        leaf_data = trees['prefix'].nodes[node_ids]
        flat = probabilities(leaf_data, 6, origin, normal)
        half = (leaf_data[:,4:7]-leaf_data[:,:3])*.5
        delta = leaf_data[:,:3]+half-origin
        h = delta @ normal
        extent = half @ np.abs(normal)
        amplitude = leaf_data[:,3] * np.clip((h+np.linalg.norm(half,axis=1)) /
                    np.sqrt(np.maximum((delta*delta).sum(1),.25)), .001, 1)
        amplitude[h+extent < -1e-6*((np.abs(delta*normal)).sum(1)+extent+1)] = 0
        near2 = (np.maximum(np.abs(delta)-half,0)**2).sum(1)
        far2 = ((np.abs(delta)+half)**2).sum(1)
        inside = (near2==0) & (amplitude>0)
        near = np.where(inside, amplitude, 0) if inside.any() else amplitude*inv(near2)
        far = amplitude*inv(far2)
        flat_halves = .5 * np.column_stack((near/near.sum(),far/far.sum()))
        np.testing.assert_allclose(flat, flat_halves.sum(axis=1), rtol=1e-13, atol=1e-16)
        pmfs = [*(evals[n][1] for n in trees), flat]
        np.testing.assert_allclose([p.sum() for p in pmfs], 1., rtol=0, atol=2e-12)
        result = []
        matrix_list = []
        raw_arrays[point+'/pmfs'] = np.column_stack(pmfs)
        raw_arrays[point+'/flat_halves'] = flat_halves
        raw_arrays['mass'] = leaf_data[:, 3]
        raw_arrays[point+'/cross_mean2'] = cross
        for (window,(metadata, observed)) in entries:
            assert metadata['receiver_audit'] == entries[0][1][0]['receiver_audit']
            mfile = RAW/'moments'/f'{point}-window{window}.npz'
            moment_hashes[mfile.name] = sha(mfile)
            with np.load(mfile) as m:
                np.testing.assert_array_equal(observed, np.c_[m['mean_rgb'],m['m2']])
                np.testing.assert_array_equal(m['complete_samples'], m['potential_samples'])
                assert np.all(m['nonfinite_samples']==0) and np.all(m['incomplete_pixel_groups']==0)
                counts = m['complete_samples']
            mean2, matrices, summarized = split_moments(observed, pmfs, cross)
            window_id = point, window
            for n in trees:
                expected = previous[window_id]['methods'][n][old['clustering_rule_screen']['method_columns'].index('variance')]
                np.testing.assert_allclose(summarized[n]['variance'], expected, rtol=1e-12)
            np.testing.assert_allclose(summarized['flat']['variance'], audited_flat[window_id]['flat_variance'], rtol=1e-12)
            raw_arrays[f'{point}/{window}/moments'] = observed
            raw_arrays[f'{point}/{window}/complete_samples'] = counts
            raw_arrays[f'{point}/{window}/mean2_within_optimal'] = np.c_[
                mean2, observed[:,3]-mean2, np.sqrt(observed[:,3])/np.sqrt(observed[:,3]).sum()]
            for n,m in matrices.items():
                raw_arrays[f'{point}/{window}/{n}/delta'] = m
            matrix_list.append(matrices)
            result.append(dict(window=window, methods=summarized, samples_per_light=int(counts[0])))
        choose_controls(keys, point, moments, pmfs, matrix_list, reasons)
        analysis[point] = dict(entries=entries, pmfs=pmfs, evals=evals, matrices=matrix_list,
                               windows=result, receiver=entries[0][1][0]['receiver_audit'],
                               origin=origin, normal=normal, flat_halves=flat_halves)

    metadata = read(RAW/'single.json')
    width, height = metadata['width'], metadata['height']
    assert (width, height) == (1920,1080)
    n_samples = width * height // (len(keys)*2)
    assert np.all(audit[:, 1, 3] == n_samples)
    rgb_frame = np.memmap(RAW/'single.bin', '<f4', mode='r', shape=(width*height,4))
    rgb_frame = rgb_frame[:n_samples*len(keys)*2].reshape(n_samples,2,len(keys),4)
    point_ids = [r['id'] for r in accepted['fixture']['receivers']]
    output = {}
    ids = sorted(reasons, key=lambda i: all_keys[i])
    for point, a in analysis.items():
        lights = []
        for i in ids:
            low = keys[i,:3]
            fine = quadrature(low, a['origin'], a['normal'], voxels, 64)
            coarse = quadrature(low, a['origin'], a['normal'], voxels, 32)
            detailed = []
            for j, (window,(_, m)) in enumerate(a['entries']):
                mu2 = float(m[i,:3] @ m[i,:3])
                # For ordinary faces the constant geometry factor lets the
                # observed M2 bound receiver-weighted RMS of arrived emission.
                # No claim about hidden face texels or direct atlas radiance.
                geom = fine['ideal_geometry_factor']
                visibility = fine['visibility_interval']
                effective = None
                visible_variance = None
                non_visibility_variance = None
                if geom and visibility and visibility[1] > 0:
                    scale = geom * raw_arrays['mass'][i]
                    effective = [float(np.sqrt(m[i,3]/p)/scale) if p > 0 else None for p in reversed(visibility)]
                    visible_variance = [mu2*(1/p-1) if p > 0 else None for p in reversed(visibility)]
                    non_visibility_variance = [float(m[i,3]-mu2/p) if p > 0 else None for p in visibility]
                detailed.append(dict(window=window, mean_rgb=m[i,:3].tolist(), m2=float(m[i,3]),
                    within_lamp=float(m[i,3]-mu2), lamp_optimal_pmf=float(np.sqrt(m[i,3])/np.sqrt(m[:,3]).sum()),
                    output_rgb_effective_rms_per_mass_interval=effective,
                    conditional_visibility_variance_interval=visible_variance,
                    conditional_remaining_variance_interval=non_visibility_variance,
                    deltas={n:a['matrices'][j][n][i].tolist() for n in METHODS}))
            paths = {}
            for n, e in a['evals'].items():
                path = []
                node = int(e[-1][i])
                while trees[n].parents[node] >= 0:
                    path.append(dict(node=node, near_half=float(e[4][node,0]), far_half=float(e[4][node,1])))
                    node = int(trees[n].parents[node])
                paths[n] = list(reversed(path))
            point_index = point_ids.index(point)
            neutral = neutral_moments(fine, receiver_neutral[point_index] * raw_arrays['mass'][i])
            lights.append(dict(key=all_keys[i], material=int(keys[i,3]), mass=float(raw_arrays['mass'][i]),
                               selection_reasons=reasons[i], pmf=dict(zip(METHODS, [float(p[i]) for p in a['pmfs']])),
                               flat_near_far_halves=a['flat_halves'][i].tolist(), paths=paths,
                               single_frame=summarize_samples(rgb_frame[:,point_index,i]), offline_neutral=neutral,
                               quadrature=fine, coarse_visibility_interval=coarse['visibility_interval'], windows=detailed))
        for window, matrices in zip(a['windows'], a['matrices'], strict=True):
            window['selected_lamp_coverage'] = {
                n: dict(selected=signed(m[ids,2]), unselected=signed(np.delete(m[:,2],ids)),
                        cumulative_share_of_positive_delta=np.cumsum(np.sort(m[m[:,2]>0,2])[::-1])[:8].tolist())
                for n,m in matrices.items() if n!='prefix'}
            for n, c in window['selected_lamp_coverage'].items():
                positive = window['methods'][n]['delta']['total']['positive']
                c['cumulative_share_of_positive_delta'] = [x/positive for x in c['cumulative_share_of_positive_delta']] if positive else []
        output[point] = dict(receiver=a['receiver'], windows=a['windows'], selected_lights=lights,
                             covered_neutral_deltas=neutral_delta(lights))
    raw_arrays['selected_ids'] = ids
    raw_arrays['single_frame_selected'] = rgb_frame[:,:,ids]
    np.savez_compressed(out/'per-lamp-moments.npz', **raw_arrays)
    shader_source = RUN/'m2-source/shaders'
    dependencies = ('pass/composite/LightSamplingInitial.comp.glsl', 'pass/prepare/LightSamplingBuild.comp.glsl',
                    'techniques/debug/LightSamplingCube.glsl', 'techniques/voxel/VoxelTrace.glsl',
                    'techniques/voxel/BlockModels.glsl', 'techniques/voxel/SurfaceData.glsl',
                    'util/Material.glsl', 'shaders.properties')
    report = dict(
        fixture_source_file_sha256={p:sha(shader_source/p) for p in dependencies},
        scope='Existing 2x32-frame Mode 3 moments plus independent deterministic area quadrature over archived cubes and block models.',
        fixture_revision=accepted['provenance']['resolved_revision'], fixture=accepted['provenance'], single_frame_id=metadata['frame_id'],
        identities=accepted['identity'], original_sha256=originals, source_sha256=sha(SOURCE),
        moment_file_sha256=moment_hashes, color_source_sha256=color_source_sha,
        model_resource_sha256={p.name:sha(p) for p in voxels.model_files},
        neutral_reference='Achromatic (1,1,1) in working space times mass * 0.5 * 2^EMISSIVE_STRENGTH, with audited receiver albedo and the original linear sRGB output transform.',
 script_sha256=sha(Path(__file__)), raw_audit=audit.tolist(),
        settings={k:v for k,v in accepted['settings'].items() if any(term in k for term in
                  ('PBR', 'EMISSIVE', 'LIGHT_', 'VOXEL_'))}, arrays_sha256=sha(out/'per-lamp-moments.npz'),
        delta_columns=['mean','within','total','cross_mean','cross_within'],
        coverage='Selected lights are shared across both receivers. Geometry/visibility fractions bound sampled quadrature rays only; The CPU DDA tests packed partial-model quads and marks out-of-grid/out-of-table/beyond-target cases unknown. Quadrature and GPU float/trace differences remain.',
        limits='Cube texture/specular atlas, hit reason, original per-sample direction and executed PDF were unavailable. There is no paired 2x2 capture. Effective output RGB includes receiving albedo and color conversion; conditional estimates only.',
        points=output)
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)

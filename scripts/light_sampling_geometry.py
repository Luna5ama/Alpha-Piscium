"""Analysis and independent PDF checks for the visibility-free geometry matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
from light_sampling_runtime import preserve, read_image, render, tree_stats, LUMA

def cube_faces(low, origin, normal):
    faces = []
    for axis in range(3):
        n = np.eye(3)[axis]
        o = origin[axis]
        if low[axis] <= o <= low[axis] + 1:
            continue
        center = low + 0.5
        center[axis] = low[axis] + (o > low[axis])
        u = np.roll(n, 1) * 0.5
        v = np.roll(n, 2) * (0.5 if o < low[axis] else -0.5)
        vertices = [center - u - v - origin, center + u - v - origin, center + u + v - origin, center - u + v - origin]
        clipped = []
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            za, zb = (a @ normal, b @ normal)
            if za >= 0:
                clipped.append(a)
            if za * zb < 0:
                clipped.append(a + (b - a) * za / (za - zb))
        if len(clipped) < 3:
            continue
        vertices = np.array(clipped)
        directions = vertices / np.linalg.norm(vertices, axis=1)[:, None]
        h = 0.0
        for a, b in zip(directions, np.roll(directions, -1, axis=0)):
            cross = np.cross(a, b)
            sine = np.linalg.norm(cross)
            if sine:
                h += 0.5 * (cross @ normal) * np.arctan2(sine, a @ b) / sine
        area = 0.0
        area_center = np.zeros(3)
        for i in range(1, len(vertices) - 1):
            a, b, c = (vertices[0], vertices[i], vertices[i + 1])
            triangle_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            area += triangle_area
            area_center += triangle_area * (a + b + c) / 3
        if area == 0:
            continue
        area_center /= area
        h = max(h, 0.0)
        tiny = h < 1e-05
        d2 = area_center @ area_center
        weight = area * max(area_center @ normal, 0) * abs(n @ area_center) / d2 ** 2 if tiny else h
        faces.append(dict(axis=axis, h=h, weight=weight, area=area, tiny=tiny))
    return faces

def cube_integral(low, origin, normal):
    return sum((f['h'] for f in cube_faces(low, origin, normal)))

def cube_pdf(low, origin, normal, direction):
    faces = cube_faces(low, origin, normal)
    entry = np.minimum((low - origin) / direction, (low + 1 - origin) / direction)
    axis = int(np.argmax(entry))
    distance = entry[axis]
    face = next((f for f in faces if f['axis'] == axis))
    conditional = distance ** 2 / (face['area'] * abs(direction[axis])) if face['tiny'] else normal @ direction / face['h']
    return face['weight'] / sum((f['weight'] for f in faces)) * conditional

def audit_cube(path):
    data = np.fromfile(path, '<f4').reshape(64, 4, 4).astype(float)
    errors = []
    for origin4, normal4, direction4, low4 in data:
        if direction4[3] <= 0:
            continue
        origin, normal, direction, low = (origin4[:3], normal4[:3], direction4[:3], low4[:3])
        assert np.isfinite(direction4).all()
        assert abs(np.linalg.norm(direction) - 1) < 2e-05
        expected = cube_pdf(low.copy(), origin, normal, direction)
        errors.append(abs(direction4[3] / expected - 1))
        t0, t1 = ((low - origin) / direction, (low + 1 - origin) / direction)
        assert np.max(np.minimum(t0, t1)) <= np.min(np.maximum(t0, t1)) + 0.0001
    assert errors and max(errors) < 0.005, errors
    return dict(samples=len(errors), max_relative_pdf_error=max(errors), p95_relative_pdf_error=float(np.quantile(errors, 0.95)))

def audit_tree(tree_path, audit_path, improved, lookahead):
    words = np.fromfile(tree_path, '<u4')[8:].reshape(-1, 8)
    nodes = words.view('<f4').astype(float)
    parents, leaves = ({}, {})
    pending = [1]
    while pending:
        index = pending.pop()
        if nodes[index, 3] == 0:
            continue
        link = int(words[index, 7])
        if link & 2147483648:
            leaves[tuple(nodes[index, :3])] = index
        else:
            parents[link] = parents[link + 1] = index
            pending.extend((link, link + 1))
    records = np.fromfile(audit_path, '<f4').reshape(64, 4, 4).astype(float)
    errors = []
    for record in records:
        if record[3, 3] == 0:
            continue
        origin, normal = (record[0, :3], record[1, :3])

        def score(index, depth):
            node = nodes[index]
            mass = node[3]
            link = int(words[index, 7])
            if mass == 0:
                return 0.0
            if depth and (not link & 2147483648):
                return score(link, depth - 1) + score(link + 1, depth - 1)
            half = (node[4:7] - node[:3]) * 0.5
            delta = node[:3] + half - origin
            d2 = delta @ delta
            r2 = half @ half
            if improved:
                e = abs(normal) @ half
                h = normal @ delta
                h = h if h >= e else 0 if h <= -e else (h + e) ** 2 / (4 * e)
                scale = d2 + r2 / 3
                return mass * max(np.clip(h / np.sqrt(scale), 0, 1), 0.001) / scale
            cosine = np.clip((normal @ delta + np.sqrt(r2)) / np.sqrt(max(d2, 0.25)), 0.001, 1)
            return mass * cosine / max(d2, r2, 0.25)
        index = leaves[tuple(record[3, :3])]
        pdf = 1.0
        while index != 1:
            parent = parents[index]
            left = int(words[parent, 7])
            depth = 2 if lookahead else 0
            a, b = (score(left, depth), score(left + 1, depth))
            pdf *= a / (a + b) if index == left else b / (a + b)
            index = parent
        errors.append(abs(record[1, 3] / pdf - 1))
    assert errors and max(errors) < 0.005, errors
    return dict(samples=len(errors), max_relative_pdf_error=max(errors))

def analyze(directory, selected_scene=None):
    from scipy.ndimage import gaussian_filter
    spec = json.loads((directory / 'matrix.json').read_text(encoding='utf-8-sig'))
    rows = []
    images = []
    sources = set()
    for scene in spec['scenes']:
        if selected_scene and selected_scene != scene['id']:
            continue
        captured = []
        scene_hashes = set()
        mask = None
        leaf_set = None
        for case in spec['cases']:
            raw = directory / 'raw' / scene['id'] / case['id']
            job = json.loads((raw / 'job-result.json').read_text())['result']
            mutation = job['action_receipts'][0]['runtime_mutation']
            assert job['provenance']['preset_id'] == scene['preset']
            assert job['provenance']['resolved_revision'].startswith(spec['revision'])
            scene_hashes.add(mutation['scene_sha256'])
            settings = {s['name']: s['value'] for s in mutation['effective_settings']['settings']}
            expected = dict(spec['common_settings'], SETTING_DEBUG_LIGHT_SAMPLING='1' if case['sampler'] == 'brdf' else '2', SETTING_DEBUG_LIGHT_CUBE=str(case['cube']).lower(), SETTING_DEBUG_LIGHT_WEIGHT=str(case['weight']).lower(), SETTING_DEBUG_LIGHT_LOOKAHEAD=str(case['lookahead']).lower(), SETTING_DEBUG_LIGHT_AUDIT='true')
            assert all((settings[k] == v for k, v in expected.items())), (scene, case)
            assert 'SETTING_DEBUG_LIGHT_VISIBILITY' not in settings
            single, sm = read_image(raw / 'single.bin')
            mean, mm = read_image(raw / 'mean32.mean.bin')
            var, _ = read_image(raw / 'mean32.variance.bin')
            count, _ = read_image(raw / 'mean32.valid_count.bin')
            inc, _ = read_image(raw / 'mean32.incomplete_batches.bin')
            assert (count == 32).all() and (inc == 0).all() and (var >= 0).all() and (var[:, :, 3] == 0).all()
            accum = mm['accumulation']
            assert accum['frames'] == 32 and accum['end_frame'] - accum['start_frame'] == 31
            assert sm['frame_id'] < accum['start_frame']
            current_mask = mean[:, :, 3] > 0
            if mask is None:
                mask = current_mask
            assert np.array_equal(mask, current_mask)
            tree = tree_stats(raw / 'tree.bin')
            tree['reserved'] = tree.pop('guided_selections')
            assert json.loads((raw / 'tree.json').read_text())['frame_id'] == json.loads((raw / 'audit.json').read_text())['frame_id']
            nodewords = np.fromfile(raw / 'tree.bin', '<u4')[8:].reshape(-1, 8)
            nodevalues = nodewords.view('<f4')
            leaves = []
            pending = [1]
            while pending:
                index = pending.pop()
                n = nodevalues[index]
                link = int(nodewords[index, 7])
                if n[3] == 0:
                    continue
                if link & 2147483648:
                    leaves.append((tuple(n[:3]), link))
                else:
                    pending.extend((link, link + 1))
            leaves.sort()
            if leaf_set is None:
                leaf_set = leaves
            assert leaves == leaf_set
            audit = None if case['sampler'] == 'brdf' else audit_tree(raw / 'tree.bin', raw / 'audit.bin', case['weight'], case['lookahead'])
            cube_audit = audit_cube(raw / 'audit.bin') if case['cube'] else None
            batches = np.array(accum['screen_batch_means'])[:, :3] @ LUMA
            row = dict(case, scene=scene['id'], source_sha256=mutation['source_sha256'], scene_sha256=mutation['scene_sha256'], settings_sha256=mutation['effective_settings']['settings_sha256'], mean_luminance=float(batches.mean()), mean_luminance_se=float(batches.std(ddof=1) / np.sqrt(len(batches))), tree=tree, pdf_audit=audit, cube_audit=cube_audit, single_frame=sm['frame_id'], mean_frames=[accum['start_frame'], accum['end_frame']])
            sources.add(row['source_sha256'])
            rows.append(row)
            captured.append((case, mean, var, row))
            for label, data, tag in [('1 spp single frame', single, 'single'), ('32 frame mean', mean, 'mean32')]:
                filename = f"{scene['id']}-{case['id']}-{tag}.png"
                render(data, directory / filename, spec['exposure_ev'])
                for ahead in [False, True] if case['sampler'] == 'brdf' else [case['lookahead']]:
                    images.append(dict(path=filename, dimensions={'Scene': scene['label'], 'Variant': case['label'], 'Lookahead': 'Two levels' if ahead else 'Off', 'Accumulation': label}))
        assert len(scene_hashes) == 1
        pooled = np.mean([m[:, :, :3] for _, m, _, _ in captured], axis=0, dtype=np.float64)
        pooled = gaussian_filter(pooled, sigma=(2, 2, 0))
        lum = pooled @ LUMA
        lit = mask & (lum > 1e-06)
        q50, q99 = np.quantile(lum[lit], [0.5, 0.99])
        regions = {'all_receivers': mask, 'dark_half': lit & (lum <= q50), 'middle_49': lit & (lum > q50) & (lum <= q99), 'brightest_1': lit & (lum > q99)}
        for label, (x0, y0, x1, y1) in spec.get('regions', {}).get(scene['id'], {}).items():
            region = np.zeros_like(mask)
            region[y0:y1, x0:x1] = True
            regions[label] = region[::-1] & mask
        sensitivity = 2 ** spec['exposure_ev'] / (1 + 2 ** spec['exposure_ev'] * np.maximum(pooled, 0)) ** 2
        for case, mean, var, row in captured:
            row['regions'] = {key: dict(pixels=int(region.sum()), rgb_variance=float(var[:, :, :3][region].sum(axis=1, dtype=np.float64).mean()), display_variance_proxy=float((var[:, :, :3] * sensitivity ** 2)[region].sum(axis=1, dtype=np.float64).mean())) for key, region in regions.items()}
            total = var[:, :, :3].sum(axis=2, dtype=np.float64)[mask]
            cut = np.quantile(total, 0.99)
            row['top_1pct_variance_share'] = float(total[total >= cut].sum() / total.sum())
        scene_rows = [x[3] for x in captured]
        brdf = next((r for r in scene_rows if r['sampler'] == 'brdf'))
        for row in scene_rows:
            se = np.hypot(row['mean_luminance_se'], brdf['mean_luminance_se'])
            row['mean_difference_z_vs_brdf'] = (row['mean_luminance'] - brdf['mean_luminance']) / se if se else 0
            row['relative_mean_difference_vs_brdf'] = row['mean_luminance'] / brdf['mean_luminance'] - 1
            for region, metrics in row['regions'].items():
                metrics['variance_over_brdf'] = metrics['rgb_variance'] / brdf['regions'][region]['rgb_variance']
    assert len(sources) == 1
    suffix = '-' + selected_scene if selected_scene else ''
    (directory / f'summary{suffix}.json').write_text(json.dumps(rows, indent=2))
    (directory / f'images{suffix}.json').write_text(json.dumps(dict(title='Cube Sampling and Tree Weights vs Two-Level Lookahead - Diffuse Direct Only', images=images), indent=2))
    print(json.dumps([dict(scene=r['scene'], case=r['id'], mean=r['mean_luminance'], z=r['mean_difference_z_vs_brdf'], dark=r['regions']['dark_half']['variance_over_brdf'], bright=r['regions']['brightest_1']['variance_over_brdf'], cube_audit=r['cube_audit']) for r in rows], indent=2))

def preserve_profile(job_path, destination):
    import hashlib
    import shutil
    result = json.loads(job_path.read_text())
    assert result['success'] and result['failed'] == 0
    assert result['passed'] == result['requested_cases'] == result['completed_cases']
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_path, destination / 'profile.json')
    manifest = []
    for index, artifact in enumerate(result['artifacts']):
        source = Path(artifact['relative_path'])
        target = destination / f'{index:03d}-{source.name}'
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        assert digest == artifact.get('sha256', digest)
        manifest.append(dict(original=str(source), path=target.name, sha256=digest))
    (destination / 'preservation.json').write_text(json.dumps(manifest, indent=2))
    return result

def analyze_profiles(directory):
    specification = json.loads((directory / 'matrix.json').read_text())
    quality = json.loads((directory / 'summary.json').read_text())
    by_key = {(r['scene'], r['id']): r for r in quality}
    scenes = {s['preset']: s['id'] for s in specification['scenes']}
    cases = {c['id']: c for c in specification['cases']}
    policy = json.loads((directory / 'profile-acceptance.json').read_text(encoding='utf-8-sig'))
    samples = {}
    for path in sorted((directory / 'profiles').glob('*/profile.json')):
        result = json.loads(path.read_text())
        assert result['success'] and result['consistency']['consistent']
        assert not result['metric_conflicts']
        for case in result['cases']:
            assert case['status'] == 'passed' and case['frames'] == 128
            provenance = case['provenance']
            scene = scenes[case['preset_id']]
            variant = cases[case['config_id']]
            assert provenance['resolved_revision'].startswith(specification['revision'])
            assert provenance['source_snapshot_sha256'] == by_key[scene, variant['id']]['source_sha256']
            assert provenance['scene_sha256'] == by_key[scene, variant['id']]['scene_sha256']
            effective = provenance['effective_settings']
            settings = {s['name']: s['value'] for s in effective['settings']}
            expected = dict(specification['common_settings'], SETTING_DEBUG_LIGHT_SAMPLING='1' if variant['sampler'] == 'brdf' else '2', SETTING_DEBUG_LIGHT_CUBE=str(variant['cube']).lower(), SETTING_DEBUG_LIGHT_WEIGHT=str(variant['weight']).lower(), SETTING_DEBUG_LIGHT_LOOKAHEAD=str(variant['lookahead']).lower(), SETTING_DEBUG_LIGHT_AUDIT='false')
            assert all((settings[k] == value for k, value in expected.items()))
            metrics = {m['metric_id']: m for m in case['metrics']['metrics'] if m.get('main_source')}
            assert all((len(m['samples_ns']) == 128 for m in metrics.values()))
            initial = metrics['composite41_compute']
            assert initial['main_source'] == '/pass/composite/LightSamplingInitial.comp.glsl'
            build = [metrics[f'prepare{i}_compute'] for i in range(6, 20)]
            row = dict(profile_file=str(path.relative_to(directory / 'profiles')).replace('\\', '/'), initial_p50_ms=initial['p50_ns'] / 1000000.0, initial_avg_ms=initial['average_ns'] / 1000000.0, leaf_build_p50_ms=metrics['prepare7_compute']['p50_ns'] / 1000000.0, build_sum_avg_ms=sum((m['average_ns'] for m in build)) / 1000000.0, sentinel_p50_ms=metrics['prepare2_compute']['p50_ns'] / 1000000.0)
            samples.setdefault((scene, variant['id']), []).append(row)
    rows = []
    for (scene, case), captured_runs in samples.items():
        replacement = next((r for r in policy['replacement_sets'] if scene in r['scenes'] and case in r['cases']), None)
        runs = [r for r in captured_runs if not replacement or r['profile_file'] == replacement['profile_file']]
        excluded = [r for r in captured_runs if replacement and r['profile_file'] != replacement['profile_file']]
        assert len(runs) == 2
        row = dict(scene=scene, id=case, runs=runs, excluded_runs=excluded, replacement_reason=replacement['reason'] if replacement else None)
        for key in runs[0]:
            if key == 'profile_file':
                continue
            values = [r[key] for r in runs]
            row[key] = float(np.median(values))
            row[key + '_range'] = [min(values), max(values)]
        rows.append(row)
    for row in rows:
        references = [v['sentinel_p50_ms'] for r in rows if r['scene'] == row['scene'] for v in r['runs']]
        reference = float(np.median(references))
        row['timing_cautions'] = []
        if max(row['sentinel_p50_ms_range']) > reference * policy['sentinel_outlier_ratio']:
            row['timing_cautions'].append('Voxel-build sentinel is an outlier; timing comparison is inconclusive.')
        all_values = [r['initial_p50_ms'] for r in row['runs'] + row['excluded_runs']]
        row['all_observed_initial_range_ms'] = [min(all_values), max(all_values)]
        previous = [r['initial_p50_ms'] for r in row['excluded_runs'] if r['sentinel_p50_ms'] <= reference * policy['sentinel_outlier_ratio']]
        if previous:
            ratio = float(np.median(previous)) / row['initial_p50_ms']
            row['cohort_median_ratio'] = ratio
            if max(ratio, 1.0 / ratio) > policy['cohort_shift_ratio']:
                row['timing_cautions'].append('Initial-pass medians shift over 25% across cohorts despite a stable sentinel; absolute speed comparison is inconclusive.')
        low, high = row['initial_p50_ms_range']
        if high / low > policy['wide_repeat_ratio']:
            row['timing_cautions'].append('Initial-pass repeat range exceeds 25%; avoid small speedup claims.')
    (directory / 'timing-summary.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return rows

def write_results(directory):
    from html import escape
    specification = json.loads((directory / 'matrix.json').read_text())
    quality = json.loads((directory / 'summary.json').read_text())
    timing = json.loads((directory / 'timing-summary.json').read_text())
    times = {(r['scene'], r['id']): r for r in timing}
    assert len(quality) == len(times) == 45
    record = dict(protocol=specification, quality=quality, timing=timing, timing_acceptance=json.loads((directory / 'profile-acceptance.json').read_text(encoding='utf-8-sig')), source_commits=dict(baseline='2caa6f23', cube='e8eb55c0', weights='60cd7a3a', numerical_repair='36fe5afd'), validation=dict(tree_pdf_samples=sum(((r['pdf_audit'] or {}).get('samples', 0) for r in quality)), cube_pdf_samples=sum(((r['cube_audit'] or {}).get('samples', 0) for r in quality)), max_tree_pdf_relative_error=max(((r['pdf_audit'] or {}).get('max_relative_pdf_error', 0) for r in quality)), max_cube_pdf_relative_error=max(((r['cube_audit'] or {}).get('max_relative_pdf_error', 0) for r in quality)), numeric_stress_frames=512, numeric_stress_nonfinite_pixels=0), limitations=['32-frame sample variance is not MSE against a converged reference.', 'Brightness regions use a shared pooled mean blurred by two pixels; they are diagnostic masks, not ground truth.', 'Two forward/reverse timing runs per configuration; inspect their ranges, especially noisy BRDF and emission-build timings.', 'The corridor has unexplained cross-cohort timing shifts despite a stable voxel-build sentinel; its absolute speed comparisons are inconclusive.', 'Tree-weight candidate combines texture-energy quadrature and finite-extent geometry; this matrix does not isolate them.', 'Projected-solid-angle inversion uses two refinement iterations; tiny slivers use clipped planar-area sampling with its own PDF.', 'BRDF does not need the light tree; its reported initial-pass timing excludes the diagnostic tree builds retained by the experiment.'])
    tracked = Path('experiments/light-sampling/20261029-geometry.json')
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text(json.dumps(record, indent=2))
    html = ['<!doctype html><meta charset="utf-8"><title>Light sampling geometry experiment</title>', '<style>body{font:15px system-ui;margin:32px;background:#14171b;color:#e5e9ef;line-height:1.5}a{color:#8cc7ff}table{border-collapse:collapse;margin:20px 0;width:100%}th,td{padding:7px 10px;border-bottom:1px solid #3c444e;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#222932}p{max-width:1100px}.note{color:#b9c5d2}h2{margin-top:36px}</style>', '<h1>Light sampling：灯内采样、树权重与两层前瞻</h1>', '<p><a href="comparison.html">打开完整图片对比器</a> · 5 场景 × 9 配置 · 720p · 1 spp 单帧 / 32 帧线性均值</p>', '<p>本轮两项组合不能普遍替代前瞻。灯内采样显著改善近灯区域；近灯墙面上不加前瞻已能追平旧前瞻的墙面方差，同时改善近灯噪声。隧道地板仍明显依赖树前瞻。当前树权重候选有严重高亮尾部，不宜采用。</p>', '<p class="note">RC、ReSTIR、降噪、TAA、visibility 引导均关闭；保留真实遮挡射线。亮度分区在所有配置间共用。方差列均以同场景 BRDF = 1，越低越好；不是高采样参考的 MSE。时间是 initial-sampling pass 两次正反序测量的 p50 中位数，括号为两次范围。叶子建树单独列出，不把各 pass 的 p50 相加。带 * 的时间重复波动较大，不支持小幅速度差异的结论。大教堂及暗走廊的四个关键对照以一次完整正反序复测替换初轮，原始数据全部保留。暗走廊补测仍出现跨轮次漂移，不能据其表中时间作确定的性能排名。</p>']
    for scene in specification['scenes']:
        html.append('<h2>' + escape(scene['label']) + '</h2>')
        rows = [r for r in quality if r['scene'] == scene['id']]
        rois = list(specification.get('regions', {}).get(scene['id'], {}))
        headers = ['配置', '前瞻', 'Initial GPU ms (range)', 'Leaf build ms (range)', '暗半区方差 / BRDF', '亮 1% 方差 / BRDF'] + [r + ' / BRDF' for r in rois] + ['均值差 / BRDF']
        html.append('<table><tr>' + ''.join(('<th>' + escape(h) + '</th>' for h in headers)) + '</tr>')
        for r in rows:
            t = times[r['scene'], r['id']]

            def duration(key):
                a, b = t[key + '_range']
                return f'{t[key]:.3f} ({a:.3f}–{b:.3f})' + (' *' if key == 'initial_p50_ms' and t['timing_cautions'] else '')
            cells = [r['label'], '2 levels' if r['lookahead'] else 'Off', duration('initial_p50_ms'), duration('leaf_build_p50_ms'), f"{r['regions']['dark_half']['variance_over_brdf']:.3f}", f"{r['regions']['brightest_1']['variance_over_brdf']:.3f}"]
            cells += [f"{r['regions'][roi]['variance_over_brdf']:.3f}" for roi in rois]
            cells += [f"{r['relative_mean_difference_vs_brdf']:+.2%}"]
            html.append('<tr>' + ''.join(('<td>' + escape(c) + '</td>' for c in cells)) + '</tr>')
        html.append('</table>')
        sentinel = [v for r in rows for v in times[r['scene'], r['id']]['sentinel_p50_ms_range']]
        html.append(f'<p class="note">工作量参照：VoxelTreeBuilder p50 {min(sentinel):.3f}–{max(sentinel):.3f} ms。</p>')
    html += ['<h2>验证与历史</h2>', '<p>主实验冻结于 commit 36fe5afd。基线 2caa6f23；灯内采样 e8eb55c0；树权重 60cd7a3a；极薄面片数值修复 36fe5afd。每步独立提交。</p>', '<p>45 组均通过实际 Iris 编译、配置与场景一致性、完整 32 帧检查；2,488 个树 PDF 和 1,129 个灯内 PDF 独立核对。大教堂极薄面片故障已保留诊断并修复，512 帧压力采集每像素完整，无非有限样本。失败前的数据保留在 raw-before-sliver-fix。</p>', '<p class="note">隧道 weight-plain 的短窗均值比 BRDF 低约 10%，大教堂 both-plain 也出现约 3 个估计标准误的差异；这提示尾部未收敛，不能当成“更少能量所以更干净”。本轮未进行高采样 MSE 验收。</p>', '<p class="note">运行环境：RTX 3080 Ti，NVIDIA 617.42，Iris 1.10.6-vibris.2。原始数据、完整计时样本、PDF 检查和配置保存在同目录；可移植结果记录在 experiments/light-sampling/20261029-geometry.json。</p>']
    (directory / 'results.html').write_text('\n'.join(html), encoding='utf-8')
    return record
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--scene')
    parser.add_argument('--profiles', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_cube(args.audit), indent=2))
    if args.analyze:
        analyze(args.analyze, args.scene)
    if args.profiles:
        analyze_profiles(args.profiles)
    if args.report:
        write_results(args.report)

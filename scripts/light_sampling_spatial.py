"""Analyze the two-scene spatial grouping experiment; preserve raw evidence separately."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from light_sampling_runtime import read_image, render, tree_stats, LUMA
from light_sampling_geometry import audit_tree, audit_cube

ROOT = Path('builds/light-sampling/20261029-spatial')
SCENES = {'gishadow': 'gi-shadow-1-720p', 'tunnel': 'codex-fg-night-tunnel'}
CASES = ['count-plain', 'count-ahead', 'spatial-plain', 'spatial-ahead']
REVISIONS = {'count': '4920b6d9', 'spatial': 'bb7c7978'}
ROIS = {'gishadow': {'wall': [80,350,550,690], 'near_light': [720,90,940,340]}, 'tunnel': {'floor': [180,520,800,700], 'wall': [950,220,1250,550]}}

def analyze():
    rows, images = [], []
    for scene, preset in SCENES.items():
        captures = []
        hashes, leaves = set(), set()
        for case in CASES:
            d = ROOT / 'raw' / scene / case
            job = json.loads((d/'job-result.json').read_text())
            assert job['success']
            job = job['result']
            assert all(r['status']=='RECEIPT_STATUS_OK' for r in job['action_receipts'])
            p = job['provenance']
            assert p['preset_id']==preset and p['resolved_revision'].startswith(REVISIONS[case.split('-')[0]])
            mutation = job['action_receipts'][0]['runtime_mutation']
            settings = {s['name']: s['value'] for s in mutation['effective_settings']['settings']}
            expected = dict(SETTING_DEBUG_LIGHT_CUBE='true', SETTING_DEBUG_LIGHT_WEIGHT='false', SETTING_DEBUG_LIGHT_SAMPLING='2', SETTING_DEBUG_LIGHT_AUDIT='true', SETTING_DEBUG_LIGHT_LOOKAHEAD=str(case.endswith('ahead')).lower())
            expected.update(json.loads(Path('builds/light-sampling/20261029-geometry-matrix/matrix.json').read_text())['common_settings'])
            assert all(settings[k]==v for k,v in expected.items())
            hashes.add(mutation['scene_sha256'])
            single, sm = read_image(d/'single.bin')
            mean, mm = read_image(d/'mean32.mean.bin')
            variance, _ = read_image(d/'mean32.variance.bin')
            count, _ = read_image(d/'mean32.valid_count.bin')
            incomplete, _ = read_image(d/'mean32.incomplete_batches.bin')
            assert (count>=0).all() and (count<=32).all() and (variance>=0).all()
            complete=(count==32).all(2) & (incomplete==0).all(2)
            a=mm['accumulation']; assert a['end_frame']-a['start_frame']==31 and sm['frame_id']<a['start_frame']
            assert json.loads((d/'tree.json').read_text())['frame_id']==json.loads((d/'audit.json').read_text())['frame_id']
            tree=tree_stats(d/'tree.bin'); leaves.add(tree['leaf_set_sha256'])
            words=np.fromfile(d/'tree.bin','<u4')[8:].reshape(-1,8)
            values=words.view('<f4'); pending=[(1,0)]; depths=[]
            while pending:
                node,depth=pending.pop()
                if values[node,3]==0: continue
                link=int(words[node,7])
                if link & 0x80000000:
                    depths.append(depth)
                else:
                    if case.startswith('spatial') and node>=8192:
                        left,right=values[link:link+2]
                        assert left[3]>0 and right[3]>0
                        assert np.any(left[4:7]<=right[:3]) or np.any(right[4:7]<=left[:3])
                    pending.extend([(link,depth+1),(link+1,depth+1)])
            tree['mean_leaf_depth']=float(np.mean(depths)); tree['max_leaf_depth']=max(depths)

            row=dict(incomplete_pixels=int((~complete).sum()),scene=scene,case=case,revision=p['resolved_revision'],scene_sha256=mutation['scene_sha256'],source_sha256=mutation['source_sha256'],tree=tree,pdf=audit_tree(d/'tree.bin',d/'audit.bin',False,case.endswith('ahead')),cube_pdf=audit_cube(d/'audit.bin'),mean_luminance=float(np.mean(mean[:,:,:3]@LUMA)))
            rows.append(row); captures.append((row,mean,variance,complete))
            for label, data in [('single',single),('mean32',mean)]:
                name=f'{scene}-{case}-{label}.png'; render(data,ROOT/name,6)
                images.append(dict(path=name,dimensions={'Scene':scene,'Grouping':case.split('-')[0],'Lookahead':'2 levels' if case.endswith('ahead') else 'Off','Accumulation':label}))
        assert len(hashes)==len(leaves)==1
        receiver=captures[0][1][:,:,3]>0
        assert all(np.array_equal(receiver,c[1][:,:,3]>0) for c in captures)
        receiver &= np.logical_and.reduce([c[3] for c in captures])
        shared=gaussian_filter(np.mean([m[:,:,:3]@LUMA for _,m,_,_ in captures],axis=0),2)
        lit=receiver & (shared>1e-6); q50,q99=np.quantile(shared[lit],[.5,.99])
        masks={'dark_half':lit & (shared<=q50),'brightest_1':lit & (shared>=q99)}
        for name,(x0,y0,x1,y1) in ROIS[scene].items():
            mask=np.zeros_like(receiver); mask[y0:y1,x0:x1]=True; masks[name]=mask[::-1] & receiver
        for row,mean,var,_ in captures:
            row['variance']={k:float(var[:,:,:3].sum(2)[mask].mean()) for k,mask in masks.items()}
        for row,_,_,_ in captures:
            row['variance_over_count_plain']={k:v/captures[0][0]['variance'][k] for k,v in row['variance'].items()}
    (ROOT/'images.json').write_text(json.dumps(dict(title='Spatial light grouping',images=images),indent=2))
    result=dict(protocol='720p diffuse direct only; cube sampling, old weights; no visibility guiding, RC, ReSTIR, denoising or TAA; 1 spp single and 32-frame linear mean. Variance is a short-window estimate, not reference MSE. Common complete-pixel mask excludes any pixel with missing samples in any case; gishadow count-ahead has one incomplete RGB pixel. Existing cube sampler numerical issue remains unresolved.',quality=rows)
    profile_path=ROOT/'profile.json'
    if profile_path.exists():
        profile=json.loads(profile_path.read_text()); assert profile['success'] and not profile['metric_conflicts']
        timing=[]
        for c in profile['cases']:
            assert c['status']=='passed'
            provenance=c['provenance']
            assert provenance['resolved_revision'].startswith(REVISIONS[c['source_id']])
            assert not provenance['stale'] and not provenance['head_changed']
            settings={v['name']:v['value'] for v in provenance['effective_settings']['settings']}
            expected=dict(json.loads(Path('builds/light-sampling/20261029-geometry-matrix/matrix.json').read_text())['common_settings'],SETTING_DEBUG_LIGHT_CUBE='true',SETTING_DEBUG_LIGHT_WEIGHT='false',SETTING_DEBUG_LIGHT_SAMPLING='2',SETTING_DEBUG_LIGHT_AUDIT='false',SETTING_DEBUG_LIGHT_LOOKAHEAD=str(c['config_id']=='ahead').lower())
            assert all(settings[k]==v for k,v in expected.items())
            quality=next(r for r in rows if SCENES[r['scene']]==c['preset_id'] and r['case']==c['source_id']+'-'+c['config_id'])
            assert quality['scene_sha256']==provenance['scene_sha256']
            assert quality['source_sha256']==provenance['source_snapshot_sha256']
            m={m['metric_id']:m for m in c['metrics']['metrics']}
            timing.append({k:v for k,v in c.items() if k not in ['metrics','attempts','provenance'] } | dict(initial_ms=m['composite41_compute']['p50_ns']/1e6,build_ms=m['prepare7_compute']['p50_ns']/1e6,sentinel_ms=m['prepare2_compute']['p50_ns']/1e6,total_build_avg_ms=sum(v['average_ns'] for k,v in m.items() if k.startswith('prepare') and k!='prepare2_compute')/1e6))
        result['timing']=timing
        result['timing_note']='128 warmup + 128 measured frames, palindrome, two runs per case. Tunnel count/plain initial pass drifted 1.014 to 1.315 ms despite stable sentinel; do not infer a precise speedup. Build and initial are reported separately.'
        import hashlib, shutil
        preserved=ROOT/'profile-artifacts'; preserved.mkdir(exist_ok=True)
        for i,a in enumerate(profile['artifacts']):
            target=preserved/f"{i:03d}-{Path(a['relative_path']).name}"
            if not target.exists(): shutil.copy2(a['relative_path'],target)
            assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(Path(a['relative_path']).read_bytes()).digest()
        html=['<!doctype html><meta charset="utf-8"><title>Spatial grouping experiment</title><style>body{background:#171a20;color:#eee;font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #555}a{color:#8bf}</style><h1>空间分组实验</h1><p><a href="comparison.html">图片对比</a></p>', '<p>'+result['protocol']+'</p><p>'+result['timing_note']+'</p>']
        for scene,preset in SCENES.items():
            html.append('<h2>'+scene+'</h2><table><tr><th>Case</th><th>Initial p50 ms: two runs</th><th>Brick build p50 ms: two runs</th><th>Variance / count-plain</th></tr>')
            for case in CASES:
                q=next(r for r in rows if r['scene']==scene and r['case']==case)
                runs=[r for r in timing if r['preset_id']==preset and r['source_id']+'-'+r['config_id']==case]
                assert len(runs)==2
                html.append('<tr><td>'+case+'</td><td>'+', '.join(f"{r['initial_ms']:.3f}" for r in runs)+'</td><td>'+', '.join(f"{r['build_ms']:.3f}" for r in runs)+'</td><td>'+', '.join(f'{k}: {v:.3f}' for k,v in q['variance_over_count_plain'].items())+'</td></tr>')
            html.append('</table>')
        html.append('<p>砖内建树由按数量归并替换为 Morton 空间前缀分组，压掉单子分支。共享内存由 16 KiB 增至 32 KiB；SSBO 容量不变。只测试近灯墙面和隧道，未测试宽树和 Gilbert。</p>')
        (ROOT/'results.html').write_text('\n'.join(html),encoding='utf-8')
        Path('experiments/light-sampling/20261029-spatial.json').write_text(json.dumps(result,indent=2))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    analyze()

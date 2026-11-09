"""Archive and score matched native LightSamplingInitial cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from matplotlib.path import Path as Polygon
import numpy as np

from light_sampling_distribution_review import REGIONS, summarize
from light_sampling_moving_analysis import bracket_identity, native_node
from light_sampling_point_moments import ROOT, read, sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree

RUN = ROOT/'builds/light-sampling/20261108-eight-hour-exploration'
EARLIER = ROOT/'builds/light-sampling/20261108-clustering-repair'


def archive(job_id, name):
    job = ROOT/'.vibris/jobs'/job_id
    request = read(job/'request.json')['arguments']
    saved = read(job/'result.json')
    assert saved['success']
    result = saved['result']
    prov = result['provenance']
    assert not prov['stale'] and not prov['patch_sha256'] and not prov['dirty_shader_delta_sha256']
    assert prov['resolved_revision'] == request['sources'][0]['resolved_revision']
    assert prov['source_snapshot_sha256'] == request['sources'][0]['snapshot_sha256']
    assert prov['preset_id'] == request['preset_id']
    assert result['restoration']['status'] == 'RECEIPT_STATUS_OK'
    assert len(request['actions']) == len(result['action_receipts'])
    assert all(r['status']=='RECEIPT_STATUS_OK' for r in result['action_receipts'])
    settings = {v['name']:v['value'] for v in prov['effective_settings']['settings']}
    for key, value in request['configs'][0]['values'].items():
        assert settings[key] == value, (key, value, settings[key])
    assert settings['SETTING_DEBUG_LIGHT_AUDIT'] == 'false'
    context = request['__vibris_scene_context']
    folder = RUN/'cases'/prov['preset_id']/name
    manifest, ref = retain_manifest(job, result, folder)
    identity = input_identity(folder, context, settings)
    metrics = next(r['gpu_metrics'] for r in result['action_receipts'] if r['kind']=='ACTION_KIND_GET_GPU_METRICS')
    timings = timing(metrics)
    accepted = dict(
        job=job_id, name=folder.name, scene=prov['preset_id'], context=context,
        manifest_id=manifest['manifest_id'], manifest_sha256=ref['sha256'],
        source=prov['resolved_revision'], source_snapshot_sha256=prov['source_snapshot_sha256'],
        preset_sha256=prov['preset_sha256'], scene_sha256=prov['scene_sha256'], settings=settings,
        restore=result['restoration'], identity=identity, timings=timings,
        resolution=context['resolution'], path=folder.relative_to(ROOT).as_posix(),
    )
    (folder/'accepted.json').write_text(json.dumps(accepted,indent=2,allow_nan=False)+'\n', encoding='utf8')
    return accepted



def retain_manifest(job, result, folder):
    ref = next(a for a in result['artifacts'] if a['kind']=='ARTIFACT_KIND_MANIFEST')
    manifest_path = Path(ref['relative_path'])
    assert sha(manifest_path) == ref['sha256']
    manifest = read(manifest_path)
    assert manifest['manifest_id'] == result['result_manifest_id']
    folder.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        rel = Path(item['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        source = manifest_path.parent/rel
        target = folder/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source,target)
            except OSError:
                shutil.copyfile(source,target)
        assert target.stat().st_size == item['byte_size'] and sha(target) == item['sha256']
    for source, name in ((manifest_path,'manifest.json'),(job/'request.json','request.json'),
                         (job/'result.json','job-result.json')):
        shutil.copyfile(source, folder/name)
    return manifest, ref


def archive_ordered(job_id, name):
    job = ROOT/'.vibris/jobs'/job_id
    request = read(job/'request.json')['arguments']
    saved = read(job/'result.json')
    assert saved['success']
    result = saved['result']
    assert result['restoration']['status'] == 'RECEIPT_STATUS_OK'
    actions = request['actions']
    receipts = result['action_receipts']
    assert len(actions) == len(receipts)
    assert all(r['status']=='RECEIPT_STATUS_OK' for r in receipts)
    context = request['__vibris_scene_context']
    folder = RUN/'ordered'/request['preset_id']/name
    manifest, ref = retain_manifest(job,result,folder)
    sources = {r['id']:r for r in request['sources']}
    configs = {r['id']:r['values'] for r in request['configs']}
    parts = []
    for action,receipt in zip(actions,receipts,strict=True):
        if action['type'] == 'load_shader':
            source = sources[action['source_id']]
            mutation = receipt['runtime_mutation']
            assert mutation['source_sha256']==source['snapshot_sha256']
            settings = {s['name']:s['value'] for s in mutation['effective_settings']['settings']}
            expected = configs[action['config_id']]
            assert all(settings[k]==v for k,v in expected.items())
            assert settings['SETTING_DEBUG_LIGHT_AUDIT']=='false'
            parts.append(dict(source=source['resolved_revision'], config=action['config_id'],
                              scene_sha256=mutation['scene_sha256'], settings=expected))
        elif action['type'] == 'get_gpu_metrics':
            parts[-1]['timing'] = timing(receipt['gpu_metrics'])
    assert len(parts)==4
    assert len({r['scene_sha256'] for r in parts})==1
    for i,part in enumerate(parts):
        links = folder/f'identity{i}'
        links.mkdir(exist_ok=True)
        # For adjacent measured segments, the later checkpoint is the input
        # identity. The following load may change only tree topology, not leaves.
        for phase, seq in (('before',i),('after',i+1)):
            for field in ('camera','tree','voxel-meta','voxel-materials'):
                source = folder/f'check{seq}_{field}.bin'
                target = links/f'{phase}_{field}.bin'
                if not target.exists():
                    os.link(source,target)
                # Validate linked identity files if this archive is replayed.
                assert os.path.samefile(source,target)
        preset = context['camera_preset_id']
        fixture_dirs = sorted((EARLIER/'cases'/preset).glob('m2-*'))
        brick = read(fixture_dirs[0]/'accepted.json')['pose']['camera_brick'] if fixture_dirs else [0,0,0]
        from light_sampling_moving_analysis import world_voxel_identity, leaves, ordered_signature
        def checkpoint(phase):
            tree = links/f'{phase}_tree.bin'
            header, nodes, links_data = load_tree(tree,1)
            root = native_node(nodes,links_data,1)
            rows = sorted(leaves(root))
            assert len(rows)==int(header[1]) and header[2]==header[3]==0
            camera = np.fromfile(links/f'{phase}_camera.bin','<f4')
            voxels = world_voxel_identity(links,int(part['settings']['SETTING_VOXEL_GRID_SIZE']),brick,
                                          prefix=phase+'_',context=context)
            return dict(camera_sha256=hashlib.sha256(np.r_[camera[16:32],camera[80:112]].tobytes()).hexdigest(),
                        leaf_sha256=hashlib.sha256(np.asarray(rows,'<f8').tobytes()).hexdigest(),
                        world_voxel_sha256=voxels['world_blocks_material_sha256'],
                        ordered_topology_sha256=ordered_signature(root))
        part['before'],part['after'] = checkpoint('before'),checkpoint('after')
    fields = ('camera_sha256','leaf_sha256','world_voxel_sha256')
    common = {key:parts[0]['before'][key] for key in fields}
    same_input = all(all(check[key]==common[key] for key in fields)
                     for part in parts for check in (part['before'],part['after']))
    medians = [r['timing']['initial']['p50_ns'] for r in parts]
    configs_order = [r['config'] for r in parts]
    assert configs_order in (['camera','candidate','candidate','camera'], ['candidate','camera','camera','candidate'])
    pooled = {cfg:float(np.mean([m for m,r in zip(medians,parts,strict=True) if r['config']==cfg]))
              for cfg in ('camera','candidate')}
    means = {cfg:float(np.mean([part['timing']['initial']['mean_ns'] for part in parts if part['config']==cfg]))
             for cfg in ('camera','candidate')}
    for cfg in ('camera','candidate'):
        assert len({p['before']['ordered_topology_sha256'] for p in parts if p['config']==cfg})==1
    paired = []
    for i,j in ((0,1),(2,3)):
        p = {parts[k]['config']:medians[k] for k in (i,j)}
        paired.append(p['candidate']/p['camera'])
    report = dict(job=job_id, path=folder.relative_to(ROOT).as_posix(), parts=parts, order=configs_order,
                  identity=common, same_input=same_input, mean_of_block_medians_ns=pooled,
                  paired_time_ratios=paired, time_ratio=pooled['candidate']/pooled['camera'],
                  mean_of_block_means_ns=means, mean_time_ratio=means['candidate']/means['camera'],
                  manifest_id=manifest['manifest_id'],manifest_sha256=ref['sha256'],
                  restoration=result['restoration'])
    (folder/'accepted.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return {key:report[key] for key in ('job','path','same_input','time_ratio','mean_time_ratio','paired_time_ratios',
                                       'manifest_id','manifest_sha256')}

def input_identity(folder, context, settings):
    preset = context['camera_preset_id']
    fixture_dirs = sorted((EARLIER/'cases'/preset).glob('m2-*'))
    # The first comparison uses the established poses for ls1 and ls5. Other
    # fixed presets can be compared in common window-relative voxel coordinates.
    brick = read(fixture_dirs[0]/'accepted.json')['pose']['camera_brick'] if fixture_dirs else [0,0,0]
    identity = bracket_identity(folder, int(settings['SETTING_VOXEL_GRID_SIZE']), brick, context)['before']
    identity['coordinates'] = 'world' if fixture_dirs else 'camera-window relative'
    identity['grid'] = int(settings['SETTING_VOXEL_GRID_SIZE'])
    identity['gbuffer'] = {f:sha(folder/f'{f}.bin') for f in ('colortex8','colortex9','colortex10')
                           if (folder/f'{f}.bin').is_file()}
    return identity


def timing(measurement):
    rows = measurement['metrics']
    initial = [r for r in rows if r.get('main_source','').endswith('/LightSamplingInitial.comp.glsl')]
    assert len(initial)==1
    build = [r for r in rows if r.get('main_source','').endswith('/LightSamplingBuild.comp.glsl')]
    samples = int(measurement['sampled_frames'])
    assert samples==128
    def collect(row):
        vals = np.asarray(row['samples_ns'], dtype=float)
        assert len(vals)==samples and np.isfinite(vals).all() and np.all(vals>=0)
        return dict(id=row['metric_id'], pass_id=row['pass_id'], source=row.get('main_source'),
                    p50_ns=float(np.median(vals)), mean_ns=float(vals.mean()),
                    p10_ns=float(np.quantile(vals,.1)), p90_ns=float(np.quantile(vals,.9)))
    prepared = [collect(row) for row in build]
    summed = np.sum([row['samples_ns'] for row in build],axis=0) if build else np.zeros(samples)
    sentinel = next((r for r in rows if r['metric_id']=='prepare2_compute'),None)
    siblings = [r for r in rows if r.get('pass_id') in ('composite40','composite42')]
    return dict(sampled_frames=samples, initial=collect(initial[0]), build_p50_sum_ns=float(np.median(summed)),
                build=prepared, sentinel=collect(sentinel) if sentinel else None,
                neighbors=[collect(r) for r in siblings])


def pixels(folder, resolution, window):
    height,width = resolution['height'],resolution['width']
    prefix = folder/f'window{window}'
    meta = read(prefix.with_suffix('.mean.json'))
    accumulation = meta['accumulation']
    assert accumulation['frames']==32 and accumulation['batches']==8
    assert accumulation['end_frame']-accumulation['start_frame']==31
    def image(field):
        return np.memmap(prefix.with_suffix(f'.{field}.bin'), '<f4', 'r', shape=(height,width,4))[::-1]
    mean, variance = image('mean'), image('variance')
    valid = (mean[:,:,3]==1) & np.isfinite(mean[:,:,:3]).all(axis=2) & np.isfinite(variance[:,:,:3]).all(axis=2)
    for name,value in [('valid_count',32),('valid_batches',8),('incomplete_batches',0),('nonfinite',0)]:
        valid &= (image(name)[:,:,:3]==value).all(axis=2)
    assert np.all(variance[valid,:3]>=0)
    return mean, variance, valid, [accumulation['start_frame'],accumulation['end_frame']]


def match_light_keys(folder, names, variant):
    data = {}
    for name in names:
        header,nodes,links = load_tree(folder/name/'before_tree.bin',1)
        assert header[2]==header[3]==0
        tree = Tree(native_node(nodes,links,1))
        leaves = dict(zip(tree.keys,tree.nodes[tree.leaves,3],strict=True))
        data[name] = (tree,leaves)
    keys = sorted(data[names[0]][1])
    assert all(sorted(row[1])==keys for row in data.values())
    result = {'world_light_keys_sha256':hashlib.sha256(np.asarray(keys,'<i8').tobytes()).hexdigest(),
              'count':len(keys)}
    tree,leaves = data['prefix']
    candidate,candidate_leaves = data[variant]
    np.testing.assert_array_equal(tree.nodes[:,[0,1,2,4,5,6]],candidate.nodes[:,[0,1,2,4,5,6]])
    np.testing.assert_array_equal(tree.parents,candidate.parents)
    np.testing.assert_array_equal(tree.materials,candidate.materials)
    mass = np.asarray([(leaves[key],candidate_leaves[key]) for key in keys])
    support_floor = .5 if variant == 'texture-mass' else 1./7
    assert np.isfinite(mass).all() and np.all(mass[:,1]>=support_floor*mass[:,0]*(1.-2e-7))
    result['changed_lamp_masses'] = int(np.sum(mass[:,0]!=mass[:,1]))
    result['mass_ratio_quantiles'] = np.quantile(mass[:,1]/mass[:,0],[0,.05,.5,.95,1.]).tolist()
    if variant == 'texture-mass':
        uv_paths = {phase:folder/variant/f'{phase}_face-uv.bin' for phase in ('before','after')}
        result['uv_tables'] = {phase:sha(p) for phase,p in uv_paths.items()}
        uv = {phase:np.fromfile(p,'<u4')[:16384*6*2].reshape(-1,6,2) for phase,p in uv_paths.items()}
        ids = sorted({key[3] for key in keys})
        result['emitting_material_uvs'] = dict(
            material_count=len(ids), changing_faces=int(np.any(uv['before'][ids]!=uv['after'][ids],axis=2).sum()),
            populated_faces=int(np.any(uv['before'][ids]!=0,axis=2).sum()))
    return result


def compare(scene, names, full_receiver=False, proposal_mass_variant=None):
    folder = RUN/'cases'/scene
    cases = {n:read(folder/n/'accepted.json') for n in names}
    base = cases[names[0]]
    common_ids = ('camera_sha256','world_voxel_sha256','coordinates','grid')
    if proposal_mass_variant is None:
        common_ids += ('leaf_sha256',)
    else:
        assert 'prefix' in names and proposal_mass_variant in names
        assert all(cases[n]['identity']['leaf_sha256']==base['identity']['leaf_sha256']
                   for n in names if n!=proposal_mass_variant)
    for n,r in cases.items():
        assert r['context']==base['context'] and r['scene_sha256']==base['scene_sha256'], n
        for key in common_ids:
            assert r['identity'][key]==base['identity'][key], (n,key,r['identity'][key],base['identity'][key])
    mass_change = match_light_keys(folder,names,proposal_mass_variant) if proposal_mass_variant else None
    dim = base['resolution']
    points = {}
    height,width = dim['height'],dim['width']
    if scene=='light-sampling-5':
        yy,xx = np.indices((height,width))
        coordinates = np.column_stack((xx.ravel()*1920/width,yy.ravel()*1080/height))
        points = {name:Polygon(vertices).contains_points(coordinates).reshape(height,width)
                  for name,vertices in REGIONS.items()}
    elif scene in ('light-sampling-1','light-sampling-failure-moving-4'):
        xx,yy = np.meshgrid(np.arange(width)*1920/width,np.arange(height)*1080/height)
        centers = ([('red-pillar',(250,400)),('central-ceiling',(1000,170))]
                   if scene=='light-sampling-1' else [('side-a',(700,583)),('side-b',(1312,515)),('top',(1000,631))])
        for name,(x,y) in centers:
            points[name] = (abs(xx-x)<=4) & (abs(yy-y)<=4)
    windows = []
    previous_end = {}
    common = np.ones((height,width),dtype=bool)
    data = {}
    for window in (1,2):
        data[window] = {}
        for n in names:
            mean, variance, mask, frames = pixels(folder/n,dim,window)
            common &= mask
            assert frames[0]>previous_end.get(n,-1)
            previous_end[n] = frames[1]
            data[window][n] = (mean,variance,frames)
    complete_pixels = int(common.sum())
    receiver_match = np.ones_like(common)
    input_payloads = [('colortex8','<u4',(height,width,4)), ('colortex10','<f4',(height,width))]
    if all((folder/n/'colortex9.bin').exists() for n in names):
        input_payloads.append(('colortex9','<u4',(height,width)))
    mismatches = {}
    for field,dtype,shape in input_payloads:
        reference = np.memmap(folder/names[0]/f'{field}.bin',dtype,'r',shape=shape)[::-1]
        for n in names[1:]:
            image = np.memmap(folder/n/f'{field}.bin',dtype,'r',shape=shape)[::-1]
            if field == 'colortex8' and not full_receiver:
                # ls_loadReceiver uses only geometric normal and material ID.
                # Lightmap coordinates can change without affecting the estimator.
                equal = (((reference[:,:,0]^image[:,:,0]) & 0xffff) == 0)
                equal &= ((reference[:,:,3]^image[:,:,3]) >> 16) == 0
            elif field == 'colortex9':
                equal = ((reference^image)&0xffffff)==0
            else:
                equal = reference==image
                if equal.ndim == 3:
                    equal = equal.all(axis=2)
            receiver_match &= equal
            mismatches[f'{n}/{field}'] = int((common & ~equal).sum())
    common &= receiver_match
    time_base = base['timings']['initial']['p50_ns']
    time_ratios = {n:r['timings']['initial']['p50_ns']/time_base for n,r in cases.items()}
    mean_time = base['timings']['initial']['mean_ns']
    mean_time_ratios = {n:r['timings']['initial']['mean_ns']/mean_time for n,r in cases.items()}
    assert common.any()
    regions = {'whole':common, **{k:v & common for k,v in points.items()}}
    for window in (1,2):
        summaries = {}
        for region,mask in regions.items():
            if not mask.any():
                continue
            methods = {}
            for n,(mean,variance,frames) in data[window].items():
                rgb = variance[mask,:3].sum(axis=1,dtype=float)
                stats = summarize(rgb,np.any(mean[mask,:3]!=0,axis=1))
                stats['pixels'] = int(mask.sum())
                stats['frames'] = frames
                stats['mean_rgb'] = mean[mask,:3].mean(axis=0,dtype=float).tolist()
                stats['variance_times_ms'] = {m:stats[m]*cases[n]['timings']['initial']['p50_ns']/1e6
                                              for m in ('mean','p99','worst05_mean')}
                stats['variance_times_mean_ms'] = {m:stats[m]*cases[n]['timings']['initial']['mean_ns']/1e6
                                                   for m in ('mean','p99','worst05_mean')}
                if scene == 'light-sampling-5':
                    stats['blue_channel'] = summarize(variance[mask,2],mean[mask,2])
                methods[n] = stats
            summaries[region] = methods
        windows.append(summaries)
    result = dict(scene=scene, baseline=names[0], resolution=dim, cases=cases,
                  time_ratios=time_ratios, mean_time_ratios=mean_time_ratios, max_ratio=1.25,
                  raw_timing_cap={n:r<=1.25 for n,r in time_ratios.items()},
                  windows=windows, joint_complete_pixels=int(common.sum()), before_receiver_match_pixels=complete_pixels,
                  receiver_data_compared=[field for field,_,_ in input_payloads],
                  receiver_mismatches=mismatches, full_receiver=full_receiver, proposal_mass_change=mass_change)
    tag = ('full-gbuffer-' if full_receiver else '') + '-'.join(names)
    out = RUN/f'{scene}-{tag}.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    ops=parser.add_subparsers(dest='op',required=True)
    capture=ops.add_parser('archive')
    capture.add_argument('job_id')
    capture.add_argument('name')
    ordered = ops.add_parser('ordered')
    ordered.add_argument('job_id')
    ordered.add_argument('name')
    report=ops.add_parser('compare')
    report.add_argument('scene')
    report.add_argument('names',nargs='+')
    report.add_argument('--full-receiver',action='store_true')
    report.add_argument('--proposal-mass-variant',choices=('texture-mass','local-escape'))
    args=parser.parse_args()
    if args.op=='archive':
        result=archive(args.job_id,args.name)
        fields = ('job','scene','name','manifest_id','manifest_sha256','path','identity','timings')
        print(json.dumps({key:result[key] for key in fields}))
    elif args.op=='ordered':
        print(json.dumps(archive_ordered(args.job_id,args.name)))
    else:
        result=compare(args.scene,args.names,args.full_receiver,args.proposal_mass_variant)
        print(result['time_ratios'],result['joint_complete_pixels'])

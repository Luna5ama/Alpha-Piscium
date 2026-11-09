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
from light_sampling_moving_analysis import bracket_identity, leaves, native_node
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
    audited = name.endswith('-audit')
    assert settings['SETTING_DEBUG_LIGHT_AUDIT'] == ('true' if audited else 'false')
    context = request['__vibris_scene_context']
    folder = RUN/'cases'/prov['preset_id']/name
    manifest, ref = retain_manifest(job, result, folder)
    identity = input_identity(folder, context, settings)
    metrics = [r['gpu_metrics'] for r in result['action_receipts'] if r['kind']=='ACTION_KIND_GET_GPU_METRICS']
    assert len(metrics)==(0 if audited else 1)
    timings = None if audited else timing(metrics[0])
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



def archive_group(job_id, name):
    """Retain one frozen-scene action group containing two quality methods."""
    job = ROOT/'.vibris/jobs'/job_id
    request = read(job/'request.json')['arguments']
    saved = read(job/'result.json')
    assert saved['success']
    result = saved['result']
    provenance = result['provenance']
    assert not provenance['stale'] and not provenance['patch_sha256'] and not provenance['dirty_shader_delta_sha256']
    assert provenance['preset_id'] == request['preset_id']
    assert result['restoration']['status']=='RECEIPT_STATUS_OK'
    assert len(request['actions'])==len(result['action_receipts'])
    assert all(row['status']=='RECEIPT_STATUS_OK' for row in result['action_receipts'])
    context = request['__vibris_scene_context']
    scene = request['preset_id']
    group = RUN/'quality-groups'/scene/name
    manifest, ref = retain_manifest(job,result,group)
    sources = {row['id']:row for row in request['sources']}
    configs = {row['id']:row['values'] for row in request['configs']}
    entries = {}
    tag = None
    for action, receipt in zip(request['actions'],result['action_receipts'],strict=True):
        kind = action['type']
        if kind == 'load_shader':
            tag = action['config_id']
            assert Path(tag).name==tag and tag not in entries
            source = sources[action['source_id']]
            m = receipt['runtime_mutation']
            assert source['snapshot_sha256']==m['source_sha256']
            settings = {pair['name']:pair['value'] for pair in m['effective_settings']['settings']}
            assert all(settings[k]==v for k,v in configs[tag].items())
            assert settings['SETTING_DEBUG_LIGHT_AUDIT']=='false'
            entries[tag] = dict(job=job_id, name=tag, scene=scene, context=context, resolution=context['resolution'],
                                source=source['resolved_revision'],source_snapshot_sha256=m['source_sha256'],
                                preset_sha256=request['__vibris_preset']['preset_sha256'],
                                scene_sha256=m['scene_sha256'],settings=settings,restore=result['restoration'])
        elif kind=='get_gpu_metrics':
            entries[tag]['timings']=timing(receipt['gpu_metrics'])
    assert len(entries)==2 and all('timings' in row for row in entries.values())
    assert provenance['source_snapshot_sha256'] == entries[tag]['source_snapshot_sha256']
    assert provenance['scene_sha256'] == entries[tag]['scene_sha256']
    for tag, entry in entries.items():
        folder = RUN/'cases'/scene/tag
        folder.mkdir(parents=True,exist_ok=True)
        prefix = tag+'--'
        files = [item for item in manifest['files'] if item['relative_path'].startswith(prefix)]
        assert len(files)>=48
        for item in files:
            source = group/item['relative_path']
            path = folder/item['relative_path'][len(prefix):]
            if not path.exists():
                os.link(source,path)
            assert os.path.samefile(source,path)
        entry['identity']=input_identity(folder,context,entry['settings'])
        entry['manifest_id']=manifest['manifest_id']
        entry['manifest_sha256']=ref['sha256']
        entry['path']=folder.relative_to(ROOT).as_posix()
        entry['group_path']=group.relative_to(ROOT).as_posix()
        (folder/'accepted.json').write_text(json.dumps(entry,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return dict(job=job_id,group=group.relative_to(ROOT).as_posix(),manifest_id=manifest['manifest_id'],
                manifest_sha256=ref['sha256'],methods=entries)

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

def archive_motion(job_id, name):
    job = ROOT/'.vibris/jobs'/job_id
    request = read(job/'request.json')['arguments']
    saved = read(job/'result.json')
    assert saved['success']
    result = saved['result']
    assert result['restoration']['status']=='RECEIPT_STATUS_OK'
    assert all(r['status']=='RECEIPT_STATUS_OK' for r in result['action_receipts'])
    folder = RUN/'motion'/request['preset_id']/name
    manifest, ref = retain_manifest(job,result,folder)
    sources = {r['id']:r for r in request['sources']}
    configs = {r['id']:r['values'] for r in request['configs']}
    methods = {}
    for action,r in zip(request['actions'],result['action_receipts'],strict=True):
        if action['type']=='load_shader':
            method = action['config_id']
            m = r['runtime_mutation']
            source = sources[action['source_id']]
            assert m['source_sha256']==source['snapshot_sha256']
            effective = {v['name']:v['value'] for v in m['effective_settings']['settings']}
            assert all(effective[k]==v for k,v in configs[method].items())
            methods[method] = dict(source=source['resolved_revision'], scene_sha256=m['scene_sha256'],
                                   effective_settings=configs[method])
        elif action['type']=='camera_motion':
            frames = r['camera_motion']['frames']
            assert len(frames)==action['keyframes'][-1]['frame_offset']+1
            ids = [int(row['rendered_frame_id']) for row in frames]
            assert all(b==a+1 for a,b in zip(ids,ids[1:]))
            poses = [[row['actual_pose'].get(k,0.) for k in ('x','y','z','yaw_degrees','pitch_degrees')]
                     for row in frames]
            brick = np.floor(np.asarray(poses)[:,:3]/16).astype(int)
            crossings = (np.flatnonzero((brick[1:]!=brick[:-1]).any(axis=1))+1).tolist()
            paths = [item['relative_path'] for item in manifest['files'] if
                     item['relative_path'].startswith(action['artifact_prefix']+'-') and
                     item['relative_path'].endswith('.png')]
            assert len(paths)==len(frames) and crossings
            methods[method].update(frames=ids, poses=poses, crossing_frame_offsets=crossings,
                                   first_brick=brick[0].tolist(), screenshots=sorted(paths))
    assert len(methods)>=2 and len({r['scene_sha256'] for r in methods.values()})==1
    first = next(iter(methods.values()))
    for case in methods.values():
        np.testing.assert_allclose(case['poses'],first['poses'],rtol=0,atol=1e-4)
        assert case['crossing_frame_offsets']==first['crossing_frame_offsets']
    accepted = dict(job=job_id, scene=request['preset_id'],path=folder.relative_to(ROOT).as_posix(),
                    manifest_id=manifest['manifest_id'],manifest_sha256=ref['sha256'],methods=methods,
                    scope='Consecutive final framebuffer and actual camera poses. No per-frame raw tree or linear radiance evidence.')
    (folder/'accepted.json').write_text(json.dumps(accepted,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return accepted


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
    if (folder/'before_face-uv.bin').is_file():
        header,nodes,links = load_tree(folder/'before_tree.bin',1)
        rows = leaves(native_node(nodes,links,1))
        assert len(rows)==int(header[1])
        ids = np.asarray(sorted({r[3] for r in rows}),dtype='<u4')
        uv = {}
        for phase in ('before','after'):
            path = folder/f'{phase}_face-uv.bin'
            # Binding 9 allocates extra space; the shader reads material * 6 + face.
            data = np.fromfile(path,'<u4',count=16384*6*2).reshape(16384,6,2)
            uv[phase] = dict(raw_sha256=sha(path), emitting_sha256=hashlib.sha256(
                ids.tobytes()+data[ids].tobytes()).hexdigest(),
                populated_emitting_faces=int(np.any(data[ids]!=0,axis=2).sum()))
        identity['face_uv'] = dict(material_count=len(ids), **uv)
    if (folder/'frozen-face-uv.bin').is_file():
        identity['frozen_face_uv_sha256'] = sha(folder/'frozen-face-uv.bin')
        identity['atlas_probes'] = {phase:{f.name.removeprefix(phase+'_'):sha(f)
                                            for f in sorted(folder.glob(phase+'_atlas-*.bin'))}
                                     for phase in ('before','after')}
        assert all(len(rows)==4 for rows in identity['atlas_probes'].values())
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
    if variant == 'local-escape':
        fixture = sorted((EARLIER/'cases'/folder.name).glob('m2-*'))
        if fixture:
            from light_sampling_exploration_escape import neighbor_words
            brick = read(fixture[0]/'accepted.json')['pose']['camera_brick']
            key_array = np.asarray(keys,dtype=int)
            words,_ = neighbor_words(key_array,folder/'prefix',brick)
            solid = ((words&1)!=0) | np.isin(words>>1,np.unique(key_array[:,3]))
            expected = mass[:,0]*(1+(~solid).sum(axis=1))/7.
            np.testing.assert_allclose(mass[:,1],expected,rtol=2e-7)
            result['cpu_mass_max_relative_error'] = float(np.max(np.abs(mass[:,1]/expected-1.)))
    if variant == 'texture-mass':
        uv_paths = {phase:folder/variant/f'{phase}_face-uv.bin' for phase in ('before','after')}
        result['uv_tables'] = {phase:sha(p) for phase,p in uv_paths.items()}
        uv = {phase:np.fromfile(p,'<u4')[:16384*6*2].reshape(-1,6,2) for phase,p in uv_paths.items()}
        ids = sorted({key[3] for key in keys})
        result['emitting_material_uvs'] = dict(
            material_count=len(ids), changing_faces=int(np.any(uv['before'][ids]!=uv['after'][ids],axis=2).sum()),
            populated_faces=int(np.any(uv['before'][ids]!=0,axis=2).sum()))
    return result


def connected_surface(folder, dim, center, axis):
    """Get the connected matching planar material/normal at a fixed receiver."""
    from scipy.ndimage import label
    height,width = dim['height'],dim['width']
    x = round(center[0]*width/1920)
    y = round(center[1]*height/1080)
    view = np.memmap(folder/'colortex10.bin','<f4','r',shape=(height,width))[::-1]
    packed = np.memmap(folder/'colortex8.bin','<u4','r',shape=(height,width,4))[::-1]
    camera = np.fromfile(folder/'before_camera.bin','<f4')
    inverse_view = camera[16:32].reshape(4,4).T.astype(float)
    inverse_projection = camera[96:112].reshape(4,4).T.astype(float)
    x_ndc = ((np.arange(width)+.5)*2./width-1.)*inverse_projection[0,0]
    y_ndc = (1.-(np.arange(height)+.5)*2./height)*inverse_projection[1,1]
    direction = inverse_view[axis,2] - (
        inverse_view[axis,0]*x_ndc[None,:] + inverse_view[axis,1]*y_ndc[:,None])
    axis_coordinate = view * direction
    same_surface = np.abs(axis_coordinate-axis_coordinate[y,x]) <= .01
    same_surface &= ((packed[:,:,0]^packed[y,x,0]) & 0xffff)==0
    same_surface &= ((packed[:,:,3]^packed[y,x,3]) >> 16)==0
    same_surface &= np.isfinite(view)&(view>-65536.)
    labels,_ = label(same_surface)
    assert labels[y,x]>0
    return labels==labels[y,x]


def compare(scene, names, full_receiver=False, proposal_mass_variant=None, surface_regions=False):
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
        if surface_regions:
            axes = [0,1] if scene=='light-sampling-1' else [2,0,1]
            points.update({name+' connected plane':connected_surface(folder/names[0],dim,center,axis)
                           for (name,center),axis in zip(centers,axes,strict=True)})
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
    uv_identities = {n:r['identity'].get('face_uv') for n,r in cases.items()}
    emitting_uv_matched = None
    if all(uv_identities.values()):
        emitting_uv_matched = len({uv[p]['emitting_sha256'] for uv in uv_identities.values()
                                   for p in ('before','after')})==1
    fixed_uv = {n:r['identity'].get('frozen_face_uv_sha256') for n,r in cases.items()}
    if any(fixed_uv.values()):
        assert all(fixed_uv.values()) and len(set(fixed_uv.values()))==1, fixed_uv
    probe_matched = None
    if all('atlas_probes' in r['identity'] for r in cases.values()):
        probes = [r['identity']['atlas_probes'][phase] for r in cases.values() for phase in ('before','after')]
        probe_matched = {name:len({row[name] for row in probes})==1 for name in probes[0]}
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
                  receiver_mismatches=mismatches, full_receiver=full_receiver, proposal_mass_change=mass_change,
                  emitting_uv_matched=emitting_uv_matched, frozen_face_uv_sha256=next(iter(fixed_uv.values())),
                  atlas_probe_matched=probe_matched)
    tag = ('full-gbuffer-' if full_receiver else '') + ('surfaces-' if surface_regions else '') + '-'.join(names)
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
    grouped=ops.add_parser('group')
    grouped.add_argument('job_id')
    grouped.add_argument('name')
    motion=ops.add_parser('motion')
    motion.add_argument('job_id')
    motion.add_argument('name')
    report=ops.add_parser('compare')
    report.add_argument('scene')
    report.add_argument('names',nargs='+')
    report.add_argument('--full-receiver',action='store_true')
    report.add_argument('--surface-regions',action='store_true')
    report.add_argument('--proposal-mass-variant',choices=('texture-mass','local-escape'))
    args=parser.parse_args()
    if args.op=='archive':
        result=archive(args.job_id,args.name)
        fields = ('job','scene','name','manifest_id','manifest_sha256','path','identity','timings')
        print(json.dumps({key:result[key] for key in fields}))
    elif args.op=='ordered':
        print(json.dumps(archive_ordered(args.job_id,args.name)))
    elif args.op=='group':
        print(json.dumps(archive_group(args.job_id,args.name)))
    elif args.op=='motion':
        print(json.dumps(archive_motion(args.job_id,args.name)))
    else:
        result=compare(args.scene,args.names,args.full_receiver,args.proposal_mass_variant,args.surface_regions)
        print(result['time_ratios'],result['joint_complete_pixels'])

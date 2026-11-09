"""Archive a pair of multi-window captures and score pooled image differences."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_exploration_capture import RUN, retain_manifest
from light_sampling_moving_analysis import leaves, native_node, world_voxel_identity
from light_sampling_point_moments import ROOT, read, sha
from light_sampling_weight_reference import load_tree


def checkpoint(folder, phase, context):
    header, nodes, links = load_tree(folder/f'{phase}_tree.bin', 6)
    rows = sorted(leaves(native_node(nodes, links, 1)))
    assert len(rows)==len(set(rows))==int(header[1]) and header[2]==header[3]==0
    cam = np.fromfile(folder/f'{phase}_camera.bin','<f4')
    voxel = world_voxel_identity(folder,16,[0,0,0],prefix=phase+'_',context=context)
    return dict(camera_sha256=hashlib.sha256(np.r_[cam[16:32],cam[80:112]].tobytes()).hexdigest(),
                emitters=len(rows),leaf_sha256=hashlib.sha256(np.asarray(rows,'<f8').tobytes()).hexdigest(),
                world_voxel_sha256=voxel['world_blocks_material_sha256'])


def receiver_mask(folder, tags, height, width):
    pair = tags[:2]
    mask = np.ones((height,width),dtype=bool)
    mismatches = {}
    for name,dtype,shape in [('colortex8','<u4',(height,width,4)),('colortex9','<u4',(height,width)),
                             ('colortex10','<f4',(height,width))]:
        a,b = [np.memmap(folder/f'{tag}--{name}.bin',dtype,'r',shape=shape)[::-1] for tag in pair]
        if name=='colortex8':
            match = (((a[:,:,0]^b[:,:,0])&0xffff)==0) & (((a[:,:,3]^b[:,:,3])>>16)==0)
        elif name=='colortex9':
            match = ((a^b)&0xffffff)==0
        else:
            match = a==b
        mismatches[name] = int((~match).sum())
        mask &= match
    return mask,mismatches


def windows(folder, tag, height, width, count):
    fields = {}
    mask = np.ones((height,width),dtype=bool)
    for i in range(1,count+1):
        stem = f'{tag}--window{i:02}'
        meta = read(folder/f'{stem}.mean.json')['accumulation']
        assert meta['frames']==32 and meta['batches']==8
        f = int(meta['start_frame']),int(meta['end_frame'])
        assert f[1]-f[0]==31 and (i==1 or f[0]>fields[i-1][1][1])
        def image(field):
            return np.memmap(folder/f'{stem}.{field}.bin','<f4','r',shape=(height,width,4))[::-1,:,:3]
        mean = image('mean')
        mask &= np.isfinite(mean).all(axis=2)
        for field,value in (('valid_count',32),('valid_batches',8),('incomplete_batches',0)):
            mask &= (image(field)==value).all(axis=2)
        fields[i] = mean,f
    return fields,mask


def pool(mean, mask, size):
    ny,nx = (a//size for a in mask.shape)
    region = mask[:ny*size,:nx*size]
    counts = region.reshape(ny,size,nx,size).sum(axis=(1,3))
    include = counts >= int(np.ceil(.95*size*size))
    values = np.where(region[:,:,None],mean[:ny*size,:nx*size],0.)
    sums = values.reshape(ny,size,nx,size,3).sum(axis=(1,3),dtype=np.float64)
    return sums[include]/counts[include,None]


def stats(x):
    assert len(x)>0 and np.isfinite(x).all()
    k = int(np.ceil(.05*len(x)))
    return dict(mean=float(x.mean()),p95=float(np.quantile(x,.95)),tail=float(np.partition(x,len(x)-k)[-k:].mean()),
                tiles=len(x))


def archive(job_id, name):
    job = ROOT/'.vibris/jobs'/job_id
    req = read(job/'request.json')['arguments']
    result = read(job/'result.json')
    assert result['success']
    result = result['result']
    assert result['restoration']['status']=='RECEIPT_STATUS_OK'
    assert all(r['status']=='RECEIPT_STATUS_OK' for r in result['action_receipts'])
    assert req['preset_id']=='light-sampling-failure-moving-4' and not req.get('unfreeze_world',False)
    folder = RUN/'lowfreq-series'/name
    manifest, ref = retain_manifest(job,result,folder)
    context = req['__vibris_scene_context']
    before,after = [checkpoint(folder,p,context) for p in ('before','after')]
    assert before==after,(before,after)
    src = {r['id']:r for r in req['sources']}
    cfg = {r['id']:r['values'] for r in req['configs']}
    methods = {}
    for action,receipt in zip(req['actions'],result['action_receipts'],strict=True):
        if action['type']!='load_shader':
            continue
        tag = action['config_id']
        source = src[action['source_id']]
        m = receipt['runtime_mutation']
        assert source['snapshot_sha256']==m['source_sha256']
        effective = {v['name']:v['value'] for v in m['effective_settings']['settings']}
        assert all(effective[k]==v for k,v in cfg[tag].items())
        assert effective['SETTING_VOXEL_GRID_SIZE']=='16' and effective['SETTING_DEBUG_LIGHT_AUDIT']=='false'
        methods[tag] = dict(source=source['resolved_revision'],scene_sha256=m['scene_sha256'])
    assert len(methods)==2 and len({m['scene_sha256'] for m in methods.values()})==1
    fixed = [sha(folder/f'{n}--frozen-face-uv.bin') for n in methods]
    assert len(set(fixed))==1
    crops = {}
    for n in methods:
        for phase in ('before','after'):
            crops[n,phase] = {p.name.split('--',2)[2]:sha(p)
                              for p in sorted(folder.glob(f'{n}--{phase}--atlas-*.bin'))}
            assert len(crops[n,phase])==4
    probe_match = {k:len({row[k] for row in crops.values()})==1 for k in next(iter(crops.values()))}
    dim = context['resolution']
    height,width = dim['height'],dim['width']
    mask, mismatches = receiver_mask(folder,list(methods),height,width)
    raw = {}
    sizes = {sum(a['type']=='accumulate_texture' and a['artifact_name'].startswith(tag+'--window')
                 for a in req['actions']) for tag in methods}
    assert len(sizes)==1 and sizes <= {4,6}
    count = sizes.pop()
    for tag in methods:
        raw[tag],valid = windows(folder,tag,height,width,count)
        mask &= valid
    assert mask.any()
    for tag in methods:
        methods[tag]['frames'] = [f for _,f in raw[tag].values()]
        methods[tag]['tile_noise'] = {}
        for s in (16,64):
            pooled = [pool(mean,mask,s) for mean,_ in raw[tag].values()]
            mse = [np.square(a-b).sum(axis=1)/2. for a,b in zip(pooled[::2],pooled[1::2],strict=True)]
            methods[tag]['tile_noise'][str(s)] = dict(pairs=[stats(y) for y in mse],average_of_pairs=stats(np.mean(mse,axis=0)))
    report = dict(job=job_id,scene=req['preset_id'],manifest_id=manifest['manifest_id'],manifest_sha256=ref['sha256'],
                  resolution=dim,identity=before,matched_complete_pixels=int(mask.sum()),receiver_mismatches=mismatches,
                  fixed_uv_sha256=fixed[0],atlas_crop_match=probe_match,methods=methods,
                  windows_per_method=count,
                  scope='Disjoint 32f mean windows, paired in acquisition order. Tile ratios are radiance differences, not a timing benchmark. Matching atlas crops do not certify all emitters or intermediate frames.')
    (folder/'accepted.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return report


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('job_id')
    p.add_argument('name')
    a = p.parse_args()
    print(json.dumps(archive(a.job_id,a.name)))

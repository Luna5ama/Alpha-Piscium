"""Analysis and independent PDF checks for the visibility-free geometry matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
from light_sampling_runtime import preserve, read_image, render, tree_stats, LUMA


def cube_faces(low, origin, normal):
    faces=[]
    for axis in range(3):
        n=np.eye(3)[axis];o=origin[axis]
        if low[axis]<=o<=low[axis]+1:continue
        center=low+.5;center[axis]=low[axis]+(o>low[axis])
        u=np.roll(n,1)*.5;v=np.roll(n,2)*(.5 if o<low[axis] else -.5)
        vertices=[center-u-v-origin,center+u-v-origin,center+u+v-origin,center-u+v-origin]
        clipped=[]
        for a,b in zip(vertices,vertices[1:]+vertices[:1]):
            za,zb=a@normal,b@normal
            if za>=0:clipped.append(a)
            if za*zb<0:clipped.append(a+(b-a)*za/(za-zb))
        if len(clipped)<3:continue
        vertices=np.array(clipped);directions=vertices/np.linalg.norm(vertices,axis=1)[:,None]
        h=0.
        for a,b in zip(directions,np.roll(directions,-1,axis=0)):
            cross=np.cross(a,b);sine=np.linalg.norm(cross)
            if sine:h+=.5*(cross@normal)*np.arctan2(sine,a@b)/sine
        area=0.;area_center=np.zeros(3)
        for i in range(1,len(vertices)-1):
            a,b,c=vertices[0],vertices[i],vertices[i+1]
            triangle_area=.5*np.linalg.norm(np.cross(b-a,c-a))
            area+=triangle_area;area_center+=triangle_area*(a+b+c)/3
        if area==0:continue
        area_center/=area;h=max(h,0.)
        tiny=h<1e-5;d2=area_center@area_center
        weight=area*max(area_center@normal,0)*abs(n@area_center)/d2**2 if tiny else h
        faces.append(dict(axis=axis,h=h,weight=weight,area=area,tiny=tiny))
    return faces


def cube_integral(low, origin, normal):
    return sum(f['h'] for f in cube_faces(low,origin,normal))


def cube_pdf(low, origin, normal, direction):
    faces=cube_faces(low,origin,normal)
    entry=np.minimum((low-origin)/direction,(low+1-origin)/direction)
    axis=int(np.argmax(entry));distance=entry[axis]
    face=next(f for f in faces if f['axis']==axis)
    conditional=distance**2/(face['area']*abs(direction[axis])) if face['tiny'] else (normal@direction)/face['h']
    return face['weight']/sum(f['weight'] for f in faces)*conditional


def audit_cube(path):
    data = np.fromfile(path, '<f4').reshape(64, 4, 4).astype(float)
    errors = []
    for origin4, normal4, direction4, low4 in data:
        if direction4[3] <= 0:
            continue
        origin, normal, direction, low = origin4[:3], normal4[:3], direction4[:3], low4[:3]
        assert np.isfinite(direction4).all()
        assert abs(np.linalg.norm(direction)-1) < 2e-5
        expected = cube_pdf(low.copy(), origin, normal, direction)
        errors.append(abs(direction4[3]/expected-1))
        t0, t1 = (low-origin)/direction, (low+1-origin)/direction
        assert np.max(np.minimum(t0,t1)) <= np.min(np.maximum(t0,t1)) + 1e-4
    assert errors
    return dict(samples=len(errors), max_relative_pdf_error=max(errors), p95_relative_pdf_error=float(np.quantile(errors,.95)))

def audit_tree(tree_path, audit_path, improved, lookahead):
    words = np.fromfile(tree_path, '<u4')[8:].reshape(-1,8)
    nodes = words.view('<f4').astype(float)
    parents, leaves = {}, {}
    pending = [1]
    while pending:
        index = pending.pop()
        if nodes[index,3] == 0:
            continue
        link = int(words[index,7])
        if link & 0x80000000:
            leaves[tuple(nodes[index,:3])] = index
        else:
            parents[link] = parents[link+1] = index
            pending.extend((link,link+1))
    records=np.fromfile(audit_path,'<f4').reshape(64,4,4).astype(float)
    errors=[]
    for record in records:
        if record[3,3] == 0:
            continue
        origin,normal=record[0,:3],record[1,:3]
        def score(index, depth):
            node=nodes[index]; mass=node[3];link=int(words[index,7])
            if mass == 0:
                return 0.
            if depth and not link & 0x80000000:
                return score(link,depth-1)+score(link+1,depth-1)
            half=(node[4:7]-node[:3])*.5
            delta=node[:3]+half-origin
            d2=delta@delta;r2=half@half
            if improved:
                e=abs(normal)@half;h=normal@delta
                h=h if h>=e else (0 if h<=-e else (h+e)**2/(4*e))
                scale=d2+r2/3
                return mass*max(np.clip(h/np.sqrt(scale),0,1),.001)/scale
            cosine=np.clip((normal@delta+np.sqrt(r2))/np.sqrt(max(d2,.25)),.001,1)
            return mass*cosine/max(d2,r2,.25)
        index=leaves[tuple(record[3,:3])];pdf=1.
        while index!=1:
            parent=parents[index];left=int(words[parent,7]);depth=2 if lookahead else 0
            a,b=score(left,depth),score(left+1,depth)
            pdf*=a/(a+b) if index==left else b/(a+b)
            index=parent
        errors.append(abs(record[1,3]/pdf-1))
    assert errors and max(errors)<.005, errors
    return dict(samples=len(errors),max_relative_pdf_error=max(errors))

def analyze(directory, selected_scene=None):
    from scipy.ndimage import gaussian_filter
    spec=json.loads((directory/'matrix.json').read_text(encoding='utf-8-sig'))
    rows=[];images=[];sources=set()
    for scene in spec['scenes']:
        if selected_scene and selected_scene != scene['id']:
            continue
        captured=[];mask=None;leaf_set=None
        for case in spec['cases']:
            raw=directory/'raw'/scene['id']/case['id']
            job=json.loads((raw/'job-result.json').read_text())['result']
            mutation=job['action_receipts'][0]['runtime_mutation']
            settings={s['name']:s['value'] for s in mutation['effective_settings']['settings']}
            expected=dict(spec['common_settings'],SETTING_DEBUG_LIGHT_SAMPLING='1' if case['sampler']=='brdf' else '2',
                SETTING_DEBUG_LIGHT_CUBE=str(case['cube']).lower(),SETTING_DEBUG_LIGHT_WEIGHT=str(case['weight']).lower(),
                SETTING_DEBUG_LIGHT_LOOKAHEAD=str(case['lookahead']).lower(),SETTING_DEBUG_LIGHT_AUDIT='true')
            assert all(settings[k]==v for k,v in expected.items()), (scene,case)
            assert 'SETTING_DEBUG_LIGHT_VISIBILITY' not in settings
            single,sm=read_image(raw/'single.bin');mean,mm=read_image(raw/'mean32.mean.bin');var,_=read_image(raw/'mean32.variance.bin')
            count,_=read_image(raw/'mean32.valid_count.bin');inc,_=read_image(raw/'mean32.incomplete_batches.bin')
            assert (count==32).all() and (inc==0).all() and (var>=0).all() and (var[:,:,3]==0).all()
            accum=mm['accumulation'];assert accum['frames']==32 and accum['end_frame']-accum['start_frame']==31
            assert sm['frame_id']<accum['start_frame']
            current_mask=mean[:,:,3]>0
            if mask is None:mask=current_mask
            assert np.array_equal(mask,current_mask)
            tree=tree_stats(raw/'tree.bin')
            tree['reserved']=tree.pop('guided_selections')
            assert json.loads((raw/'tree.json').read_text())['frame_id']==json.loads((raw/'audit.json').read_text())['frame_id']
            nodewords=np.fromfile(raw/'tree.bin','<u4')[8:].reshape(-1,8);nodevalues=nodewords.view('<f4')
            leaves=[];pending=[1]
            while pending:
                index=pending.pop();n=nodevalues[index];link=int(nodewords[index,7])
                if n[3]==0:continue
                if link&0x80000000:leaves.append((tuple(n[:3]),link))
                else:pending.extend((link,link+1))
            leaves.sort()
            # Allocation may differ; the actual emitter positions and materials must agree.
            if leaf_set is None:leaf_set=leaves
            assert leaves==leaf_set
            audit=None if case['sampler']=='brdf' else audit_tree(raw/'tree.bin',raw/'audit.bin',case['weight'],case['lookahead'])
            cube_audit=audit_cube(raw/'audit.bin') if case['cube'] else None
            batches=np.array(accum['screen_batch_means'])[:,:3]@LUMA
            row=dict(case,scene=scene['id'],source_sha256=mutation['source_sha256'],scene_sha256=mutation['scene_sha256'],
                     settings_sha256=mutation['effective_settings']['settings_sha256'],mean_luminance=float(batches.mean()),
                     mean_luminance_se=float(batches.std(ddof=1)/np.sqrt(len(batches))),tree=tree,pdf_audit=audit,cube_audit=cube_audit,
                     single_frame=sm['frame_id'],mean_frames=[accum['start_frame'],accum['end_frame']])
            sources.add(row['source_sha256']);rows.append(row);captured.append((case,mean,var,row))
            for label,data,tag in [('1 spp single frame',single,'single'),('32 frame mean',mean,'mean32')]:
                filename=f"{scene['id']}-{case['id']}-{tag}.png"
                render(data,directory/filename,spec['exposure_ev'])
                for ahead in ([False,True] if case['sampler']=='brdf' else [case['lookahead']]):
                    images.append(dict(path=filename,dimensions={'Scene':scene['label'],'Variant':case['label'],
                        'Lookahead':'Two levels' if ahead else 'Off','Accumulation':label}))
        pooled=np.mean([m[:,:,:3] for _,m,_,_ in captured],axis=0,dtype=np.float64)
        pooled=gaussian_filter(pooled,sigma=(2,2,0))
        lum=pooled@LUMA;lit=mask&(lum>1e-6)
        q50,q99=np.quantile(lum[lit],[.5,.99])
        regions={'all_receivers':mask,'dark_half':lit&(lum<=q50),'middle_49':lit&(lum>q50)&(lum<=q99),'brightest_1':lit&(lum>q99)}
        sensitivity=2**spec['exposure_ev']/(1+2**spec['exposure_ev']*np.maximum(pooled,0))**2
        for case,mean,var,row in captured:
            row['regions']={key:dict(pixels=int(region.sum()),rgb_variance=float(var[:,:,:3][region].sum(axis=1,dtype=np.float64).mean()),
                display_variance_proxy=float((var[:,:,:3]*sensitivity**2)[region].sum(axis=1,dtype=np.float64).mean())) for key,region in regions.items()}
            total=var[:,:,:3].sum(axis=2,dtype=np.float64)[mask];cut=np.quantile(total,.99)
            row['top_1pct_variance_share']=float(total[total>=cut].sum()/total.sum())
        scene_rows=[x[3] for x in captured];brdf=next(r for r in scene_rows if r['sampler']=='brdf')
        for row in scene_rows:
            se=np.hypot(row['mean_luminance_se'],brdf['mean_luminance_se'])
            row['mean_difference_z_vs_brdf']=(row['mean_luminance']-brdf['mean_luminance'])/se if se else 0
            row['relative_mean_difference_vs_brdf']=row['mean_luminance']/brdf['mean_luminance']-1
            for region,metrics in row['regions'].items():
                metrics['variance_over_brdf']=metrics['rgb_variance']/brdf['regions'][region]['rgb_variance']
    assert len(sources)==1
    suffix='-'+selected_scene if selected_scene else ''
    (directory/f'summary{suffix}.json').write_text(json.dumps(rows,indent=2))
    (directory/f'images{suffix}.json').write_text(json.dumps(dict(title='Cube Sampling and Tree Weights vs Two-Level Lookahead - Diffuse Direct Only',images=images),indent=2))
    print(json.dumps([dict(scene=r['scene'],case=r['id'],mean=r['mean_luminance'],z=r['mean_difference_z_vs_brdf'],
         dark=r['regions']['dark_half']['variance_over_brdf'],bright=r['regions']['brightest_1']['variance_over_brdf'],cube_audit=r['cube_audit']) for r in rows],indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--scene')
    args=parser.parse_args()
    if args.audit:print(json.dumps(audit_cube(args.audit), indent=2))
    if args.analyze:analyze(args.analyze,args.scene)

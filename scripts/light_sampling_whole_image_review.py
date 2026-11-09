"""Reuse matched Iris frames to compare full images and absolute noise budgets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'builds/light-sampling'
OUT = RUN / '20261108-whole-image-review'
GENERATOR = Path('J:/.codex/skills/pyxis-image-compare/scripts/generate_compare.py')
SHAPE = (1080, 1920, 4)
BLOCK = 16
NAMES = {'prefix': 'World Prefix + A + Near/Far', 'camera': 'Camera grouping + A + Near/Far',
         'sah': 'Global area Light-SAH + A + Near/Far', 'brdf': 'Cosine BRDF-only'}
VIEWS = ('32-frame mean', '1 spp', 'Absolute noise (16px tiles)', 'Absolute variance change vs Prefix')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_array(folder, name):
    path = folder / name
    meta = read(path.with_suffix('.json'))
    assert (meta['width'], meta['height'], meta['channel_count'], meta['scalar_type']) == (1920,1080,4,'FLOAT32')
    assert path.stat().st_size == 1920*1080*16
    return np.memmap(path, '<f4', 'r', shape=SHAPE)


def tone(rgb):
    x = np.maximum(np.nan_to_num(rgb.astype(float), nan=0, posinf=0, neginf=0), 0)*16
    x /= 1+x
    return np.rint(np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)*255).astype('uint8')


def block_sum(values):
    padded = np.pad(values, ((0,8),(0,0)), constant_values=0)
    return padded.reshape(68,BLOCK,120,BLOCK).sum(axis=(1,3), dtype=float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT/'png'
    png.mkdir(exist_ok=True)
    entries = read(RUN/'20261108-grouping-control/cases.json')
    sah = read(RUN/'20261108-clustering-repair/cases/light-sampling-5/sah-6a254ce1/accepted.json')
    scenes = {}
    sources = {}
    image_entries = []
    fields = ('mean','variance','valid_count','valid_batches','incomplete_batches','nonfinite')
    for scene in (1,5):
        cases = {c['method']: c for c in entries if c['scene']==scene}
        if scene==5: cases['sah'] = sah
        assert len({c['identity']['camera_sha256'] for c in cases.values()})==1
        assert len({c['identity']['leaf_sha256'] for c in cases.values()})==1
        assert len({c['identity']['world_voxel_sha256'] for c in cases.values()})==1
        assert all(c['window']['frames']==32 for c in cases.values())
        arrays = {}
        common = np.ones(SHAPE[:2],bool)
        receivers = common.copy()
        for method,case in cases.items():
            folder = ROOT/case['path']
            a = {f:image_array(folder,'quality.'+f+'.bin') for f in fields}
            receiver = np.isfinite(a['mean'][:,:,3]) & (a['mean'][:,:,3]>0)
            finite = np.isfinite(a['mean'][:,:,:3]).all(2) & np.isfinite(a['variance'][:,:,:3]).all(2)
            good = receiver & finite
            for f,value in [('valid_count',32),('valid_batches',8),('incomplete_batches',0),('nonfinite',0)]:
                good &= (a[f][:,:,:3]==value).all(2)
            common &= good
            receivers &= receiver
            arrays[method] = a
            sources[f'{scene}/{method}'] = dict(path=case['path'], identity=case['identity'], window=case['window'],
                raw_sha256={f:sha(folder/('quality.'+f+'.bin')) for f in fields}, single_sha256=sha(folder/'single.bin'))
        # All map coordinates are upright display pixels, including partial bottom tiles.
        mask = common[::-1]
        counts = block_sum(mask)
        valid = counts>0
        methods = {}
        for method in NAMES:
            if method not in cases: continue
            a = arrays[method]
            var = a['variance'][::-1,:,:3].sum(axis=2,dtype=float)
            assert np.all(var[mask]>=0)
            tiles = np.divide(block_sum(np.where(mask,var,0)),counts,out=np.zeros_like(counts),where=valid)
            mu = a['mean'][::-1,:,:3]
            tail_count = max(1,int(np.ceil(valid.sum()*.05)))
            top = np.flatnonzero(valid.ravel())[np.argsort(tiles[valid])[-tail_count:]]
            zero = (mu==0).all(2) & mask
            methods[method] = dict(tile_variance=tiles, variance=float(var[mask].mean()),
                rms_noise=float(np.sqrt(var[mask].mean())), tile_p95=float(np.quantile(tiles[valid],.95)),
                worst5_tile_variance=float(np.average(tiles.ravel()[top],weights=counts.ravel()[top])),
                zero_mean_fraction=float(zero.sum()/mask.sum()), mean_rgb=mu[mask].mean(axis=0,dtype=float).tolist())
            folder = ROOT/cases[method]['path']
            for view,rawname in [(VIEWS[0],'quality.mean.bin'),(VIEWS[1],'single.bin')]:
                suffix = 'mean' if rawname.startswith('quality') else 'single'
                old_png = RUN/f'20261108-grouping-control/viewer/png/{scene}-{method}-{rawname}.png'
                if method=='sah': old_png=RUN/f'20261108-clustering-repair/viewer/comparison.png/sah-{suffix}.png'
                expected = tone(image_array(folder,rawname)[::-1,:,:3])
                np.testing.assert_array_equal(np.asarray(Image.open(old_png).convert('RGB')),expected)
                target = png/f'ls{scene}-{method}-{suffix}.png'
                target.write_bytes(old_png.read_bytes())
                image_entries.append(dict(path=str(target),dimensions={'Scene':f'light-sampling-{scene}','Method':NAMES[method],'View':view}))
        prefix = methods['prefix']['tile_variance']
        for method,m in methods.items():
            delta = m['tile_variance']-prefix
            m.update(delta_variance=m['variance']-methods['prefix']['variance'],
                variance_ratio=m['variance']/methods['prefix']['variance'],
                tile_delta=delta,
                positive_variance_budget=float(np.sum(np.maximum(delta,0)*counts)/mask.sum()),
                negative_variance_budget=float(np.sum(np.minimum(delta,0)*counts)/mask.sum()))
        scenes[str(scene)]=dict(methods=methods,mask=mask,counts=counts,
            common_pixels=int(mask.sum()),excluded_receivers=int(receivers.sum()-mask.sum()),
            identity={key:next(iter(cases.values()))['identity'][key]
                      for key in ('camera_sha256','leaf_sha256','world_voxel_sha256')})
    magnitude = np.concatenate([np.sqrt(m['tile_variance'][s['counts']>0]) for s in scenes.values()
                                for name,m in s['methods'].items() if name!='brdf'])
    differences = np.concatenate([np.abs(m['tile_delta'][s['counts']>0]) for s in scenes.values()
                                  for name,m in s['methods'].items() if name not in ('prefix','brdf')])
    sigma_max = float(np.quantile(magnitude,.995))
    delta_max = float(np.quantile(differences,.995))
    sigma_soft, delta_soft = sigma_max/1000, delta_max/1000
    for scene,s in scenes.items():
        for method,m in s['methods'].items():
            sigma=np.sqrt(m['tile_variance'])
            norm=np.log1p(sigma/sigma_soft)/np.log1p(sigma_max/sigma_soft)
            delta=m['tile_delta']
            signed=np.sign(delta)*np.log1p(np.abs(delta)/delta_soft)/np.log1p(delta_max/delta_soft)
            for view,values,cmap in [(VIEWS[2],norm,'magma'),(VIEWS[3],.5+.5*signed,'RdBu_r')]:
                colors=np.rint(matplotlib.colormaps[cmap](np.clip(values,0,1))[:,:,:3]*255).astype('uint8')
                colors=colors.repeat(BLOCK,axis=0).repeat(BLOCK,axis=1)[:1080,:1920]
                colors[~s['mask']]=[70,70,70]
                target=png/f"ls{scene}-{method}-{'noise' if view==VIEWS[2] else 'delta'}.png"
                Image.fromarray(colors).save(target)
                image_entries.append(dict(path=str(target),dimensions={'Scene':f'light-sampling-{scene}','Method':NAMES[method],'View':view}))
            m['noise_map_saturated_fraction']=float(np.sum(s['counts'][sigma>sigma_max])/s['common_pixels'])
            m['delta_map_saturated_fraction']=float(np.sum(s['counts'][np.abs(delta)>delta_max])/s['common_pixels'])
    # Reuse original decoded-equal AVIFs; encode only newly drawn diagnostic maps.
    spec=importlib.util.spec_from_file_location('comparison_generator',GENERATOR)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    cache={}
    for page in [RUN/'20261108-grouping-control/viewer/comparison.html',RUN/'20261108-clustering-repair/viewer/comparison.html']:
        data=read_payload(page)
        for item in data['images']:
            source=Path(item['label']);asset=page.parent/item['src']
            np.testing.assert_array_equal(np.asarray(Image.open(source).convert('RGB')),np.asarray(Image.open(asset).convert('RGB')))
            cache[sha(source)]=asset
    original_encoder=generator.encode_avif
    def encode(path,encoder,quality):
        cached=cache.get(sha(path))
        return cached.read_bytes() if cached else original_encoder(path,encoder,quality)
    generator.encode_avif=encode
    # The first entry opens a complete ls5 frame, without selecting a diagnostic point.
    scene_order={'light-sampling-5':0,'light-sampling-1':1}
    method_order={n:i for i,n in enumerate(NAMES.values())}
    image_entries.sort(key=lambda e:(scene_order[e['dimensions']['Scene']],method_order[e['dimensions']['Method']],VIEWS.index(e['dimensions']['View'])))
    manifest=OUT/'manifest.json'
    manifest.write_text(json.dumps(dict(title='Light sampling: full images and absolute noise',images=image_entries),indent=2)+'\n',encoding='utf-8')
    sys.argv=[str(GENERATOR),str(manifest),'-o',str(OUT/'comparison.html'),'--storage','external']
    assert generator.main()==0
    summaries={scene:{'common_pixels':s['common_pixels'],'excluded_receivers':s['excluded_receivers'],'identity':s['identity'],
        'methods':{name:{k:v for k,v in m.items() if k not in ('tile_variance','tile_delta')} for name,m in s['methods'].items()}}
        for scene,s in scenes.items()}
    rule=read(ROOT/'experiments/light-sampling/20261108-clustering-repair.json')['clustering_rule_screen']
    column=rule['method_columns'].index('variance')
    points=[]
    for row in rule['windows']:
        if row['cohort']=='new/1':
            base=row['methods']['prefix'][column]; candidate=row['methods']['upper/flux-diagonal'][column]
            points.append(dict(point=row['point'],window=row['window'],prefix_variance=base,upper_variance=candidate,delta=candidate-base,ratio=candidate/base))
    stats=dict(scope='32-frame sample variance in linear sRGB, summed across RGB. Common complete receiver mask, no brightness filter. Not error against ground truth; zero observed mean is not proof of no light.',
        tile_size=BLOCK,tail='Each method: weighted mean variance of its highest-variance 5% of nonempty 16x16 tiles; sets may differ.',
        display='Original 1-spp and 32-frame PNGs unchanged: x16 exposure, Reinhard, sRGB; all rows verified against raw[1079-display_y].',
        maps=dict(noise='sqrt(tile mean RGB variance)',delta='tile mean RGB variance minus Prefix',
            scale='log1p / signed log1p, fixed across both scenes and all methods; limits are pooled tree-method P99.5, not per-method autoscaling.',
            sigma_max=sigma_max,delta_max=delta_max,sigma_soft=sigma_soft,delta_soft=delta_soft,excluded_color=[70,70,70]),
        missing=['upper/flux-diagonal: no full-frame capture','flat: no full-frame capture','global area Light-SAH: no ls1 full-frame capture'],
        point_only_separate_cohort=dict(world_voxel='4dfca2b6ba4714126b4962a206a4bda14e358bf9a33eaa67ac85bfb99e9d61df',values=points),
        scenes=summaries,sources=sources)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    ui=dict(scenes={f'light-sampling-{scene}':dict(common=s['common_pixels'],excluded=s['excluded_receivers'],
        counts=s['counts'].astype(int).ravel().tolist(),methods={NAMES[k]:dict(**summaries[scene]['methods'][k],
        tiles=np.round(m['tile_variance'],12).ravel().tolist()) for k,m in s['methods'].items()}) for scene,s in scenes.items()},
        sigma_max=sigma_max,delta_max=delta_max)
    page=OUT/'comparison.html'; text=page.read_text(encoding='utf-8')
    data=read_payload(page)
    data['dimensions'][1]['values'] += ['Upper flux-diagonal (no image yet)','Flat reference (no image yet)']
    payload=json.dumps(data,separators=(',',':')).replace('<','\\u003c')
    text=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],text,flags=re.S)
    text=text.replace('<div id="dimensions">',HEADER+'\n<div id="dimensions">')
    text=text.replace('<label>Comparison dimension', '<button id="fit">Fit whole image</button><label>Comparison dimension')
    text=text.replace("const names = data.dimensions",'const absoluteData = '+json.dumps(ui,separators=(',',':'))+';\nconst names = data.dimensions')
    text=text.replace('initialized = true;', 'initialized = true; setScale(Math.min(viewport.clientWidth/shownImage.naturalWidth, viewport.clientHeight/shownImage.naturalHeight));')
    text=text.replace('function render() {','function render() {\n  updateAbsoluteSummary();')
    text=text.replace('render();\n</script>',JS+'\ncompare.value = "1";\nrender();\n</script>')
    text=text.replace('</style>',STYLE+'\n</style>')
    page.write_text(text,encoding='utf-8')
    verification=[]
    for item in data['images']:
        source=Path(item['label']);asset=OUT/item['src']
        np.testing.assert_array_equal(np.asarray(Image.open(source).convert('RGB')),np.asarray(Image.open(asset).convert('RGB')))
        verification.append(dict(values=item['values'],png_sha256=sha(source),avif_sha256=sha(asset)))
    for scene in (1,5):
        columns=['camera','prefix','sah'] if scene==5 else ['camera','prefix','brdf']
        sheet=Image.new('RGB',(1920,884),'#202123');draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',19)
        for col,name in enumerate(columns):
            draw.text((640*col+10,8),NAMES[name],fill='white',font=font)
            for row,view in enumerate(['mean','noise']):
                im=Image.open(png/f'ls{scene}-{name}-{view}.png').resize((640,360),Image.Resampling.NEAREST)
                sheet.paste(im,(640*col,40+row*390))
                label='32-frame mean' if row==0 else 'Absolute RGB noise; one shared color scale'
                draw.text((640*col+10,405+row*390),label,fill='white',font=font)
        strip=np.rint(matplotlib.colormaps['magma'](np.linspace(0,1,780))[None,:,:3]*255).astype('uint8')
        sheet.paste(Image.fromarray(strip).resize((780,16),Image.Resampling.NEAREST),(550,832))
        draw.text((10,826),'Noise magnitude, shared log scale:',fill='white',font=font)
        for value in [0,sigma_soft,sigma_soft*10,sigma_soft*100,sigma_max]:
            x=550+780*np.log1p(value/sigma_soft)/np.log1p(sigma_max/sigma_soft)
            draw.text((x-20,853),f'{value:.3g}',fill='white',font=font)
        sheet.save(OUT/f'ls{scene}-overview.png')
    (OUT/'verification.json').write_text(json.dumps(dict(images=len(verification),images_verified=verification,
        raw_orientation_and_original_pngs='all pixels exact for 14 source images',html_sha256=sha(page),
        browser='Not run: local browser use disabled in Work settings.',script_sha256=sha(Path(__file__))),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(page),images=len(verification),scenes=summaries,point_absolute=points),ensure_ascii=False))


def read_payload(page):
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',page.read_text(encoding='utf-8'),re.S)[1])


HEADER = '''<h1 style="font-size:18px;margin:0">Full images and absolute noise</h1>
<div>Same-input comparisons within each scene · All tree methods use Half-space A + Near/Far · No fixed-point crop</div>
<div style="color:#ffd089">Upper flux-diagonal and flat have no full-frame capture. The ls5 SAH image is the earlier global area candidate.</div>
<details><summary>Absolute statistics and coverage</summary><div id="absolute-table"></div></details>
<div id="map-guide"></div><div id="pixel-readout">Hover anywhere to inspect the absolute variance of its 16×16 tile.</div>
<div style="font-size:12px;color:#b8babf">32-frame observations, not ground-truth error. Zero samples can miss rare light. Same display curve for original images. <a href="statistics.json">Full statistics / sources</a></div>'''
STYLE='''h1{line-height:1.4}a{color:#a9d2ff}#absolute-table table{border-collapse:collapse;font-size:12px;width:100%;margin:4px 0}#absolute-table td,#absolute-table th{padding:3px 7px;text-align:right;border-bottom:1px solid #555}#absolute-table td:first-child,#absolute-table th:first-child{text-align:left}#map-guide,#pixel-readout{font-size:12px;min-height:16px}details{max-height:190px;overflow:auto}'''
JS=r'''
function number(x) { return Number.isFinite(x) ? x.toExponential(4) : 'n/a'; }
function updateAbsoluteSummary() {
  const scene=absoluteData.scenes[current[0]], method=scene.methods[current[1]];
  let table='<table><tr><th>Method</th><th>Mean RGB variance</th><th>Absolute change</th><th>Worst 5% tiles</th><th>Zero 32-frame mean</th></tr>';
  for (const [name,m] of Object.entries(scene.methods)) table+=`<tr><td>${name}</td><td>${number(m.variance)}</td><td>${number(m.delta_variance)}</td><td>${number(m.worst5_tile_variance)}</td><td>${(100*m.zero_mean_fraction).toFixed(2)}%</td></tr>`;
  document.querySelector('#absolute-table').innerHTML=table+`</table>${scene.common.toLocaleString()} common complete receivers; ${scene.excluded} receiver pixels excluded. Worst 5% sets are selected independently for each method.`;
  const guide=document.querySelector('#map-guide');
  if (!method) guide.textContent='No capture for this method/scene. Fixed-point predictions cannot supply this image.';
  else if(current[2].startsWith('Absolute noise')) guide.innerHTML=`<span style="display:inline-block;width:160px;height:10px;background:linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf)"></span> sqrt(mean tile RGB variance), 0 → ${number(absoluteData.sigma_max)}; shared log scale, ${(100*method.noise_map_saturated_fraction).toFixed(2)}% saturated; gray = excluded.`;
  else if(current[2].startsWith('Absolute variance')) guide.innerHTML=`<span style="display:inline-block;width:160px;height:10px;background:linear-gradient(90deg,#053061,#f7f7f7,#67001f)"></span> Blue = less variance; white = zero change; red = more. Shared signed-log range ±${number(absoluteData.delta_max)}; ${(100*method.delta_map_saturated_fraction).toFixed(2)}% saturated. No division by baseline variance.`;
  else guide.textContent='Original direct-light capture. Exposure ×16 and the same tone curve for every method. Fit shows the whole image; 1× shows native pixels.';
}
document.querySelector('#fit').addEventListener('click',()=>{if(!shownImage)return;centerX=shownImage.naturalWidth/2;centerY=shownImage.naturalHeight/2;setScale(Math.min(viewport.clientWidth/shownImage.naturalWidth,viewport.clientHeight/shownImage.naturalHeight));});
viewport.addEventListener('pointermove',event=>{
  const rect=viewport.getBoundingClientRect();
  const x=Math.floor(centerX+(event.clientX-rect.left-rect.width/2)/scale), y=Math.floor(centerY+(event.clientY-rect.top-rect.height/2)/scale);
  if(x<0||y<0||x>=1920||y>=1080)return;
  const scene=absoluteData.scenes[current[0]],method=scene.methods[current[1]],idx=Math.floor(y/16)*120+Math.floor(x/16);
  if(!method)return;
  const base=scene.methods['World Prefix + A + Near/Far'].tiles[idx],value=method.tiles[idx];
  document.querySelector('#pixel-readout').textContent=`Display (${x},${y}), 16×16 tile: ${scene.counts[idx]} complete receivers · Prefix ${number(base)} · Selected ${number(value)} · Absolute change ${number(value-base)}`;
});
'''


if __name__=='__main__': main()

"""Compare absolute spatial noise distributions and the reported ls5 surfaces."""
import importlib.util
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as Polygon
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from PIL import Image

from light_sampling_whole_image_review import ROOT, OUT, GENERATOR, NAMES, SHAPE, image_array, block_sum, read, read_payload, sha

REGIONS = {
    'A: left sandstone wall': [(25,225),(175,245),(175,360),(25,360)],
    'B: blue-lit pillar': [(1118,440),(1157,431),(1121,599),(1089,630)],
    'C: blue floor below pillar': [(1005,678),(1032,697),(969,725),(919,733),(894,721)],
    'D: blue floor in right shadow': [(1330,735),(1410,779),(1371,888),(1326,840),(1302,800)],
}
METRICS = ['mean','p05','p50','p95','p99','worst05_mean','spatial_std','zero_mean_fraction']
PERCENTILES = np.r_[np.linspace(0,95,191), np.linspace(95.1,99.9,97)]


def summarize(variance, mean):
    v = np.sort(variance.astype(float))
    top = max(1, int(np.ceil(v.size*.05)))
    q = np.quantile(v, [.05,.5,.95,.99])
    return dict(mean=float(v.mean()), p05=float(q[0]), p50=float(q[1]), p95=float(q[2]), p99=float(q[3]),
                worst05_mean=float(v[-top:].mean()), spatial_std=float(v.std(ddof=0)),
                zero_mean_fraction=float((mean==0).mean()))


def main():
    source = read(OUT/'statistics.json')
    original_page = OUT/'comparison.html'
    page_text = original_page.read_text(encoding='utf-8')
    data = read_payload(original_page)
    ui = json.loads(re.search(r'const absoluteData = (.*);\nconst names', page_text)[1])
    view = 'Absolute variance change vs Camera'
    data['dimensions'][2]['values'].append(view)
    spec = importlib.util.spec_from_file_location('comparison_generator', GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    encoder = generator.find_encoder()
    assets = OUT/'distribution-review.assets'
    assets.mkdir(exist_ok=True)
    yy, xx = np.indices(SHAPE[:2])
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    polygons = {name:Polygon(vertices).contains_points(pixels).reshape(SHAPE[:2]) for name,vertices in REGIONS.items()}
    output = dict(definition='Per-pixel V = Var(R)+Var(G)+Var(B) from 32-frame samples in linear sRGB. '
                  'Spatial statistics use the same complete receiver mask; no brightness filter. '
                  'Worst 5% = largest ceil(N*0.05) pixel variances, independently sorted for each method. '
                  'Spatial standard deviation is population std of V, not sqrt(mean(V)). '
                  'Zero observed contribution can hide rare light; none of these statistics is error against truth.',
                  source_statistics_sha256=sha(OUT/'statistics.json'), original_page_sha256=sha(original_page),
                  script_sha256=sha(Path(__file__)), region_selection='Fixed interior polygons from the user description, '
                  'chosen before inspecting their statistics. Nonreceiver pixels remain excluded.',
                  regions_display_xy=REGIONS, scenes={}, validation=dict(new_runtime_jobs=0, old_page_unchanged=True))
    curves = {}
    new_images = []
    for scene in ('1','5'):
        cases = {key.split('/')[1]:value for key,value in source['sources'].items() if key.startswith(scene+'/')}
        arrays = {}
        common = np.ones(SHAPE[:2], bool)
        for name,case in cases.items():
            folder = ROOT/case['path']
            a = {field:image_array(folder, 'quality.'+field+'.bin')[::-1] for field in case['raw_sha256']}
            for field in ('mean','variance'):
                assert sha(folder/('quality.'+field+'.bin')) == case['raw_sha256'][field]
            good = (a['mean'][:,:,3]>0) & np.isfinite(a['mean'][:,:,:3]).all(2) & np.isfinite(a['variance'][:,:,:3]).all(2)
            for field,value in [('valid_count',32),('valid_batches',8),('incomplete_batches',0),('nonfinite',0)]:
                good &= (a[field][:,:,:3]==value).all(2)
            common &= good
            arrays[name] = a
        assert common.sum() == source['scenes'][scene]['common_pixels']
        masks = {'Whole image':common}
        if scene=='5':
            masks.update({name:common & mask for name,mask in polygons.items()})
        counts = block_sum(common)
        tile_values = {name:np.divide(block_sum(np.where(common,a['variance'][:,:,:3].sum(2,dtype=float),0)),
                                     counts, out=np.zeros_like(counts), where=counts>0) for name,a in arrays.items()}
        scene_output = dict(identity=source['scenes'][scene]['identity'], regions={})
        for region,mask in masks.items():
            region_output = dict(pixels=int(mask.sum()), methods={})
            if region!='Whole image':
                gbuffer = np.memmap(ROOT/cases['camera']['path']/'colortex8.bin', '<u4', 'r', shape=SHAPE)[::-1]
                normals = np.unique(gbuffer[mask,0] & 0xffff)
                assert normals.size==1
                region_output['packed_world_normal_oct16'] = int(normals[0])
                region_output['polygon_pixels'] = int(polygons[region].sum())
            for name,a in arrays.items():
                v = a['variance'][mask,:3].astype(float)
                mu = a['mean'][mask,:3]
                rgb = v.sum(1)
                assert np.all(v>=0)
                metrics = dict(rgb=summarize(rgb, np.any(mu!=0,axis=1)), blue=summarize(v[:,2],mu[:,2]),
                               mean_rgb=mu.mean(0,dtype=float).tolist())
                region_output['methods'][NAMES[name]] = metrics
                if region=='Whole image':
                    np.testing.assert_allclose(metrics['rgb']['mean'],source['scenes'][scene]['methods'][name]['variance'],rtol=1e-12)
                    curves[scene,name] = np.quantile(rgb,PERCENTILES/100)
            scene_output['regions'][region] = region_output
        output['scenes']['light-sampling-'+scene] = scene_output
        for name,a in arrays.items():
            delta = tile_values[name]-tile_values['camera']
            limit, soft = source['maps']['delta_max'], source['maps']['delta_soft']
            signed = np.sign(delta)*np.log1p(np.abs(delta)/soft)/np.log1p(limit/soft)
            colors = np.rint(matplotlib.colormaps['RdBu_r'](np.clip(.5+.5*signed,0,1))[:,:,:3]*255).astype('uint8')
            colors = colors.repeat(16,0).repeat(16,1)[:1080,:1920]
            colors[~common] = [70,70,70]
            png = assets/f'ls{scene}-{name}-camera-delta.png'
            avif = png.with_suffix('.avif')
            Image.fromarray(colors).save(png)
            avif.write_bytes(generator.encode_avif(png,encoder))
            np.testing.assert_array_equal(np.asarray(Image.open(avif).convert('RGB')),colors)
            item = dict(values=['light-sampling-'+scene,NAMES[name],view],src=avif.relative_to(OUT).as_posix(),label=str(png))
            data['images'].append(item)
            new_images.append(dict(values=item['values'],png_sha256=sha(png),avif_sha256=sha(avif)))
            ui['scenes']['light-sampling-'+scene]['methods'][NAMES[name]]['camera_delta_saturated_fraction'] = float(counts[np.abs(delta)>limit].sum()/common.sum())

    fig, axes = plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for name in ('camera','sah','prefix','brdf'):
        for ax in axes:
            ax.plot(PERCENTILES,curves['5',name],label=NAMES[name])
            ax.set_yscale('symlog',linthresh=1e-7)
            ax.set_xlabel('Pixel-variance percentile (%)')
            ax.set_ylabel('Absolute RGB variance')
            ax.grid(alpha=.2)
    axes[0].set_xlim(0,99.9)
    axes[1].set_xlim(95,99.9)
    tail_values = np.concatenate([curve[PERCENTILES>=95] for (scene,_),curve in curves.items() if scene=='5'])
    axes[1].set_ylim(float(tail_values.min())*.8,float(tail_values.max())*1.15)
    axes[0].set_title('ls5: full spatial distribution')
    axes[1].set_title('ls5: high-variance tail (95 to 99.9%)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Same complete receiver pixels; 32-frame estimates. Zeros may miss rare contributions.')
    fig.savefig(OUT/'ls5-variance-distribution.png',dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(16,9))
    ax.imshow(Image.open(OUT/'png/ls5-camera-mean.png'))
    for i,(name,vertices) in enumerate(REGIONS.items()):
        color=plt.get_cmap('tab10')(i)
        ax.add_patch(PolygonPatch(vertices,closed=True,fill=False,edgecolor=color,linewidth=2))
        x,y=vertices[0]
        ax.text(x,y-12,name.split(':')[0],color='white',fontsize=14,bbox=dict(facecolor=color,edgecolor='none'))
    ax.set_axis_off()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.savefig(OUT/'ls5-review-regions.png',dpi=120)
    plt.close(fig)

    payload=json.dumps(data,separators=(',',':')).replace('<','\\u003c')
    page_text=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],page_text,flags=re.S)
    page_text=re.sub(r'const absoluteData = .*;\nconst names',lambda m:'const absoluteData = '+json.dumps(ui,separators=(',',':'))+';\nconst names',page_text)
    page_text=page_text.replace('<details><summary>Absolute statistics and coverage</summary>',HEADER+'\n<details><summary>Earlier tile summary (relative to Prefix)</summary>')
    page_text=page_text.replace('Full images and absolute noise</h1>','Full images and noise tails</h1>')
    page_text=page_text.replace('  updateAbsoluteSummary();','  updateAbsoluteSummary();\n  updateDistribution();')
    page_text=page_text.replace("const base=scene.methods['World Prefix + A + Near/Far'].tiles[idx],value=method.tiles[idx];", "const reference=current[2].endsWith('Prefix')?'World Prefix + A + Near/Far':'Camera grouping + A + Near/Far';\n  const base=scene.methods[reference].tiles[idx],value=method.tiles[idx];")
    page_text=page_text.replace('receivers · Prefix ${number(base)}','receivers · ${reference} ${number(base)}')
    page_text=page_text.replace("(100*method.delta_map_saturated_fraction).toFixed(2)","(100*(current[2].endsWith('Camera')?method.camera_delta_saturated_fraction:method.delta_map_saturated_fraction)).toFixed(2)")
    injection='const distributionData = '+json.dumps(output['scenes'],separators=(',',':'))+';\n'+JS
    page_text=page_text.replace('compare.value = "1";\nrender();',injection+'\ncompare.value = "1";\ncurrent[1]="Camera grouping + A + Near/Far";\nrender();')
    page_text=page_text.replace('</style>',STYLE+'\n</style>')
    page=OUT/'distribution-review.html'
    page.write_text(page_text,encoding='utf-8')
    scripts=re.findall(r'<script(?![^>]*type="application/json")[^>]*>(.*?)</script>',page_text,re.S)
    (assets/'syntax-check.js').write_text('\n'.join(scripts),encoding='utf-8')
    assert sha(original_page)==output['original_page_sha256']
    for item in data['images']:
        np.testing.assert_array_equal(np.asarray(Image.open(item['label']).convert('RGB')),
                                      np.asarray(Image.open(OUT/item['src']).convert('RGB')))
    output['validation'].update(lossless_pairs=len(data['images']),new_images=new_images,
        browser='Not checked: local browser was disabled in Work settings.', html_sha256=sha(page))
    (OUT/'distribution-statistics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(page=str(page),images=len(data['images']),regions={s:{r:v['pixels'] for r,v in x['regions'].items()} for s,x in output['scenes'].items()})))


HEADER='''<details open id="distribution-panel"><summary>Absolute pixel-variance distribution: tail first</summary>
<div>Primary quality comparison: SAH versus Camera grouping. Prefix and Cosine BRDF remain controls.</div>
<div><label>Region <select id="noise-region"></select></label> <label>Channel <select id="noise-channel"><option value="rgb">RGB sum</option><option value="blue">Blue</option></select></label>
<a href="ls5-review-regions.png">ls5 region map A-D</a> · <a href="ls5-variance-distribution.png">ls5 distribution plot</a> · <a href="distribution-statistics.json">Definitions and data</a></div>
<div id="distribution-table"></div><div>Spatial std is the spread of per-pixel variances. Worst 5% selects each method's own tail. No brightness filter; zero estimates can miss rare light.</div></details>'''
STYLE='''#distribution-panel{max-height:235px;font-size:12px}#distribution-panel table{border-collapse:collapse;width:100%;margin:5px 0}#distribution-panel th,#distribution-panel td{text-align:right;padding:3px 8px;border-bottom:1px solid #555}#distribution-panel td:first-child,#distribution-panel th:first-child{text-align:left}#distribution-panel select{padding:2px}'''
JS=r'''
function updateDistribution(){
  const regions=distributionData[current[0]].regions,select=document.querySelector('#noise-region');
  const old=select.value, keys=Object.keys(regions);
  if([...select.options].map(o=>o.value).join('|')!==keys.join('|')){
    select.replaceChildren(...keys.map(key=>new Option(key,key)));
    if(keys.includes(old))select.value=old;
  }
  const region=regions[select.value],channel=document.querySelector('#noise-channel').value;
  let table='<table><tr><th>Method</th><th>Worst 5% mean</th><th>P99</th><th>P95</th><th>Median</th><th>Mean</th><th>Spatial std</th><th>P5</th><th>Zero mean</th></tr>';
  for(const name of ['Camera grouping + A + Near/Far','Global area Light-SAH + A + Near/Far','World Prefix + A + Near/Far','Cosine BRDF-only']){
    if(!region.methods[name])continue;
    const m=region.methods[name][channel];
    table+=`<tr><td>${name}</td>`+['worst05_mean','p99','p95','p50','mean','spatial_std','p05'].map(k=>`<td>${number(m[k])}</td>`).join('')+`<td>${(100*m.zero_mean_fraction).toFixed(2)}%</td></tr>`;
  }
  document.querySelector('#distribution-table').innerHTML=table+`</table>${region.pixels.toLocaleString()} common complete receiver pixels. 32-frame estimates; not convergence or ground-truth error.`;
}
document.querySelector('#noise-region').addEventListener('change',updateDistribution);
document.querySelector('#noise-channel').addEventListener('change',updateDistribution);
'''


if __name__=='__main__':
    main()

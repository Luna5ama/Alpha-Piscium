"""Prepare the requested temporary mixed-batch comparison using existing PNGs only."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'builds/light-sampling/20261031-nearfar-iteration'
PREVIOUS = ROOT / 'builds/light-sampling/20261031-weights-presets1080'
DEFAULT_OUTPUT = ROOT / 'builds/light-sampling/20261031-temporary-baselines'
METHODS = (
    ('a', 'Half-space A', 'halfspace-a'),
    ('nearfar', 'Near/Far Mix', 'near-far'),
    ('candidate6', 'Half-space A + Near/Far Mix', None),
    ('brdf', 'Cosine BRDF', 'brdf'),
    ('original', 'Original', 'original'),
)
SAMPLES = (('1 spp', 'single'), ('32 frame mean', '32mean'))
NOTICE = '''<p style="max-width:1100px;padding:10px 14px;background:#332b16;color:#ffe3a0;border:1px solid #736039;border-radius:6px">
Temporary visual comparison using existing captures from different batches; no new rendering or benchmarking.
Cosine BRDF and Original reuse the previous weights run. All available ls3 images also come from that run;
Half-space A + Near/Far Mix was never captured for ls3 and is intentionally unavailable.
Input geometry and light sets can differ between batches (especially ls4); this page does not establish a controlled ranking.
<a href="provenance.json" style="color:inherit">Per-image sources and capture revisions</a>.
</p>'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    current_manifest = read(CURRENT / 'image-manifest-final.json')
    previous_manifest = read(PREVIOUS / 'comparison-manifest.json')
    accepted = {(row['scene'], row['method']): ROOT / row['path']
                for row in read(ROOT / 'experiments/light-sampling/20261031-nearfar-iteration.json')['accepted_quality']}
    current = {(row['dimensions']['Scene'], row['dimensions']['Method'], row['dimensions']['Sample']): row
               for row in current_manifest['images']}
    previous = {(row['dimensions']['Scene'], row['dimensions']['Sampler'], row['dimensions']['Accumulation']): row
                for row in previous_manifest['images']}
    case_sources = {}
    images, sources, missing = [], [], []
    for scene in range(1, 6):
        for method, label, old_method in METHODS:
            for sample, old_sample in SAMPLES:
                dims = {'Scene': f'ls{scene}', 'Method': label, 'Sample': sample}
                if method in ('a', 'nearfar', 'candidate6') and scene != 3:
                    row = current[f'ls{scene}', method, sample]
                    folder = accepted[scene, method]
                    batch = 'targeted-nearfar'
                    mean_metadata = folder / 'quality.mean.json'
                    tree = folder / 'iris_ssbo_12.bin'
                elif old_method is not None:
                    row = previous[f'light-sampling-{scene}', old_method, old_sample]
                    folder = Path(row['path']).parent
                    batch = 'previous-weights1080'
                    mean_metadata = folder / 'mean32.mean.json'
                    tree = folder / 'tree.bin'
                else:
                    missing.append(dict(dimensions=dims, reason='No existing ls3 half-space + near/far capture; user requested reuse without new sampling.'))
                    continue
                path = Path(row['path'])
                assert path.exists() and sha256(path) == row['sha256'], path
                with Image.open(path) as image:
                    assert image.size == (1920, 1080), path
                key = str(folder)
                if key not in case_sources:
                    native = read(folder / 'job-result.json')['result']
                    p = native['provenance']
                    assert p['preset_id'] == f'light-sampling-{scene}' and not p['stale'] and not p['head_changed']
                    settings = {s['name']: s['value'] for s in p['effective_settings']['settings']}
                    expected_sampling = '1' if method == 'brdf' else '2'
                    expected_model = {'a': '1', 'nearfar': '3', 'candidate6': '6', 'brdf': '0', 'original': '0'}[method]
                    assert settings['SETTING_DEBUG_LIGHT_SAMPLING'] == expected_sampling
                    assert settings['SETTING_DEBUG_LIGHT_WEIGHT_MODEL'] == expected_model
                    meta = read(mean_metadata)
                    acc = meta['accumulation']
                    assert meta['width'] == 1920 and meta['height'] == 1080 and acc['frames'] == 32
                    assert acc['end_frame'] - acc['start_frame'] == 31
                    with tree.open('rb') as stream:
                        header = struct.unpack('<8I', stream.read(32))
                    case_sources[key] = dict(batch=batch, capture_directory=str(folder.relative_to(ROOT)).replace('\\', '/'),
                        revision=p['resolved_revision'], source_snapshot_sha256=p['source_snapshot_sha256'],
                        preset_id=p['preset_id'], preset_sha256=p['preset_sha256'], scene_sha256=p['scene_sha256'],
                        sampling=expected_sampling, weight_model=expected_model, emitters_in_native_header=header[1],
                        frames=32, start_frame=acc['start_frame'], end_frame=acc['end_frame'],
                        native_result_sha256=sha256(folder / 'job-result.json'))
                images.append(dict(path=str(path.resolve()), dimensions=dims, sha256=row['sha256']))
                sources.append(dict(dimensions=dims, source_png=str(path.resolve()), sha256=row['sha256'], **case_sources[key]))
    assert len(images) == 48 and len(missing) == 2
    assert len({tuple(row['dimensions'].values()) for row in images}) == 48
    manifest = dict(title='Light Sampling - Temporary Mixed-Batch Comparison', images=images)
    provenance = dict(schema=1, scope='Temporary visual juxtaposition, not a new controlled experiment or numerical ranking.',
        new_sampling_jobs=0, new_benchmark_jobs=0, scenes=5, methods=[m[1] for m in METHODS],
        images=len(images), missing=missing, display='Existing +4EV/Reinhard PNG previews are reused without changes.',
        notes='Original and Cosine BRDF come from the previous weights run; ls3 also reuses its previous Half-space A and Near/Far Mix. Different input cohorts are retained honestly.',
        current_manifest_sha256=sha256(CURRENT / 'image-manifest-final.json'),
        previous_manifest_sha256=sha256(PREVIOUS / 'comparison-manifest.json'), sources=sources)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    (output / 'notice.html').write_text(NOTICE+'\n', encoding='utf-8')
    print(json.dumps(dict(images=len(images), missing=len(missing), new_sampling_jobs=0, output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.output_dir)

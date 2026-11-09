"""Freeze upper-only flux-diagonal atlas against the validated frozen GPU loader."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from light_sampling_clustering_frozen import ATLAS_WIDTH, PATH, pack
from light_sampling_clustering_hierarchy import brick_roots
from light_sampling_clustering_repair import EVIDENCE, build, build_primitives, control_root
from light_sampling_clustering_rule_screen import RULES
from light_sampling_moving_analysis import leaves, native_node, ordered_signature
from light_sampling_weight_reference import load_tree


def main(output, moving_tree=None, lookahead=False, global_rule=None):
    output.mkdir(parents=True, exist_ok=True)
    reference = EVIDENCE/'20261108-clustering-repair-reference.npz'
    parts = [np.zeros((ATLAS_WIDTH,4), dtype='<u4')]
    index = []
    with np.load(reference) as source:
        roots = {slot:control_root(source,f'{slot}/prefix/') for slot in (1,5)}
    if moving_tree:
        header, nodes, links = load_tree(moving_tree,1)
        assert header[2]==header[3]==0
        roots[6] = native_node(nodes,links,1)
        assert len(leaves(roots[6]))==int(header[1])
    for slot,root in roots.items():
        rows = sorted(leaves(root))
        if global_rule:
            records = np.asarray(rows)
            rule = RULES['flux-diagonal'] if global_rule=='flux-diagonal' else None
            result = build(records[:,:4].astype('<i4'),records[:,4],True,partition=rule)
        else:
            atoms = brick_roots(root)
            result = build_primitives(np.asarray(list(atoms)),list(atoms.values()),True,
                                      partition=RULES['flux-diagonal'])
            restored = brick_roots(result)
            assert restored.keys()==atoms.keys()
            assert all(restored[key] is node for key,node in atoms.items())
        assert sorted(leaves(result))==rows
        native = pack(result)
        parts[0][slot-1] = [sum(len(p) for p in parts),len(native),len(rows),0]
        parts.append(native.reshape(-1,4))
        index.append(dict(slot=slot, nodes=len(native)-1, emitters=len(rows),
                          leaf_sha256=hashlib.sha256(np.asarray(rows,'<f8').tobytes()).hexdigest(),
                          ordered_sha256=ordered_signature(result)))
    parts.append(np.zeros(((-sum(len(p) for p in parts))%ATLAS_WIDTH,4), dtype='<u4'))
    atlas = np.concatenate(parts)
    files = ['scripts/shaders.properties','shaders/shaders.properties']
    if lookahead:
        files.append('shaders/pass/prepare/LightSamplingBuild.comp.glsl')
    for file in files:
        properties = subprocess.check_output(['git','show',f'52025a877:{file}']).decode('utf8')
        updated = properties
        if file.endswith('.properties'):
            updated, count = re.subn(r'(usam_lsFrozenTree=.+ RGBA32UI 2048 )\d+',
                                     lambda m: m[1]+str(len(atlas)//ATLAS_WIDTH), updated)
            assert count == 1
        if lookahead:
            flag = 'SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 6'
            assert updated.count(flag)==(1 if file.endswith('.properties') else 2)
            updated = updated.replace(flag,'('+flag+' || SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 11)')
        path = output/file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(updated.encode('utf8'))
    texture = output/'shaders'/PATH
    texture.parent.mkdir(parents=True, exist_ok=True)
    texture.write_bytes(atlas.tobytes())
    (output/'atlas.json').write_text(json.dumps(dict(trees=index, width=ATLAS_WIDTH,
        height=len(atlas)//ATLAS_WIDTH, sha256=hashlib.sha256(atlas.tobytes()).hexdigest()),indent=2)+'\n',
        encoding='utf8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--moving-tree',type=Path)
    parser.add_argument('--lookahead',action='store_true')
    parser.add_argument('--global-rule',choices=('area','flux-diagonal'))
    options = parser.parse_args()
    main(options.output,options.moving_tree,options.lookahead,options.global_rule)

"""Bake a frozen 32-byte-node Light-SAH tree atlas from accepted native inputs."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import ROOT, build, stats
from light_sampling_moving_analysis import leaves, native_node, ordered_signature
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree

ATLAS_WIDTH = 2048
PATH = 'textures/light_sampling_frozen_sah.bin'


def pack(root):
    tree = Tree(root)
    nodes = np.zeros((len(tree.nodes)+1, 8), dtype='<u4')
    nodes[1:,:7] = np.asarray(tree.nodes, '<f4').view('<u4')
    for i, children in enumerate(tree.children):
        if len(children):
            assert 2 <= len(children) <= 4 and np.all(np.diff(children)==1)
            link = (int(children[0])+1) | ((len(children)-1)<<28)
        else:
            link = 0x80000000 | int(tree.materials[i])
        nodes[i+1,7] = link
    return nodes


def bake(inputs):
    texels = [np.zeros((ATLAS_WIDTH,4), dtype='<u4')]
    descriptors = []
    next_texel = ATLAS_WIDTH
    for slot,path in inputs:
        header, nodes, links = load_tree(path, 1)
        records = sorted(leaves(native_node(nodes,links,1)))
        assert len(records) == int(header[1]) and header[2] == 0
        assert 1 <= slot <= 6
        assert not np.any(texels[0][slot-1])
        keys = np.asarray([row[:4] for row in records],dtype='<i4')
        flux = np.asarray([row[4] for row in records],dtype='<f4')
        start = time.perf_counter()
        root = build(keys,flux,True)
        packed = pack(root)
        elapsed = time.perf_counter()-start
        np.testing.assert_array_equal(np.asarray(sorted(leaves(root)),dtype='<f8'), records)
        assert len(packed) <= len(nodes)
        texels[0][slot-1] = [next_texel, len(packed), len(records), 0]
        texels.append(packed.reshape(-1,4))
        next_texel += packed.size//4
        descriptors.append(dict(scene_slot=slot, input=path.relative_to(ROOT).as_posix(),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            leaf_sha256=hashlib.sha256(np.asarray(records,'<f8').tobytes()).hexdigest(),
            native_tree_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
            semantic_topology_sha256=ordered_signature(root), stats=stats(Tree(root)), cpu_seconds=elapsed))
    texels.append(np.zeros(((-next_texel)%ATLAS_WIDTH,4),dtype='<u4'))
    return np.concatenate(texels),descriptors


def install(source, data):
    texture=source/'shaders'/PATH
    texture.write_bytes(data.tobytes())
    props=source/'scripts/shaders.properties'
    text=props.read_text(encoding='utf8')
    binding=f'customTexture.usam_lsFrozenTree={PATH} TEXTURE_2D RGBA32UI {ATLAS_WIDTH} {len(data)//ATLAS_WIDTH} RGBA_INTEGER UNSIGNED_INT'
    anchor='#if SETTING_DEBUG_LIGHT_SAMPLING == 3\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,'#if SETTING_DEBUG_LIGHT_SAMPLING == 2 && SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 6\n'+binding+'\n#endif\n'+anchor)
    props.write_text(text,encoding='utf8',newline='\n')
    program=source/'shaders/pass/prepare/LightSamplingBuild.comp.glsl'
    text=program.read_text(encoding='utf8')
    start='const ivec3 workGroups = ivec3(1, 1, 1);\nshared uvec3 ls_topKeys[16];\nshared uint ls_topNodes[16];'
    end='    #else\n    if (gl_GlobalInvocationID.x != 0u) return;\n    uvec3 low, dimensions;'
    assert text.count(start)==text.count(end)==1
    text=text.replace(start, '''const ivec3 workGroups = ivec3(256, 1, 1);
uniform usampler2D usam_lsFrozenTree;

uvec4 ls_frozenTexel(uint address) {
    uint width = uint(textureSize(usam_lsFrozenTree, 0).x);
    return texelFetch(usam_lsFrozenTree, ivec2(address % width, address / width), 0);
}''')
    i=text.index(end)
    text=text[:i]+'''    #else
    uint index = gl_GlobalInvocationID.x;
    uvec4 header = ls_frozenTexel(uint(SETTING_DEBUG_LIGHT_CLUSTER_SCENE - 1));
    if (header.z != ls_emitterCount || header.y < 2u || header.y > uint(ls_nodes.length())) {
        if (index == 0u) atomicAdd(ls_overflowCount, 1u);
        return;
    }
    for (; index < header.y; index += 65536u) {
        uint address = header.x + 2u * index;
        LightSamplingNode node = ls_emptyNode();
        node.boundsMinMass = uintBitsToFloat(ls_frozenTexel(address));
        node.boundsMaxLink = uintBitsToFloat(ls_frozenTexel(address + 1u));
        uint link = floatBitsToUint(node.boundsMaxLink.w);
        if ((link & LS_LEAF_BIT) != 0u) {
            uint material = voxel_getMaterialID(ivec3(node.boundsMinMass.xyz));
            if (material != (link & ~LS_LEAF_BIT)) {
                atomicAdd(ls_overflowCount, 1u);
            } else {
                HardcodedPBR pbr = hardcodedpbr_decode(material);
                float mass = ldexp(pow(pbr.emissive, SETTING_EMISSIVE_PBR_VALUE_CURVE), pbr.emissiveMultiplier);
                if (abs(node.boundsMinMass.w - mass) > 0.000002 * mass) atomicAdd(ls_overflowCount, 1u);
            }
        }
        ls_nodes[index] = node;
    }
    #endif
}
'''
    # This is a frozen model-6 experiment pack, not a production option.
    text=text.replace('#else\nconst ivec3 workGroups = ivec3(256, 1, 1);',
                      '#elif SETTING_DEBUG_LIGHT_SAMPLING == 2 && SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 6\nconst ivec3 workGroups = ivec3(256, 1, 1);')
    text=text.replace('    #else\n    uint index = gl_GlobalInvocationID.x;', '    #elif SETTING_DEBUG_LIGHT_SAMPLING == 2 && SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 6\n    uint index = gl_GlobalInvocationID.x;')
    program.write_text(text,encoding='utf8',newline='\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('inputs',nargs='+',help='scene=path/to/tree.bin')
    args=parser.parse_args()
    inputs=[(int(slot), Path(path).resolve()) for slot,path in (x.split('=',1) for x in args.inputs)]
    atlas,description=bake(inputs)
    install(args.source,atlas)
    manifest=dict(width=ATLAS_WIDTH,height=len(atlas)//ATLAS_WIDTH,bytes=atlas.nbytes,
        sha256=hashlib.sha256(atlas.tobytes()).hexdigest(),trees=description)
    args.manifest.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')

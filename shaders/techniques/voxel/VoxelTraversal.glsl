#ifndef INCLUDE_techniques_voxel_VoxelTraversal_glsl
#define INCLUDE_techniques_voxel_VoxelTraversal_glsl a

#include "/techniques/voxel/Voxelization.glsl"

layout(std430, binding = 8) restrict readonly buffer VoxelTreeData {
    uint voxel_treeScalar[];
};

#if VOXEL_GRID_SIZE == 16
uint _voxel_spreadBits(uint x) {
    x = (x * 257u) & 0x00F00Fu;
    x = (x * 17u) & 0x0C30C3u;
    x = (x * 5u) & 0x249249u;
    return x;
}
#else
uint _voxel_spreadBits(uint x) {
    x &= 0x000003FFu;
    x = (x * 0x00010001u) & 0xFF0000FFu;
    x = (x * 0x00000101u) & 0x0F00F00Fu;
    x = (x * 0x00000011u) & 0xC30C30C3u;
    x = (x * 0x00000005u) & 0x49249249u;
    return x;
}
#endif

shared uint _voxel_levelOffsets[6];
shared ivec2 _voxel_levelSizeMask[6];
shared uint _voxel_spreadLUT[VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE];
shared vec3 _voxel_gridOriginF;

uint _voxel_packBlockPos(ivec3 blockPos) {
    return _voxel_spreadLUT[uint(blockPos.x)] +
        (_voxel_spreadLUT[uint(blockPos.y)] << 1u) +
        (_voxel_spreadLUT[uint(blockPos.z)] << 2u);
}

void voxel_initShared() {
    if (gl_LocalInvocationIndex == 0u) {
        ivec3 cameraBrick = cameraPositionInt >> 4;
        _voxel_gridOriginF = vec3((cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4);
        _voxel_levelOffsets[0] = 0u << 1u;
        _voxel_levelOffsets[1] = uint(VOXEL_TREE_OFFSET_L1) << 1u;
        _voxel_levelOffsets[2] = uint(VOXEL_TREE_OFFSET_L2) << 1u;
        _voxel_levelOffsets[3] = uint(VOXEL_TREE_OFFSET_L3) << 1u;
        _voxel_levelOffsets[4] = uint(VOXEL_TREE_OFFSET_L4) << 1u;
        #if VOXEL_TREE_TOP_LEVEL == 5
        _voxel_levelOffsets[5] = uint(VOXEL_TREE_OFFSET_L5) << 1u;
        #else
        _voxel_levelOffsets[5] = 0u << 1u;
        #endif
    }

    uint localSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    for (uint i = gl_LocalInvocationIndex; i < 256u; i += localSize) {
        uint spread = _voxel_spreadBits(i);
        _voxel_spreadLUT[i] = spread;
        #if VOXEL_GRID_SIZE >= 32
        _voxel_spreadLUT[i + 256u] = spread | 0x01000000u;
        #endif
        #if VOXEL_GRID_SIZE >= 64
        _voxel_spreadLUT[i + 512u] = spread | 0x08000000u;
        _voxel_spreadLUT[i + 768u] = spread | 0x09000000u;
        #endif
    }

    if (gl_LocalInvocationIndex == 0u) {
        _voxel_levelSizeMask[0] = ivec2(0);
    } else if (gl_LocalInvocationIndex < 6u) {
        int cellShift = (int(gl_LocalInvocationIndex) - 1) << 1;
        int sizeMask = -(1 << cellShift);
        // Store absolute cell size (1 << cellShift) in the Y component
        _voxel_levelSizeMask[gl_LocalInvocationIndex] = ivec2(sizeMask, 1 << cellShift);
    }

    barrier();
}

#endif

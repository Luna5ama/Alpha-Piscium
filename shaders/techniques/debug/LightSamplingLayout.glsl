#ifndef INCLUDE_debug_LightSamplingLayout_glsl
#define INCLUDE_debug_LightSamplingLayout_glsl

#if VOXEL_GRID_SIZE == 64
#define LS_GRID_BITS 6
#if LS_TREE_ARITY == 2
#define LS_OUTER_NODE_COUNT 558854u
#elif LS_TREE_ARITY == 4
#define LS_OUTER_NODE_COUNT 380432u
#else
#define LS_OUTER_NODE_COUNT 333944u
#endif
#elif VOXEL_GRID_SIZE == 32
#define LS_GRID_BITS 5
#if LS_TREE_ARITY == 2
#define LS_OUTER_NODE_COUNT 74464u
#elif LS_TREE_ARITY == 4
#define LS_OUTER_NODE_COUNT 51732u
#else
#define LS_OUTER_NODE_COUNT 46448u
#endif
#else
#define LS_GRID_BITS 4
#if LS_TREE_ARITY == 2
#define LS_OUTER_NODE_COUNT 10570u
#elif LS_TREE_ARITY == 4
#define LS_OUTER_NODE_COUNT 7620u
#else
#define LS_OUTER_NODE_COUNT 7144u
#endif
#endif
#define LS_WORLD_LEVELS ((3 * LS_GRID_BITS + LS_TREE_BITS - 1) / LS_TREE_BITS)

uvec3 ls_worldShift(uint level) {
    uint bits = level * uint(LS_TREE_BITS);
    return uvec3((bits + 2u) / 3u, (bits + 1u) / 3u, bits / 3u);
}

uvec3 ls_unsignedBrick(ivec3 worldBrick) {
    return uvec3(worldBrick + ivec3(134217728));
}

void ls_worldRange(uint level, out uvec3 low, out uvec3 dimensions) {
    uvec3 first = ls_unsignedBrick((cameraPositionInt >> 4) - ivec3(VOXEL_GRID_SIZE / 2));
    uvec3 shift = ls_worldShift(level);
    low = first >> shift;
    dimensions = ((first + uint(VOXEL_GRID_SIZE - 1)) >> shift) - low + 1u;
}

uint ls_levelCapacity(uint level) {
    uvec3 stride = uvec3(1u) << ls_worldShift(level);
    uvec3 dimensions = (uvec3(VOXEL_GRID_SIZE - 2) + stride) / stride + 1u;
    return dimensions.x * dimensions.y * dimensions.z;
}

uint ls_levelOffset(uint level) {
    uint offset = 32u;
    for (uint i = 1u; i < level; ++i) offset += uint(LS_TREE_ARITY) * ls_levelCapacity(i);
    return offset;
}

uint ls_worldIndex(uvec3 cell, uvec3 low, uvec3 dimensions) {
    uvec3 relative = cell - low;
    return relative.x + dimensions.x * (relative.y + dimensions.y * relative.z);
}

uvec3 ls_worldCell(uint index, uvec3 low, uvec3 dimensions) {
    return low + uvec3(index % dimensions.x, (index / dimensions.x) % dimensions.y,
                      index / (dimensions.x * dimensions.y));
}

uint ls_worldOrdinal(uvec3 brick, uint parentLevel) {
    uint ordinal = 0u;
    for (uint i = 0u; i < uint(LS_TREE_BITS); ++i) {
        uint bit = (parentLevel - 1u) * uint(LS_TREE_BITS) + i;
        uint axis = bit % 3u;
        uint coordinate = axis == 0u ? brick.x : (axis == 1u ? brick.y : brick.z);
        ordinal |= ((coordinate >> (bit / 3u)) & 1u) << i;
    }
    return ordinal;
}

uint ls_worldAddress(uvec3 brick, uint level) {
    uvec3 low, dimensions;
    if (level == uint(LS_WORLD_LEVELS)) {
        ls_worldRange(level, low, dimensions);
        return 2u + ls_worldIndex(brick >> ls_worldShift(level), low, dimensions);
    }
    uint parent = level + 1u;
    ls_worldRange(parent, low, dimensions);
    return ls_levelOffset(parent) + uint(LS_TREE_ARITY)
        * ls_worldIndex(brick >> ls_worldShift(parent), low, dimensions) + ls_worldOrdinal(brick, parent);
}

#endif

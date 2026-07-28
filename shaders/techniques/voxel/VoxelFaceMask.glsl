// Publishes per-voxel open-face masks after shadow voxelization.
//
// Two checkerboard parity dispatches keep every SSBO neighbor read disjoint
// from the bricks being written by that dispatch. Within a brick, all material
// IDs are snapshotted to shared memory before any writeback.

#define VOXEL_BRICK_DATA_MODIFIER restrict readonly buffer
#define VOXEL_MATERIAL_VEC4 a
#define VOXEL_MATERIAL_DATA_MODIFIER restrict buffer
#define RC_DATA_MODIFIER restrict buffer
#include "/techniques/voxel/Voxelization.glsl"
#include "/techniques/gi/RadianceCache.glsl"

#ifndef VOXEL_FACE_MASK_PARITY
#define VOXEL_FACE_MASK_PARITY 0
#endif

layout(local_size_x = 64) in;
#if VOXEL_GRID_SIZE == 64
const ivec3 workGroups = ivec3(131072, 1, 1);
#elif VOXEL_GRID_SIZE == 32
const ivec3 workGroups = ivec3(16384, 1, 1);
#else
const ivec3 workGroups = ivec3(2048, 1, 1);
#endif

shared uvec4 brickMaterials[1024];

uint voxel_materialLane(uvec4 materials, uint lane) {
    if (lane == 0u) return materials.x;
    if (lane == 1u) return materials.y;
    if (lane == 2u) return materials.z;
    return materials.w;
}

uint voxel_loadMaterialID(uint allocID, uint blockMorton) {
    uvec4 materials = voxel_materials_v4[allocID * 1024u + (blockMorton >> 2u)];
    return voxel_materialLane(materials, blockMorton & 3u) & 0xFFFFu;
}

bool voxel_isTrackedMaterial(uint materialID) {
    return materialID != 0u && materialID != MATERIAL_ID_WATER;
}

bool voxel_isFullCubeMaterial(uint materialID) {
    if (!voxel_isTrackedMaterial(materialID)) {
        return false;
    }
    return ((texelFetch(usam_pbrLUT0, int(materialID), 0).x >> 28u) & 1u) != 0u;
}

uvec4 voxel_markFullCubes(uvec4 materials) {
    materials &= uvec4(0xFFFFu);
    if (voxel_isFullCubeMaterial(materials.x)) materials.x |= 0x80000000u;
    if (voxel_isFullCubeMaterial(materials.y)) materials.y |= 0x80000000u;
    if (voxel_isFullCubeMaterial(materials.z)) materials.z |= 0x80000000u;
    if (voxel_isFullCubeMaterial(materials.w)) materials.w |= 0x80000000u;
    return materials;
}

bool voxel_loadBrickFullCube(uint blockMorton) {
    uint material = voxel_materialLane(brickMaterials[blockMorton >> 2u], blockMorton & 3u);
    return (material & 0x80000000u) != 0u;
}

bool voxel_fullCubeAtGridBlock(ivec3 gridBlockPos, ivec3 ownerBrickCoord) {
    int gridExtent = VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE;
    if (any(lessThan(gridBlockPos, ivec3(0))) || any(greaterThanEqual(gridBlockPos, ivec3(gridExtent)))) {
        return false;
    }

    ivec3 brickCoord = gridBlockPos >> 4;
    uint blockMorton = voxel_blockMorton(gridBlockPos & 15);
    if (all(equal(brickCoord, ownerBrickCoord))) {
        return voxel_loadBrickFullCube(blockMorton);
    }
    uint allocID = voxel_brickAllocID[voxel_brickMorton(brickCoord)];
    if (allocID == VOXEL_UNALLOCATED) {
        return false;
    }
    return voxel_isFullCubeMaterial(voxel_loadMaterialID(allocID, blockMorton));
}

void rc_markPendingVisibleFace(ivec3 worldBlockPos, uint faceBits) {
    vec3 ownerBlockCenter = vec3(worldBlockPos) + vec3(0.5);
    for (uint level = 0u; level < RC_CLIP_LEVELS; level++) {
        ivec3 worldCellCoord = rc_worldCellCoord(ownerBlockCenter, level);
        if (rc_worldCellInCurrentClip(level, worldCellCoord)) {
            uint entryIndex = rc_entryIndex(level, worldCellCoord);
            uint bufferIndex = rc_bufferEntryIndex(rc_currentSide(), entryIndex);
            atomicOr(rc_indirection[bufferIndex].w, rc_entryMetaPendingFaceBits(faceBits));
        }
    }
}

void main() {
    uint halfGridSize = uint(VOXEL_GRID_SIZE / 2);
    uint groupIndex = gl_WorkGroupID.x;
    uint xPair = groupIndex % halfGridSize;
    uint yz = groupIndex / halfGridSize;
    uint y = yz % uint(VOXEL_GRID_SIZE);
    uint z = yz / uint(VOXEL_GRID_SIZE);
    uint x = xPair * 2u + ((uint(VOXEL_FACE_MASK_PARITY) ^ y ^ z) & 1u);
    ivec3 brickCoord = ivec3(x, y, z);
    uint brickMorton = voxel_brickMorton(brickCoord);
    uint allocID = voxel_brickAllocID[brickMorton];
    if (allocID == VOXEL_UNALLOCATED) {
        return;
    }

    uint localID = gl_LocalInvocationID.x;
    uint sharedBase = localID * 16u;
    uint materialBase = allocID * 1024u;
    for (uint i = 0u; i < 16u; i++) {
        uint index = sharedBase + i;
        brickMaterials[index] = voxel_markFullCubes(voxel_materials_v4[materialBase + index]);
    }
    barrier();

    ivec3 cameraBrick = cameraPositionInt >> 4;
    ivec3 gridOrigin = (cameraBrick - ivec3(VOXEL_GRID_SIZE / 2)) << 4;
    ivec3 brickBlockBase = brickCoord << 4;
    for (uint i = 0u; i < 16u; i++) {
        uint index = sharedBase + i;
        uvec4 materials = brickMaterials[index] & uvec4(0xFFFFu);
        for (uint lane = 0u; lane < 4u; lane++) {
            uint materialID = materials[lane];
            uint faceBits = 0u;
            if (voxel_isTrackedMaterial(materialID)) {
                uint blockMorton = index * 4u + lane;
                ivec3 blockInBrick = ivec3(morton3D_12bDecode(blockMorton));
                ivec3 gridBlockPos = brickBlockBase + blockInBrick;
                for (uint faceId = 0u; faceId < 6u; faceId++) {
                    ivec3 neighborGridBlock = gridBlockPos + rc_faceNormalI(faceId);
                    if (!voxel_fullCubeAtGridBlock(neighborGridBlock, brickCoord)) {
                        faceBits |= rc_faceBit(faceId);
                    }
                }
                if (faceBits != 0u) {
                    rc_markPendingVisibleFace(gridOrigin + gridBlockPos, faceBits);
                }
            }
            materials[lane] = materialID | (faceBits << 16u);
        }
        voxel_materials_v4[materialBase + index] = materials;
    }
}

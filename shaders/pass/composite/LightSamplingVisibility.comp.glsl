#if LSV_STAGE == 2
layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
layout(local_size_x = 256) in;
#if LSV_STAGE == 0 || LSV_STAGE == 1
const ivec3 workGroups = ivec3(256, 1, 1);
#elif LSV_STAGE == 3 || LSV_STAGE == 4
const ivec3 workGroups = ivec3(4096, 1, 1);
#else
const ivec3 workGroups = ivec3(1, 1, 1);
#endif
#endif

#include "/techniques/debug/LightSampling.glsl"
#define VOXEL_TRACE_TRUST_MATERIAL_ID a
#define VOXEL_EMISSIVE_CUBES a
#include "/techniques/voxel/VoxelTrace.glsl"

#if LSV_STAGE == 3
shared uint lsv_count;
#endif

void main() {
    #if LSV_STAGE == 0
    for (uint i = gl_GlobalInvocationID.x; i < 37892096u; i += 65536u) lsv_data[i] = 0u;
    if (gl_GlobalInvocationID.x == 0u) {
        lsv_frame = 0u;
        lsv_ready = 0u;
        lsv_probeCount = 0u;
        lsv_exhausted = 0u;
        lsv_activeBricks = 0u;
        lsv_originCount = 0u;
        lsv_hashXor = 0u;
        lsv_hashSum = 0u;
        lsv_originHashXor = 0u;
        lsv_originHashSum = 0u;
        lsv_trainingRuns = 0u;
        lsv_cameraBrickX = 0u;
        lsv_cameraBrickY = 0u;
        lsv_cameraBrickZ = 0u;
        lsv_reserved3 = 0u;
        lsv_reserved4 = 0u;
    }
    #elif LSV_STAGE == 1
    if (gl_GlobalInvocationID.x < 256u) lsv_audit[gl_GlobalInvocationID.x] = vec4(0.0);
    if (lsv_frame != 31u) return;
    uint i = gl_GlobalInvocationID.x;
    if (i < LSV_BRICKS * LSV_ORIGINS) lsv_data[i] = LSV_INVALID;
    #elif LSV_STAGE == 2
    if (lsv_frame != 31u) return;
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, uval_mainImageSizeI))) return;
    vec3 origin, normal, albedo;
    if (!ls_loadReceiver(pixel, origin, normal, albedo)) return;
    uint brick = lsv_brick(origin);
    if (brick == LSV_INVALID) return;
    uint pixelIndex = uint(pixel.y * uval_mainImageSizeI.x + pixel.x);
    uint hashed = hash_pcg_11(pixelIndex);
    uint slot = hashed & 15u;
    // The experiment is 1280x720: 20 bits retain the complete pixel index.
    uint key = (((hashed >> 4u) & 4095u) << 20u) | pixelIndex;
    atomicMin(lsv_data[brick * LSV_ORIGINS + slot], key);
    #elif LSV_STAGE == 3
    voxel_initShared();
    if (lsv_frame != 31u) return;
    uint brick = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    if (lane == 0u) {
        lsv_count = 0u;
        for (uint i = 0u; i < LSV_ORIGINS; ++i) {
            uint key = lsv_data[brick * LSV_ORIGINS + i];
            if (key == LSV_INVALID) continue;
            ++lsv_count;
            uint pixelIndex = key & 0xfffffu;
            ivec2 pixel = ivec2(pixelIndex % uint(uval_mainImageSizeI.x), pixelIndex / uint(uval_mainImageSizeI.x));
            vec3 origin, normal, albedo;
            ls_loadReceiver(pixel, origin, normal, albedo);
            uvec3 bits = floatBitsToUint(origin);
            uint value = hash_pcg_11(key ^ hash_pcg_11(brick * LSV_ORIGINS + i));
            value ^= hash_pcg_11(bits.x) ^ hash_pcg_11(bits.y + 1u) ^ hash_pcg_11(bits.z + 2u);
            atomicXor(lsv_originHashXor, value);
            atomicAdd(lsv_originHashSum, hash_pcg_11(value));
        }
        lsv_data[LSV_COUNTS_OFFSET + brick] = lsv_count;
        if (lsv_count > 0u) {
            atomicAdd(lsv_activeBricks, 1u);
            atomicAdd(lsv_originCount, lsv_count);
        }
    }
    barrier();
    if (lsv_count == 0u) return;
    for (uint bin = lane; bin < LSV_DIRECTIONS; bin += 256u) {
        vec3 direction = lsv_direction(bin);
        vec2 moments = vec2(0.0);
        for (uint i = 0u; i < LSV_ORIGINS; ++i) {
            uint key = lsv_data[brick * LSV_ORIGINS + i];
            float transformed = -1.0;
            if (key != LSV_INVALID) {
                uint pixelIndex = key & 0xfffffu;
                ivec2 pixel = ivec2(pixelIndex % uint(uval_mainImageSizeI.x), pixelIndex / uint(uval_mainImageSizeI.x));
                vec3 origin, normal, albedo;
                ls_loadReceiver(pixel, origin, normal, albedo);
                VoxelRay ray = voxelray_setup(origin, direction, 0u);
                VoxelHit hit = voxel_traceRay(ray, 2048, true);
                if (voxel_traceExhausted(hit, ray)) atomicAdd(lsv_exhausted, 1u);
                float distance = hit.hit ? length(hit.hitPos - origin) : 0.0;
                transformed = hit.hit ? distance / (distance + 16.0) : 1.0;
                moments += vec2(transformed, transformed * transformed);
            }
            uint index = (brick * LSV_DIRECTIONS + bin) * LSV_ORIGINS + i;
            uint word = floatBitsToUint(transformed);
            lsv_data[LSV_DISTANCES_OFFSET + index] = word;
            uint fingerprint = hash_pcg_11(word ^ hash_pcg_11(index));
            atomicXor(lsv_hashXor, fingerprint);
            atomicAdd(lsv_hashSum, hash_pcg_11(fingerprint));
        }
        lsv_storePair(LSV_MOMENTS_OFFSET + (brick * LSV_DIRECTIONS + bin) * 2u, moments / float(lsv_count));
        atomicAdd(lsv_probeCount, lsv_count);
    }
    #elif LSV_STAGE == 4
    if (lsv_frame != 31u) return;
    uint brick = gl_WorkGroupID.x;
    if (lsv_data[LSV_COUNTS_OFFSET + brick] == 0u) return;
    uint coefficient = gl_LocalInvocationID.x;
    if (coefficient >= 9u) return;
    vec2 value = vec2(0.0);
    for (uint bin = 0u; bin < LSV_DIRECTIONS; ++bin) {
        value += lsv_loadPair(LSV_MOMENTS_OFFSET + (brick * LSV_DIRECTIONS + bin) * 2u)
            * lsv_projectionWeight(lsv_direction(bin), coefficient);
    }
    lsv_storePair(LSV_SH_OFFSET + (brick * 9u + coefficient) * 2u, value);
    #else
    if (gl_GlobalInvocationID.x == 0u) {
        if (lsv_frame == 31u) {
            lsv_ready = 1u;
            ++lsv_trainingRuns;
        }
        ++lsv_frame;
        lsv_cameraBrickX = uint(cameraPositionInt.x >> 4);
        lsv_cameraBrickY = uint(cameraPositionInt.y >> 4);
        lsv_cameraBrickZ = uint(cameraPositionInt.z >> 4);
    }
    #endif
}

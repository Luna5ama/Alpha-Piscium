#include "/techniques/Bloom.glsl"

#ifndef BLOOM_DPF_D2_MASK
#define BLOOM_DPF_D2_MASK 0
#endif
#if defined(BLOOM_DPF_FEATURE_COUNT) && defined(BLOOM_DOWNSAMPLE)
    #if BLOOM_PASS == 2
    #define BLOOM_DPF_WRITE_MASK (((1 << BLOOM_DPF_FEATURE_COUNT) - 1) ^ BLOOM_DPF_D2_MASK)
    #elif BLOOM_PASS == 3
    #define BLOOM_DPF_WRITE_MASK BLOOM_DPF_D2_MASK
    #endif
#endif
#ifndef BLOOM_DPF_WRITE_MASK
#define BLOOM_DPF_WRITE_MASK 0
#endif

layout(local_size_x = 16, local_size_y = 16) in;
#if BLOOM_DPF_WRITE_MASK != 0 && BLOOM_PASS == 2
const vec2 workGroupsRender = vec2(0.5, 0.5);
#elif BLOOM_DPF_WRITE_MASK != 0 && BLOOM_PASS == 3
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif defined(BLOOM_FINAL_RECONSTRUCTION) && !defined(BLOOM_DPF_HAS_D0)
const ivec3 workGroups = ivec3(1, 1, 1);
#else
const vec2 workGroupsRender = vec2(BLOOM_RENDER_SCALE, BLOOM_RENDER_SCALE);
#endif

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

void main() {
    #if !defined(BLOOM_FINAL_RECONSTRUCTION) || defined(BLOOM_DPF_HAS_D0)
    #if !defined(BLOOM_DOWNSAMPLE) && !defined(BLOOM_FINAL_RECONSTRUCTION)
    bloom_prepareInjection();
    #endif
    #ifdef BLOOM_DOWNSAMPLE
    const int outputLevel = BLOOM_PASS;
    #else
    const int outputLevel = BLOOM_PASS - 1;
    #endif
    ivec4 tile = global_mipmapTileCeilPadded[outputLevel];
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    #if BLOOM_DPF_WRITE_MASK != 0
    // The feature slots are disjoint from the raw and reconstruction atlases.
    ivec4 raw = global_mipmapTileCeilPadded[BLOOM_PASS - 1];
    if (all(lessThan(pos, raw.zw))) {
        vec2 center = vec2(pos) + 0.5;
        #if (BLOOM_DPF_WRITE_MASK & 1) != 0
        transient_gi_blurDiff2_store(pos, vec4(bloom_dpfFeature(center, vec4(raw), 0), 0.0));
        #endif
        #if (BLOOM_DPF_WRITE_MASK & 2) != 0
        transient_gi_blurSpec2_store(pos, vec4(bloom_dpfFeature(center, vec4(raw), 1), 0.0));
        #endif
        #if (BLOOM_DPF_WRITE_MASK & 4) != 0
        transient_gi_diffShadingOutput_store(pos, vec4(bloom_dpfFeature(center, vec4(raw), 2), 0.0));
        #endif
        #if (BLOOM_DPF_WRITE_MASK & 8) != 0
        transient_gi3Reprojected_store(pos, vec4(bloom_dpfFeature(center, vec4(raw), 3), 0.0));
        #endif
        #if (BLOOM_DPF_WRITE_MASK & 16) != 0
        transient_gi_specShadingOutput_store(pos, vec4(bloom_dpfFeature(center, vec4(raw), 4), 0.0));
        #endif
        #if (BLOOM_DPF_WRITE_MASK & 32) != 0
        transient_gi_blurSpec1_store(pos, vec4(bloom_dpfFeature(center, vec4(raw), 5), 0.0));
        #endif
    }
    #endif
    bool valid = all(lessThan(pos, tile.zw));
    #ifdef BLOOM_FINAL_RECONSTRUCTION
    vec3 fine = bloom_fullResolutionInjection(vec2(pos) + 0.5, vec4(tile), valid);
    #endif
    if (valid) {
        ivec2 atlasPos = tile.xy + pos;
        vec3 value = bloom_filterPixel(pos);
        #ifdef BLOOM_DOWNSAMPLE
        transient_bloomRaw_store(atlasPos, vec4(value, 0.0));
        #else
        #ifdef BLOOM_FIRST_UPSAMPLE
        value *= bloom_deepestWeight;
        #endif
        #ifdef BLOOM_FINAL_RECONSTRUCTION
        // The packed raw pyramid is dead once reconstruction reaches D0.
        transient_bloomRaw_store(atlasPos, vec4(value + fine, 0.0));
        #else
        value += bloom_injection(vec2(pos) + 0.5, vec4(tile));
        transient_bloomReconstruction_store(atlasPos, vec4(value, 0.0));
        #endif
        #endif
    }
    #endif
}

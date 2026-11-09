#include "/techniques/Bloom.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(BLOOM_RENDER_SCALE, BLOOM_RENDER_SCALE);

#ifdef BLOOM_DOWNSAMPLE
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
#else
layout(rgba16f) uniform image2D uimg_rgba16f;
#endif

void main() {
    #ifdef BLOOM_DOWNSAMPLE
    const int outputLevel = BLOOM_PASS;
    #else
    const int outputLevel = BLOOM_PASS - 1;
    #endif
    ivec4 tile = global_mipmapTileCeilPadded[outputLevel];
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(pos, tile.zw))) {
        ivec2 atlasPos = tile.xy + pos;
        vec3 value = bloom_filterPixel(pos);
        #ifndef BLOOM_DOWNSAMPLE
        #ifdef BLOOM_FIRST_UPSAMPLE
        value *= 0.125;
        #endif
        // The finer tile still holds the downsample, read by its own writer.
        value += transient_bloom_load(atlasPos).rgb * 0.125;
        #endif
        transient_bloom_store(atlasPos, vec4(value, 0.0));
    }
}

#include "/techniques/Bloom.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(BLOOM_RENDER_SCALE, BLOOM_RENDER_SCALE);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

void main() {
    #ifndef BLOOM_DOWNSAMPLE
    bloom_prepareInjection();
    #endif
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
        #ifdef BLOOM_DOWNSAMPLE
        transient_bloomRaw_store(atlasPos, vec4(value, 0.0));
        #else
        #ifdef BLOOM_FIRST_UPSAMPLE
        value *= bloom_deepestWeight;
        #endif
        value += bloom_injection(vec2(pos) + 0.5, vec4(tile));
        transient_bloomReconstruction_store(atlasPos, vec4(value, 0.0));
        #endif
    }
}

#define BLOOM_ZERO_EXTEND
#include "/techniques/Bloom.glsl"
#include "/techniques/BloomShapeTaps.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

void main() {
    ivec4 tile = global_mipmapTileCeilPadded[1];
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(pos, tile.zw))) {
        vec3 value = bloom_shape(vec2(pos) + 0.5, vec4(tile));
        transient_bloomShape_store(pos, vec4(value * bloom_shapeGain, 0.0));
    }
}

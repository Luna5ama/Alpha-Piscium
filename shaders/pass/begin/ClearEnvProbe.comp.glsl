#include "/Base.glsl"
#include "/techniques/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(16, 16, 1);

layout(rgba32ui) uniform writeonly uimage2D uimg_frgba32ui;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    persistent_envProbe_store(texelPos + ivec2(ENV_PROBE_SIZEI.x * 2, 0), uvec4(0u));
}

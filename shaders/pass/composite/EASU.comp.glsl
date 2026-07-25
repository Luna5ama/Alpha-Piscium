#include "/Base.glsl"
#include "/techniques/ffx/fsr1/ffx_fsr1_easu.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f) uniform restrict writeonly image2D uimg_fsr1Easu;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, imageSize(uimg_fsr1Easu)))) {
        imageStore(uimg_fsr1Easu, texelPos, vec4(fsr1_easu(texelPos), 1.0));
    }
}

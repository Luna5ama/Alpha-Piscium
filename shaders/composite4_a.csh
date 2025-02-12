    #version 460 compatibility

#include "general/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 2) in;
const ivec3 workGroups = ivec3(32, 32, 1);

uniform sampler2D usam_temp2;
uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;

layout(rgba32ui) uniform writeonly uimage2D uimg_envProbe;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uvec4 outputData;

    if (envProbe_update(usam_gbufferData, usam_gbufferViewZ, usam_temp2, texelPos, outputData)) {
        imageStore(uimg_envProbe, texelPos, outputData);
    }
}
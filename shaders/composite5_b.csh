#version 460 compatibility

#include "/general/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(2048, 1, 1);


layout(rgba32ui) uniform restrict uimage2D uimg_envProbe;

void main() {
    ivec2 texelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 outputPos = texelPos;
    outputPos.x += 512;

    EnvProbeData outputData;
    if (envProbe_update(usam_gbufferData32UI, usam_gbufferData8UN, usam_gbufferViewZ, usam_packedZN, texelPos, outputData)) {
        imageStore(uimg_envProbe, outputPos, envProbe_encode(outputData));
    }
}
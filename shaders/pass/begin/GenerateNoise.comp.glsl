#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(16, 16, 1);

layout(rgba8) uniform writeonly image2D uimg_noiseGen;

void main() {
    vec4 noiseOutput = vec4(0.0);
    noiseOutput.xy = hash_uintToFloat(hash_22_q3(gl_GlobalInvocationID.xy));
    imageStore(uimg_noiseGen, ivec2(gl_GlobalInvocationID.xy), noiseOutput);
}
#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(2048, 1, 1);

layout(rgba16f) uniform restrict writeonly image2D uimg_cfrgba16f;

void main() {
    ivec2 texelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 inputPos = texelPos;
    uvec4 prevData = texelFetch(usam_envProbe, inputPos, 0);
    EnvProbeData outputData = envProbe_decode(prevData);

    ivec2 prevPosWritePos = texelPos;
    prevPosWritePos.y += ENV_PROBE_SIZEI.y;
    imageStore(uimg_cfrgba16f, prevPosWritePos, vec4(outputData.scenePos, 0.0));

    ivec2 outputPos = texelPos;
    if (envProbe_reproject(texelPos, outputData, outputPos)) {
        imageStore(uimg_cfrgba16f, outputPos, vec4(outputData.scenePos, 1.0));
    }
}
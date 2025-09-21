#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(512, 2, 3);

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_envProbe;

void main() {
    ivec2 sliceTexelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 sliceID = ivec2(gl_GlobalInvocationID.yz);
    ivec2 inputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;

    vec4 currData = texelFetch(usam_cfrgba16f, inputPos, 0);
    float worldDistance = currData.w == 0.0 ? 65536.0 : length(currData.xyz);

    vec2 centerCurrSliceUV = coords_texelToUV(sliceTexelPos, ENV_PROBE_RCP);
    vec2 centerCurrSliceID = vec2(sliceID);
    vec3 centerCurrWorldDir = vec3(-1.0);
    coords_cubeMapBackward(centerCurrWorldDir, centerCurrSliceUV, centerCurrSliceID);

    vec3 currScenePos = worldDistance * centerCurrWorldDir;
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec3 currToPrevScenePos = currScenePos + cameraDelta;
    vec3 currToPrevWorldDir = normalize(currToPrevScenePos);
    vec2 currToPrevSliceUV = vec2(-1.0);
    vec2 currToPrevSliceID = vec2(-1.0);
    coords_cubeMapForward(currToPrevWorldDir, currToPrevSliceUV, currToPrevSliceID);

    if (any(notEqual(currToPrevSliceUV, saturate(currToPrevSliceUV)))) {
        return;
    }

    vec2 centerToPrevTexelPosF = (currToPrevSliceUV + currToPrevSliceID) * ENV_PROBE_SIZE;
    ivec2 centerTexelPos = ivec2(centerToPrevTexelPosF);
    EnvProbeData dataSum;
    envProbe_initData(dataSum);
    {
        float maxDot = 0.999;

        for (int yo = -1; yo <= 1; ++yo) {
            for (int xo = -1; xo <= 1; ++xo) {
                ivec2 offset = ivec2(xo, yo);
                ivec2 samplePos = (centerTexelPos + offset);
                EnvProbeData sampleData = envProbe_decode(texelFetch(usam_envProbe, samplePos, 0));

                vec3 samplePrevPos = sampleData.scenePos;
                vec3 sampleCurrPos = samplePrevPos - cameraDelta;
                vec3 sampleCurrDir = normalize(sampleCurrPos);
                if (envProbe_isSky(sampleData)) {
                    sampleCurrPos = sampleCurrDir * 4096.0;
                }
                sampleData.scenePos = sampleCurrPos;

                float dirDot = dot(centerCurrWorldDir, sampleCurrDir);

                if (dirDot > maxDot) {
                    maxDot = dirDot;
                    dataSum = sampleData;
                }
            }
        }
    }

    ivec2 outputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;
    outputPos.x += ENV_PROBE_SIZEI.x * 2;
    imageStore(uimg_envProbe, outputPos, envProbe_encode(dataSum));
}
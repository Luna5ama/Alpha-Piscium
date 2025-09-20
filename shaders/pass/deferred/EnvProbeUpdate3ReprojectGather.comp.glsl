#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(2048, 1, 1);

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_envProbe;

void main() {
    ivec2 texelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 centerCurrTexelPos = texelPos;

    vec4 currData = texelFetch(usam_cfrgba16f, centerCurrTexelPos, 0);
    float leng = currData.w == 0.0 ? 65536.0 : length(currData.xyz);

    vec2 centerCurrTexelPosF = vec2(texelPos) + 0.5;
    vec2 centerCurrScreenPos = centerCurrTexelPosF * ENV_PROBE_RCP;
    vec3 centerCurrWorldDir = coords_mercatorBackward(centerCurrScreenPos);

    vec3 currScenePos = leng * centerCurrWorldDir;
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec3 currToPrevScenePos = currScenePos + cameraDelta;
    vec3 currToPrevWorldDir = normalize(currToPrevScenePos);
    vec2 currToPrevScreenPos = coords_mercatorForward(currToPrevWorldDir);

    if (any(notEqual(currToPrevScreenPos, saturate(currToPrevScreenPos)))) {
        return;
    }

    vec2 centerToPrevTexelPosF = currToPrevScreenPos * ENV_PROBE_SIZE;
    ivec2 centerTexelPos = ivec2(centerToPrevTexelPosF);
    EnvProbeData dataSum;
    envProbe_initData(dataSum);
    {
        vec2 centerPosF = vec2(centerTexelPos) + 0.5;
        float maxDot = 0.99;

        for (int yo = -1; yo <= 1; ++yo) {
            for (int xo = -1; xo <= 1; ++xo) {
                ivec2 offset = ivec2(xo, yo);
                ivec2 samplePos = (centerTexelPos + offset) % ENV_PROBE_SIZEI;
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

    ivec2 outputPos = texelPos;
    outputPos.x += ENV_PROBE_SIZEI.x;
    imageStore(uimg_envProbe, outputPos, envProbe_encode(dataSum));
}
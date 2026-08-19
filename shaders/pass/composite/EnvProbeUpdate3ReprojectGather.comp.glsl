#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"
#include "/techniques/gi/RadianceCache.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(512, 2, 3);

layout(rgba32ui) uniform restrict writeonly uimage2D uimg_envProbe;

vec3 envProbe_decodeScenePos(uvec4 packedData) {
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    return vec3(temp2.y, temp3);
}

void main() {
    ivec2 sliceTexelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 sliceID = ivec2(gl_GlobalInvocationID.yz);
    ivec2 inputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;

    vec4 currData = persistent_envProbeTemp_fetch(inputPos);
    float worldDistance = currData.w == 0.0 ? 65536.0 : length(currData.xyz);

    vec2 centerCurrSliceUV = coords_texelToUV(sliceTexelPos, ENV_PROBE_RCP);
    vec2 centerCurrSliceID = vec2(sliceID);
    vec3 centerCurrWorldDir = vec3(-1.0);
    coords_cubeMapBackward(centerCurrWorldDir, centerCurrSliceUV, centerCurrSliceID);

    vec3 currScenePos = worldDistance * centerCurrWorldDir;
    vec3 currToPrevScenePos = currScenePos + uval_cameraDelta;
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
        uvec4 bestPackedData = uvec4(0u);
        vec3 bestScenePos = vec3(0.0);

        for (int yo = -1; yo <= 1; ++yo) {
            for (int xo = -1; xo <= 1; ++xo) {
                ivec2 offset = ivec2(xo, yo);
                ivec2 samplePos = (centerTexelPos + offset);
                uvec4 samplePackedData = texelFetch(usam_envProbe, samplePos, 0);

                vec3 samplePrevPos = envProbe_decodeScenePos(samplePackedData);
                vec3 sampleCurrPos = samplePrevPos - uval_cameraDelta;
                vec3 sampleCurrDir = normalize(sampleCurrPos);
                float samplePrevDistSq = dot(samplePrevPos, samplePrevPos);
                if (samplePrevDistSq == 0.0 || samplePrevDistSq > 4194304.0) {
                    sampleCurrPos = sampleCurrDir * 4096.0;
                }

                float dirDot = dot(centerCurrWorldDir, sampleCurrDir);

                if (dirDot > maxDot) {
                    maxDot = dirDot;
                    bestPackedData = samplePackedData;
                    bestScenePos = sampleCurrPos;
                }
            }
        }

        if (maxDot > 0.999) {
            dataSum = envProbe_decode(bestPackedData);
            dataSum.scenePos = bestScenePos;
        }
    }

    ivec2 outputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;
    outputPos.x += ENV_PROBE_SIZEI.x * 2;
    imageStore(uimg_envProbe, outputPos, envProbe_encode(dataSum));
}

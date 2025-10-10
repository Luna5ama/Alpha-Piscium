#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(512, 2, 3);

layout(rgba32ui) uniform restrict uimage2D uimg_envProbe;

bool envProbe_update(ivec2 sliceTexelPos, ivec2 sliceID, inout EnvProbeData outputData) {
    vec3 pixelWorldDir = vec3(0.0);
    vec2 sliceUV = coords_texelToUV(sliceTexelPos, ENV_PROBE_RCP);
    coords_cubeMapBackward(pixelWorldDir, sliceUV, vec2(sliceID));

    vec3 pixelViewDir = coords_dir_worldToView(pixelWorldDir);
    vec4 clipPos = global_camProj * vec4(pixelViewDir, 1.0);

    envProbe_initData(outputData);

    if (!all(lessThan(abs(clipPos.xyz), clipPos.www))) {
        return false;
    }

    vec2 ndcPos = clipPos.xy / clipPos.w;
    vec2 screenPos = ndcPos * 0.5 + 0.5;
    ivec2 texelPos2x2 = ivec2(screenPos * global_mipmapSizes[1]);
    uint edgeFlag = uint(any(lessThanEqual(texelPos2x2, ivec2(1))));
    edgeFlag |= uint(any(greaterThanEqual(texelPos2x2, global_mipmapSizesI[1] - 2)));
    if (bool(edgeFlag)) {
        return false;
    }

    ivec2 texelPos1x1 = texelPos2x2 << 1;

    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos1x1, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos1x1, 0), gData);
    if (gData.isHand) {
        return false;
    }

    float viewZ = texelFetch(usam_gbufferViewZ, texelPos1x1, 0).r;
    vec3 realViewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec4 realScenePos = gbufferModelViewInverse * vec4(realViewPos, 1.0);

    uvec2 radianceData = texelFetch(usam_packedZN, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), 0).xy;
    vec4 radiance = vec4(unpackHalf2x16(radianceData.x), unpackHalf2x16(radianceData.y));

    outputData.radiance = viewZ == -65536.0 ? vec3(0.0) : radiance.rgb;
    outputData.normal = mat3(gbufferModelViewInverse) * gData.normal;
    outputData.scenePos = normalize(realScenePos.xyz) * min(length(realScenePos.xyz), 8192.0);

    return true;
}

void main() {
    ivec2 sliceTexelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 sliceID = ivec2(gl_GlobalInvocationID.yz);
    ivec2 outputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;

    EnvProbeData outputData;
    envProbe_initData(outputData);
    ivec2 prevTexelPos = outputPos;
    prevTexelPos.x += ENV_PROBE_SIZEI.x * 2;
    EnvProbeData prevData = envProbe_decode(imageLoad(uimg_envProbe, prevTexelPos));

    if (envProbe_update(sliceTexelPos, sliceID, outputData)) {
        if (envProbe_hasData(prevData)) {
            outputData.radiance = mix(outputData.radiance, prevData.radiance, 0.8 * global_historyResetFactor);
            outputData.normal = normalize(mix(outputData.normal, prevData.normal, 0.8 * global_historyResetFactor));
        }
        imageStore(uimg_envProbe, outputPos, envProbe_encode(outputData));
    } else {
        if (global_historyResetFactor > 0.1) {
            outputData = prevData;
            outputData.radiance *= global_historyResetFactor;
        }
        imageStore(uimg_envProbe, outputPos, envProbe_encode(outputData));
    }
}
#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Colors.glsl"
#include "/util/Material.glsl"

const ivec2 ENV_PROBE_SIZEI = ivec2(512, 512);
const vec2 ENV_PROBE_SIZE = vec2(512.0, 512.0);
const vec2 ENV_PROBE_RCP = vec2(1.0 / 512.0, 1.0 / 512.0);

struct EnvProbeData {
    vec3 radiance;
    vec3 normal;
    vec3 scenePos;
};

void envProbe_initData(out EnvProbeData envProbeData) {
    envProbeData.radiance = vec3(0.0);
    envProbeData.normal = vec3(0.0);
    envProbeData.scenePos = vec3(0.0);
}

bool envProbe_isSky(EnvProbeData envProbeData) {
    float distSq = dot(envProbeData.scenePos, envProbeData.scenePos);
    uint flag = uint(distSq == 0.0);
    flag |= uint(distSq > 268386304.0);
    return bool(flag);
}

bool envProbe_hasData(EnvProbeData envProbeData) {
    return !all(equal(envProbeData.scenePos, vec3(0.0)));
}

EnvProbeData envProbe_add(EnvProbeData a, EnvProbeData b) {
    EnvProbeData result;
    result.radiance = a.radiance + b.radiance;
    result.normal = a.normal + b.normal;
    result.scenePos = a.scenePos + b.scenePos;
    return result;
}

EnvProbeData envProbe_decode(uvec4 packedData) {
    EnvProbeData unpackedData;
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    unpackedData.radiance = vec3(temp1, temp2.x);
    unpackedData.scenePos = vec3(temp2.y, temp3);
    unpackedData.normal = coords_octDecode11(unpackSnorm2x16(packedData.w));
    return unpackedData;
}

uvec4 envProbe_encode(EnvProbeData unpackedData) {
    uvec4 packedData;
    packedData.x = packHalf2x16(unpackedData.radiance.rg);
    packedData.y = packHalf2x16(vec2(unpackedData.radiance.b, unpackedData.scenePos.x));
    packedData.z = packHalf2x16(unpackedData.scenePos.yz);
    packedData.w = packSnorm2x16(coords_octEncode11(unpackedData.normal));
    return packedData;
}

bool envProbe_reproject(ivec2 prevEnvProbeTexelPos, inout EnvProbeData envProbeData, out ivec2 outputTexelPos) {
    if (all(equal(envProbeData.scenePos, vec3(0.0)))) {
        return false;
    }

    vec3 prevEnvProbeScenePos = envProbeData.scenePos;
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec3 currEnvProbeScenePos = prevEnvProbeScenePos - cameraDelta * 2.0;
    vec3 currEnvProbeWorldDir = normalize(currEnvProbeScenePos);
    vec2 currEnvProbeScreenPos = coords_mercatorForward(currEnvProbeWorldDir);

    if (any(notEqual(currEnvProbeScreenPos, saturate(currEnvProbeScreenPos)))) {
        return false;
    }

    outputTexelPos = ivec2(currEnvProbeScreenPos * ENV_PROBE_SIZE);

    envProbeData.scenePos = currEnvProbeScenePos;

    if (envProbe_isSky(envProbeData)) {
        envProbeData.scenePos = normalize(envProbeData.scenePos) * 65536.0;
    }

    return true;
}

bool envProbe_update(
usampler2D gbufferData32UI, sampler2D gbufferData8UN, sampler2D gbufferViewZ, usampler2D packedZN,
ivec2 envProbeTexelPos, out EnvProbeData outputData
) {
    vec2 envProbeScreenPos = (vec2(envProbeTexelPos) + 0.5) * ENV_PROBE_RCP;
    vec3 envProbePixelWorldDir = coords_mercatorBackward(envProbeScreenPos);
    vec4 viewPos = gbufferModelView * vec4(envProbePixelWorldDir, 1.0);
    viewPos.xyz -= gbufferModelView[3].xyz;
    vec4 clipPos = gbufferProjection * viewPos;

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

    GBufferData gData;
    gbufferData1_unpack(texelFetch(gbufferData32UI, texelPos1x1, 0), gData);
    gbufferData2_unpack(texelFetch(gbufferData8UN, texelPos1x1, 0), gData);
    if (gData.isHand) {
        return false;
    }

    float viewZ = texelFetch(gbufferViewZ, texelPos1x1, 0).r;
    vec3 realViewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);
    vec4 realScenePos = gbufferModelViewInverse * vec4(realViewPos, 1.0);

    uvec2 radianceData = texelFetch(packedZN, texelPos2x2 + ivec2(0, global_mipmapSizesI[1].y), 0).xy;
    vec4 radiance = vec4(unpackHalf2x16(radianceData.x), unpackHalf2x16(radianceData.y));

    outputData.radiance = viewZ == -65536.0 ? vec3(0.0) : radiance.rgb;
    outputData.normal = mat3(gbufferModelViewInverse) * gData.normal;
    outputData.scenePos = realScenePos.xyz;

    return true;
}
#include "../_Util.glsl"

const ivec2 ENV_PROBE_SIZEI = ivec2(512, 512);
const vec2 ENV_PROBE_SIZE = vec2(512.0, 512.0);
const vec2 ENV_PROBE_RCP = vec2(1.0 / 512.0, 1.0 / 512.0);

struct EnvProbeData {
    vec3 radiance;
    vec3 normal;
    float dist;
    float emissive;
};

EnvProbeData envProbe_decode(uvec4 packedData) {
    EnvProbeData unpackedData;
    vec2 radianceRG = unpackHalf2x16(packedData.x);
    vec2 radianceBA = unpackHalf2x16(packedData.y);
    unpackedData.radiance = vec3(radianceRG, radianceBA.x);
    unpackedData.emissive = radianceBA.y;
    unpackedData.normal = coords_octDecode11(unpackSnorm2x16(packedData.z));
    unpackedData.dist = uintBitsToFloat(packedData.w);
    return unpackedData;
}

uvec4 envProbe_encode(EnvProbeData unpackedData) {
    uvec4 packedData;
    packedData.x = packHalf2x16(vec2(unpackedData.radiance.r, unpackedData.radiance.g));
    packedData.y = packHalf2x16(vec2(unpackedData.radiance.b, unpackedData.emissive));
    packedData.z = packSnorm2x16(coords_octEncode11(unpackedData.normal));
    packedData.w = floatBitsToUint(unpackedData.dist);
    return packedData;
}

bool envProbe_update(
usampler2D gbufferData, sampler2D gbufferViewZ,
sampler2D inputViewColor,
//sampler2D inputColorZ,
ivec2 envProbeTexelPos,
out uvec4 outputData
) {
    vec2 envCacheScreenPos = (vec2(envProbeTexelPos) + 0.5) * ENV_PROBE_RCP;
    vec3 envCachePixelWorldDir = coords_mercatorBackward(envCacheScreenPos);
    vec4 viewPos = gbufferModelView * vec4(envCachePixelWorldDir, 1.0);
    viewPos.xyz -= gbufferModelView[3].xyz;
    vec4 clipPos = gbufferProjection * viewPos;

    outputData = uvec4(0.0);
    if (!all(lessThan(abs(clipPos.xyz), clipPos.www))) {
        return false;
    }

    vec2 ndcPos = clipPos.xy / clipPos.w;
    vec2 screenPos = ndcPos * 0.5 + 0.5;
    ivec2 texelPos = ivec2(screenPos * global_mainImageSize);

    GBufferData gData;
    gbuffer_unpack(texelFetch(gbufferData, texelPos, 0), gData);
    if (gData.isHand) {
        return false;
    }

    EnvProbeData envProbeData;
    envProbeData.normal = mat3(gbufferModelViewInverse) * gData.normal;
    envProbeData.radiance = texture(inputViewColor, screenPos).rgb;

    Material material = material_decode(gData);
    envProbeData.emissive = colors_srgbLuma(material.emissive);

    float viewZ = texelFetch(gbufferViewZ, texelPos, 0).r;
    vec3 realViewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);
    vec4 realScenePos = gbufferModelViewInverse * vec4(realViewPos, 1.0);
    envProbeData.dist = viewZ == -65536.0 ? 32768.0 : length(realScenePos.xyz);

    outputData = envProbe_encode(envProbeData);

    return true;
}
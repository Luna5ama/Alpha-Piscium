#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Colors.glsl"
#include "/util/Material.glsl"

const ivec2 ENV_PROBE_SIZEI = ivec2(256, 256);
const vec2 ENV_PROBE_SIZE = vec2(256.0, 256.0);
const vec2 ENV_PROBE_RCP = vec2(1.0 / 256.0, 1.0 / 256.0);

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
    flag |= uint(distSq > 4194304.0);
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

EnvProbeData envProbe_addWeighted(EnvProbeData a, EnvProbeData b, float weight) {
    EnvProbeData result;
    result.radiance = a.radiance + b.radiance * weight;
    result.normal = a.normal + b.normal * weight;
    result.scenePos = a.scenePos + b.scenePos * weight;
    return result;
}

EnvProbeData envProbe_decode(uvec4 packedData) {
    EnvProbeData unpackedData;
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    unpackedData.radiance = vec3(temp1, temp2.x);
    unpackedData.scenePos = vec3(temp2.y, temp3);
    unpackedData.normal = nzpacking_unpackNormalOct32(packedData.w);
    return unpackedData;
}

uvec4 envProbe_encode(EnvProbeData unpackedData) {
    uvec4 packedData;
    packedData.x = packHalf2x16(unpackedData.radiance.rg);
    packedData.y = packHalf2x16(vec2(unpackedData.radiance.b, unpackedData.scenePos.x));
    packedData.z = packHalf2x16(unpackedData.scenePos.yz);
    packedData.w = nzpacking_packNormalOct32(unpackedData.normal);
    return packedData;
}
#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(512, 2, 3);

layout(rgba16f) uniform restrict writeonly image2D uimg_frgba16f;

vec3 envProbe_decodeScenePos(uvec4 packedData) {
    return vec3(unpackHalf2x16(packedData.y).y, unpackHalf2x16(packedData.z));
}

bool envProbe_reproject(vec3 prevScenePos, out vec3 outputScenePos, out ivec4 outputCubeMapPos) {
    if (all(equal(prevScenePos, vec3(0.0)))) {
        return false;
    }

    vec3 cameraDelta = uval_cameraDelta;
    vec3 currScenePos = prevScenePos - cameraDelta;
    vec3 currWorldDir = normalize(currScenePos);
    vec2 currSliceUV = vec2(-1.0);
    vec2 currSliceID = vec2(-1.0);
    coords_cubeMapForward(currWorldDir, currSliceUV, currSliceID);

    if (any(notEqual(currSliceUV, saturate(currSliceUV)))) {
        return false;
    }

    outputCubeMapPos = ivec4(ivec2(currSliceUV * ENV_PROBE_SIZE), currSliceID);

    outputScenePos = currScenePos;

    float distSq = dot(outputScenePos, outputScenePos);
    if (distSq == 0.0 || distSq > 4194304.0) {
        outputScenePos = normalize(outputScenePos) * 4096.0;
    }

    return true;
}

void main() {
    ivec2 sliceTexelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 sliceID = ivec2(gl_GlobalInvocationID.yz);
    ivec2 inputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;
    uvec4 prevData = texelFetch(usam_envProbe, inputPos, 0);
    vec3 outputScenePos;

    ivec4 outputCubeMapPos;
    if (envProbe_reproject(envProbe_decodeScenePos(prevData), outputScenePos, outputCubeMapPos)) {
        ivec2 outputPos = outputCubeMapPos.xy + outputCubeMapPos.zw * ENV_PROBE_SIZEI;
        persistent_envProbeTemp_store(outputPos, vec4(outputScenePos, 1.0));
    }
}
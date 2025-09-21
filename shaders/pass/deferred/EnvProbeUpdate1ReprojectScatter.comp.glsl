#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(512, 2, 3);

layout(rgba16f) uniform restrict writeonly image2D uimg_cfrgba16f;

bool envProbe_reproject(ivec4 inputCubeMapPos, inout EnvProbeData envProbeData, out ivec4 outputCubeMapPos) {
    if (all(equal(envProbeData.scenePos, vec3(0.0)))) {
        return false;
    }

    vec3 prevScenePos = envProbeData.scenePos;
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec3 currScenePos = prevScenePos - cameraDelta;
    vec3 currWorldDir = normalize(currScenePos);
    vec2 currSliceUV = vec2(-1.0);
    vec2 currSliceID = vec2(-1.0);
    coords_cubeMapForward(currWorldDir, currSliceUV, currSliceID);

    if (any(notEqual(currSliceUV, saturate(currSliceUV)))) {
        return false;
    }

    outputCubeMapPos = ivec4(ivec2(currSliceUV * ENV_PROBE_SIZE), currSliceID);

    envProbeData.scenePos = currScenePos;

    if (envProbe_isSky(envProbeData)) {
        envProbeData.scenePos = normalize(envProbeData.scenePos) * 4096.0;
    }

    return true;
}

void main() {
    ivec2 sliceTexelPos = ivec2(morton_32bDecode(gl_GlobalInvocationID.x));
    ivec2 sliceID = ivec2(gl_GlobalInvocationID.yz);
    ivec2 inputPos = sliceTexelPos + sliceID * ENV_PROBE_SIZEI;
    uvec4 prevData = texelFetch(usam_envProbe, inputPos, 0);
    EnvProbeData outputData = envProbe_decode(prevData);

    ivec4 inputCubeMapPos = ivec4(sliceTexelPos, sliceID);
    ivec4 outputCubeMapPos = inputCubeMapPos;
    if (envProbe_reproject(inputCubeMapPos, outputData, outputCubeMapPos)) {
        ivec2 outputPos = outputCubeMapPos.xy + outputCubeMapPos.zw * ENV_PROBE_SIZEI;
        imageStore(uimg_cfrgba16f, outputPos, vec4(outputData.scenePos, 1.0));
    }
}
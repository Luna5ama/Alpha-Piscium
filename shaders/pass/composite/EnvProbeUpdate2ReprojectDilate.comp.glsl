#include "/techniques/EnvProbe.glsl"
#include "/util/Morton.glsl"

#if PASS == 1
#define READ_OFFSET 0
#define WRITE_OFFSET (ENV_PROBE_SIZEI.x * 2)
#else
#define READ_OFFSET (ENV_PROBE_SIZEI.x * 2)
#define WRITE_OFFSET 0
#endif

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(16, 16, 6);

layout(rgba16f) uniform image2D uimg_frgba16f;

shared vec4 shared_scenePos[20][20];

ivec2 sliceID = ivec2(0);

void loadSharedData(uint index) {
    if (index < 400) {
        uvec2 localPos = uvec2(index % 20, index / 20);
        ivec2 globalPos = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
        globalPos += ivec2(localPos);
        globalPos -= 2;
        globalPos = clamp(globalPos, ivec2(0), ENV_PROBE_SIZEI - 1);
        globalPos += sliceID * ENV_PROBE_SIZEI;
        globalPos.x += READ_OFFSET;
        shared_scenePos[localPos.y][localPos.x] = persistent_envProbeTemp_load(globalPos);
    }
}

void main() {
    sliceID = ivec2(gl_WorkGroupID.z % 2, gl_WorkGroupID.z / 2);
    loadSharedData(gl_LocalInvocationIndex);
    loadSharedData(gl_LocalInvocationIndex + 256u);
    barrier();

    vec4 sum = vec4(0.0);

    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            if (x != 0 || y != 0) {
                ivec2 offset = ivec2(x, y);
                const float K = 0.1;
                vec2 weightXY = rcp(vec2(abs(offset) + 1));
                float weight = weightXY.x * weightXY.y;
                ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + offset + 2;
                vec4 data = shared_scenePos[localPos.y][localPos.x];
                sum += data * weight;
            }
        }
    }
    ivec2 centerLocalPos = ivec2(gl_LocalInvocationID.xy) + 2;
    vec4 centerData = shared_scenePos[centerLocalPos.y][centerLocalPos.x];
    sum = sum.w < 0.001 || centerData.a > 0.1 ? centerData : sum / sum.w;

    ivec2 centerGlobalPos = ivec2(gl_GlobalInvocationID.xy);
    centerGlobalPos += sliceID * ENV_PROBE_SIZEI;
    centerGlobalPos.x += WRITE_OFFSET;
    persistent_envProbeTemp_store(centerGlobalPos, sum);
}
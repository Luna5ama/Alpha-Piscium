#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "RTWSM.glsl"

layout(local_size_x = RTWSM_IMAP_SIZE, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 2, 1);

layout(r32f) uniform restrict image2D uimg_fr32f;

shared float shared_buffer[RTWSM_IMAP_SIZE];

#define RADIUS 2
#define SAMPLES 5

void main() {
    ivec2 ti = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dataPos = ti;
    shared_buffer[gl_LocalInvocationID.x] = persistent_rtwsm_importance1D_load(dataPos).r;

    barrier();

    {
        float value = 0.0;
        for (int i = 0; i < SAMPLES; i++) {
            int pos = int(gl_LocalInvocationID.x) + i - RADIUS;
            pos = clamp(pos, 0, int(gl_WorkGroupSize.x) - 1);
            value += shared_buffer[pos];
        }
        value /= float(SAMPLES);
        shared_buffer[gl_LocalInvocationID.x] = value;
        barrier();
    }

    {
        float value = 0.0;
        for (int i = 0; i < SAMPLES; i++) {
            int pos = int(gl_LocalInvocationID.x) + i - RADIUS;
            pos = clamp(pos, 0, int(gl_WorkGroupSize.x) - 1);
            value += shared_buffer[pos];
        }
        value /= float(SAMPLES);
        persistent_rtwsm_importance1D_store(dataPos, vec4(value));
    }
}

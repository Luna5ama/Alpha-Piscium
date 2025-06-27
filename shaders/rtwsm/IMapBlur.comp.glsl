#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "RTWSM.glsl"

layout(local_size_x = SETTING_RTWSM_IMAP_SIZE, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 2, 1);

layout(r32f) uniform restrict image2D uimg_rtwsm_imap;

shared float shared_buffer[SETTING_RTWSM_IMAP_SIZE];

#if SETTING_RTWSM_IMAP_SIZE == 256
#define RADIUS 2
#define SAMPLES 5
#elif SETTING_RTWSM_IMAP_SIZE == 512
#define RADIUS 4
#define SAMPLES 9
#elif SETTING_RTWSM_IMAP_SIZE == 1024
#define RADIUS 8
#define SAMPLES 17
#endif

void main() {
    ivec2 ti = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dataPos = ti;
    dataPos.y += IMAP1D_X_Y;
    shared_buffer[gl_LocalInvocationID.x] = imageLoad(uimg_rtwsm_imap, dataPos).r;

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
        imageStore(uimg_rtwsm_imap, dataPos, vec4(value));
    }
}

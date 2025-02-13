#version 460 compatibility

#include "/rtwsm/RTWSM.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(IMAP_SIZE_D16, IMAP_SIZE_D16, 1);

layout(r32ui) uniform writeonly uimage2D uimg_rtwsm_imap;
#define CLEAR_IMAGE1 uimg_rtwsm_imap
#define CLEAR_IMAGE_BOUND ivec4(0, 0, SETTING_RTWSM_IMAP_SIZE, SETTING_RTWSM_IMAP_SIZE)
#define CLEAR_COLOR1 uvec4(0u)

#include "/general/Clear.comp.glsl"
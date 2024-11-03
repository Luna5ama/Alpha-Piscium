#define RENDER_RESOLUTION 1.0 // [0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0]

#define SETTING_SHADOW_MR 2048 // [256 512 1024 2048 4096]
#define SETTING_SHADOW_RD 12 // [4 8 12 16 20 24 28 32]

#define SETTING_RTWSM_IMAP_SIZE 1024 // RTWSM importance map resolution [128 256 512 1024]
#define SETTING_RTWSM_IMP_BBASE 8.0 // [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
#define SETTING_RTWSM_IMP_D 16 // [0 1 2 4 8 16 32 64 128 256]
#define SETTING_RTWSM_IMP_SN 4.0 // [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
#define SETTING_RTWSM_IMP_SE 0.5 // [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

//#define RTWSM_DEBUG

#define SETTING_PCSS_BPF 0.5 // [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0]
#define SETTING_PCSS_VPF 32 // [0 1 2 4 8 16 32 64 128 256 512]
#define SETTING_PCSS_SAMPLE_COUNT 16 // [1 2 4 8 16 32 64 128]
#define SETTING_PCSS_BLOCKER_SEARCH_COUNT 4 // [1 2 4 8 16]
#define SETTING_PCSS_BLOCKER_SEARCH_LOD 4 // [0 1 2 3 4 5 6 7 8]

// Post processing
#define SETTING_OUTPUT_GAMMA 2.2 // [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2.0 2.05 2.1 2.15 2.2 2.25 2.3 2.35 2.4 2.45 2.5 2.55 2.6 2.65 2.7 2.75 2.8 2.85 2.9 2.95 3.0 3.05 3.1 3.15 3.2 3.25 3.3 3.35 3.4 3.45 3.5 3.55 3.6 3.65 3.7 3.75 3.8 3.85 3.9 3.95 4.0]

const int shadowMapResolution = SETTING_SHADOW_MR;
const vec2 SHADOW_MAP_SIZE = vec2(float(shadowMapResolution), 1.0 / float(shadowMapResolution));
const float shadowDistance = SETTING_SHADOW_RD * 16.0;
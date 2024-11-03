#define RENDER_RESOLUTION 1.0 // [0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0]

#define SETTING_SHADOW_MAP_RESOLUTION 2048 // [256 512 1024 2048 4096]

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


const int shadowMapResolution = SETTING_SHADOW_MAP_RESOLUTION;
const vec2 SHADOW_MAP_SIZE = vec2(float(shadowMapResolution), 1.0 / float(shadowMapResolution));
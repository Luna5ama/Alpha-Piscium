#ifndef INCLUDE_base_SuperResolution_glsl
#define INCLUDE_base_SuperResolution_glsl a

#if defined(SR_ENABLE) && SR_ENABLE
#define SUPER_RESOLUTION_ACTIVE 1
#else
#define SUPER_RESOLUTION_ACTIVE 0
#endif

#if SETTING_AA_MODE == 2 && !SUPER_RESOLUTION_ACTIVE
#define INTERNAL_FSR3_ACTIVE 1
#else
#define INTERNAL_FSR3_ACTIVE 0
#endif

#if SETTING_AA_MODE == 1 && !SUPER_RESOLUTION_ACTIVE
#define INTERNAL_TAA_ACTIVE 1
#else
#define INTERNAL_TAA_ACTIVE 0
#endif

#if INTERNAL_FSR3_ACTIVE || (SUPER_RESOLUTION_ACTIVE && SR_SHOULD_APPLY_SCALE)
#define UPSCALER_RECONSTRUCTION_ACTIVE 1
#else
#define UPSCALER_RECONSTRUCTION_ACTIVE 0
#endif

#if SUPER_RESOLUTION_ACTIVE
#define RENDER_SCALE_ACTIVE SR_SHOULD_APPLY_SCALE

uniform vec2 SRJitterOffset;
uniform vec2 SRPreviousJitterOffset;

#define uval_taaJitter SRJitterOffset
#define uval_prevTaaJitter SRPreviousJitterOffset
#define uval_taaJitterUV (SRJitterOffset * uval_mainImageSizeRcp)
#define uval_prevTaaJitterUV (SRPreviousJitterOffset * uval_mainImageSizeRcp)
#elif SETTING_RENDER_SCALE < 10
#define RENDER_SCALE_ACTIVE 1
#else
#define RENDER_SCALE_ACTIVE 0
#endif

#endif

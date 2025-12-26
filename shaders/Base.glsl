#ifndef INCLUDE_Base_glsl
#define INCLUDE_Base_glsl a

#include "/base/Options.glsl"

#if SETTING_SHADOW_MAP_RESOLUTION == 1024
#define SHADOW_MAP_SIZE_D16 64
const int shadowMapResolution = 1024;
#elif SETTING_SHADOW_MAP_RESOLUTION == 2048
#define SHADOW_MAP_SIZE_D16 128
const int shadowMapResolution = 2048;
#elif SETTING_SHADOW_MAP_RESOLUTION == 3072
#define SHADOW_MAP_SIZE_D16 192
const int shadowMapResolution = 3072;
#elif SETTING_SHADOW_MAP_RESOLUTION == 4096
#define SHADOW_MAP_SIZE_D16 256
const int shadowMapResolution = 4096;
#endif

#include "/base/Configs.glsl"
#include "/base/Uniforms.glsl"
#include "/base/CustomUniforms.glsl"
#include "/base/SSBO.glsl"
#include "/base/Textures.glsl"
#include "/base/Textile.glsl"

#endif
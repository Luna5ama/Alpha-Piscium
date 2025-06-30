/*
const int colortex0Format = RGBA16F; // Main 1
const int colortex1Format = RGBA16F; // Temp1
const int colortex2Format = RGBA16F; // Temp2
const int colortex3Format = RGBA16F; // Temp3
const int colortex4Format = RGBA16F; // Temp4
const int colortex5Format = RGBA8; // Temp5
const int colortex6Format = RGBA8; // Temp6
const int colortex7Format = R32UI; // Geometry Normal
const int colortex8Format = RGBA32UI; // GBuffer Data 32UI
const int colortex9Format = RGBA8; // GBuffer Data 8UN
const int colortex10Format = R32F; // GBuffer ViewZ
const int colortex11Format = RGBA16F; // Translucent Color
const int colortex12Format = RGBA32UI; // tempRGBA32UI
const int colortex13Format = RG32UI; // tempR32UI
const int colortex14Format = RG32UI; // packedZN
const int colortex15Format = RGBA16F; // TAA Last

const int shadowcolor0Format = R16F; // Depth offset
const int shadowcolor1Format = RGBA8; // Normal
const int shadowcolor2Format = RGBA8; // Translucent color
*/

#define usam_main colortex0
#define uimg_main colorimg0

#define usam_temp1 colortex1
#define uimg_temp1 colorimg1

#define usam_temp2 colortex2
#define uimg_temp2 colorimg2

#define usam_temp3 colortex3
#define uimg_temp3 colorimg3

#define usam_temp4 colortex4
#define uimg_temp4 colorimg4

#define usam_temp5 colortex5
#define uimg_temp5 colorimg5

#define usam_temp6 colortex6
#define uimg_temp6 colorimg6

#define usam_geometryNormal colortex7
#define uimg_geometryNormal colorimg7

#define usam_gbufferData32UI colortex8
#define uimg_gbufferData32UI colorimg8

#define usam_gbufferData8UN colortex9
#define uimg_gbufferData8UN colorimg9

#define usam_gbufferViewZ colortex10
#define uimg_gbufferViewZ colorimg10

#define usam_translucentColor colortex11
#define uimg_translucentColor colorimg11

#define usam_tempRGBA32UI colortex12
#define uimg_tempRGBA32UI colorimg12

#define usam_tempRG32UI colortex13
#define uimg_tempRG32UI colorimg13

#define usam_packedZN colortex14
#define uimg_packedZN colorimg14

#define usam_taaLast colortex15
#define uimg_taaLast colorimg15

const bool colortex0Clear = false;

const bool colortex1Clear = true;
const vec4 colortex1ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex2Clear = true;
const vec4 colortex2ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex3Clear = true;
const vec4 colortex3ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex4Clear = true;
const vec4 colortex4ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex5Clear = true;
const vec4 colortex5ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex6Clear = true;
const vec4 colortex6ClearColor = vec4(0.0, 0.0, 0.0, 0.0);

const bool colortex7Clear = false;
const bool colortex8Clear = false;
const bool colortex9Clear = false;
const bool colortex10Clear = false;

const bool colortex11Clear = true;
const vec4 colortex11ClearColor = vec4(0.0, 0.0, 0.0, 0.0);

const bool colortex12Clear = false;
const bool colortex13Clear = false;
const bool colortex14Clear = false;

const bool colortex15Clear = false;

const bool shadowcolor0Clear = true;
const vec4 shadowcolor0ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool shadowcolor1Clear = true;
const vec4 shadowcolor1ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool shadowcolor2Clear = true;
const vec4 shadowcolor2ClearColor = vec4(1.0, 1.0, 1.0, 1.0);

// ------------------------------------------------- Colortex Samplers -------------------------------------------------
uniform sampler2D usam_main;
uniform sampler2D usam_temp1;
uniform sampler2D usam_temp2;
uniform sampler2D usam_temp3;
uniform sampler2D usam_temp4;
uniform sampler2D usam_temp5;
uniform sampler2D usam_temp6;
uniform usampler2D usam_geometryNormal;
uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_translucentColor;
uniform usampler2D usam_tempRGBA32UI;
uniform usampler2D usam_tempRG32UI;
uniform usampler2D usam_packedZN;
uniform sampler2D usam_taaLast;

// -------------------------------------------------- Shadow Samplers --------------------------------------------------
uniform sampler2D shadowtex0;
uniform sampler2DShadow shadowtex0HW;
uniform sampler2D shadowtex1;
uniform sampler2DShadow shadowtex1HW;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;
uniform sampler2D shadowcolor2;

// --------------------------------------------------- Custom Images ---------------------------------------------------
uniform sampler2D usam_rtwsm_imap;
uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_multiSctrLUT;
uniform sampler2D usam_skyLUT;
uniform usampler2D usam_epipolarData;
uniform usampler2D usam_csrgba32ui;
uniform sampler3D usam_cloudsAmbLUT;
uniform usampler2D usam_envProbe;

// -------------------------------------------------- Custom Textures --------------------------------------------------
uniform sampler2D noisetex;
uniform sampler3D usam_blueNoise3D;
uniform sampler3D usam_whiteNoise3D;
uniform sampler3D usam_stbnVec1;
uniform sampler3D usam_stbnUnitVec2;
uniform sampler3D usam_stbnVec2;
uniform sampler2D usam_starmap;
uniform sampler2D usam_constellations;
uniform sampler2D usam_cirrus;
uniform sampler2D usam_cloudPhases;
/*const*/
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

#define usam_overlays colortex6
#define uimg_overlays colorimg6

#define usam_geometryNormal colortex7
#define uimg_geometryNormal colorimg7

#define usam_gbufferData1 colortex8
#define uimg_gbufferData1 colorimg8

#define usam_gbufferData2 colortex9
#define uimg_gbufferData2 colorimg9

#define usam_gbufferViewZ colortex10
#define uimg_gbufferViewZ colorimg10

#define usam_translucentColor colortex11
#define uimg_translucentColor colorimg11

#define usam_translucentData colortex12
#define uimg_translucentData colorimg12

#define usam_tempRG32UI colortex13
#define uimg_tempRG32UI colorimg13

#define usam_packedZN colortex14
#define uimg_packedZN colorimg14

#define usam_taaLast colortex15
#define uimg_taaLast colorimg15

#define usam_shadow_unwarpedUV shadowcolor3
#define uimg_shadow_unwarpedUV shadowcolorimg3

#define usam_shadow_pixelArea shadowcolor4
#define uimg_shadow_pixelArea shadowcolorimg4

#define usam_shadow_waterMask shadowcolor5
#define uimg_shadow_waterMask shadowcolorimg5

#define usam_shadow_waterNormal shadowcolor6
#define uimg_shadow_waterNormal shadowcolorimg6
/*const*/

// ------------------------------------------------- Colortex Samplers -------------------------------------------------
uniform sampler2D usam_main;
uniform sampler2D usam_temp1;
uniform sampler2D usam_temp2;
uniform sampler2D usam_temp3;
uniform sampler2D usam_temp4;
uniform sampler2D usam_temp5;
uniform sampler2D usam_overlays;
uniform usampler2D usam_geometryNormal;
uniform usampler2D usam_gbufferData1;
uniform usampler2D usam_gbufferData2;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_translucentColor;
uniform sampler2D usam_translucentData;
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
uniform sampler2D usam_shadow_unwarpedUV;
uniform sampler2D usam_shadow_pixelArea;
uniform sampler2D usam_shadow_waterMask;
uniform sampler2D usam_shadow_waterNormal;

// --------------------------------------------------- Custom Images ---------------------------------------------------
uniform sampler2D usam_cfrgba16f;
uniform sampler2D usam_csr32f;
uniform sampler2D usam_csrg32f;

uniform usampler2D usam_rgba32ui;
uniform sampler2D usam_rgba16f;
uniform sampler2D usam_rgb10_a2;
uniform sampler2D usam_rgba8;
uniform sampler2D usam_r32f;

uniform sampler2D usam_rtwsm_imap;
uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_multiSctrLUT;
uniform sampler2D usam_skyLUT;
uniform sampler3D usam_skyViewLUT;
uniform usampler2D usam_epipolarData;
uniform sampler3D usam_cloudsAmbLUT;
uniform usampler2D usam_envProbe;

// -------------------------------------------------- Custom Textures --------------------------------------------------
uniform sampler2D noisetex;
uniform sampler2D usam_waveNoise;
uniform sampler2D usam_waveHFCurl;
uniform sampler3D usam_blueNoise3D;
uniform sampler3D usam_whiteNoise3D;
uniform sampler3D usam_stbnVec1;
uniform sampler3D usam_stbnUnitVec2;
uniform sampler3D usam_stbnVec2;
uniform sampler3D usam_stbnUnitVec3Cosine;
uniform sampler2D usam_starmap;
uniform sampler2D usam_constellations;
uniform sampler2D usam_cirrus;
uniform sampler2D usam_cloudPhases;
uniform sampler2D usam_hiz;

uniform sampler3D usam_cumulusDetail1;
uniform sampler3D usam_cumulusCurl;
uniform sampler2D usam_cumulusBase;

/*
const int colortex0Format = RGBA16F; // Main 1
const int colortex1Format = RGBA16F; // Temp1
const int colortex2Format = RGBA16F; // Temp2
const int colortex3Format = RGBA16F; // Temp3
const int colortex4Format = RGBA16F; // Temp4
const int colortex5Format = RGBA8; // Temp5
const int colortex8Format = RGBA32UI; // GBuffer
const int colortex9Format = R32F; // GBuffer ViewZ
const int colortex10Format = RGBA16F; // Translucent Color
const int colortex14Format = RGBA16F; // SSVBIL
const int colortex15Format = RGBA16F; // TAA Last
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

#define usam_gbufferData colortex8
#define uimg_gbufferData colorimg8

#define usam_gbufferViewZ colortex9
#define uimg_gbufferViewZ colorimg9

#define usam_translucentColor colortex10
#define uimg_translucentColor colorimg10

#define usam_ssvbil colortex14
#define uimg_ssvbil colorimg14

#define usam_taaLast colortex15
#define uimg_taaLast colorimg15

const bool shadowcolor0Clear = true;
const vec4 shadowcolor0ClearColor = vec4(1.0, 1.0, 1.0, 1.0);

const bool colortex0Clear = false;

const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;
const bool colortex4Clear = false;

const bool colortex5Clear = false;

// Currently unused
const bool colortex6Clear = false;
const bool colortex7Clear = false;

const bool colortex8Clear = false;
const bool colortex9Clear = false;

const bool colortex10Clear = false;

// Currently unused
const bool colortex11Clear = false;
const bool colortex12Clear = false;
const bool colortex13Clear = false;

const bool colortex14Clear = false;
const bool colortex15Clear = false;
/*
const int colortex0Format = RGBA16F; // Main 1
const int colortex1Format = RGBA16F; // Temp1
const int colortex2Format = RGBA16F; // Temp2
const int colortex3Format = RGBA16F; // Temp3
const int colortex4Format = RGBA16F; // Temp4
const int colortex8Format = RGBA32UI;
const int colortex9Format = R32F; // ViewZ
const int colortex12Format = RG8; // Temporal Project Rejection
const int colortex13Format = RG32UI; // Last Normal ViewZ
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

#define usam_gbuffer colortex8
#define uimg_gbuffer colorimg8

#define usam_viewZ colortex9
#define uimg_viewZ colorimg9

#define usam_projReject colortex12
#define uimg_projReject colorimg12

#define usam_lastNZ colortex13
#define uimg_lastNZ colorimg13

#define usam_ssvbil colortex14
#define uimg_ssvbil colorimg14

#define usam_taaLast colortex15
#define uimg_taaLast colorimg15

const bool shadowcolor0Clear = true;
const vec4 shadowcolor0ClearColor = vec4(1.0, 1.0, 1.0, 1.0);

const bool colortex0Clear = true;
const vec4 colortex0ClearColor = vec4(0.0, 0.0, 0.0, 0.0);

const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;

const bool colortex9Clear = true;
const vec4 colortex9ClearColor = vec4(-65536.0, -65536.0, -65536.0, -65536.0);

const bool colortex11Clear = false;
const bool colortex12Clear = false;
const bool colortex13Clear = false;
const bool colortex14Clear = false;
const bool colortex15Clear = false;
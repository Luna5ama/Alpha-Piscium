const float ambientOcclusionLevel = 0.0;
const float shadowDistanceRenderMul = 1.0;
const float shadowIntervalSize = 0.0;
const bool generateShadowMipmap = true;
const bool shadowHardwareFiltering = true;
const bool shadowtexMipmap = true;
const bool shadowtex0Mipmap = true;
const bool shadowtex1Mipmap = true;

/*
const int colortex0Format = RGBA16F; // Main 1
const int colortex1Format = RGBA16F; // Temp1
const int colortex2Format = RGBA16F; // Temp2
const int colortex3Format = RGBA16F; // Temp3
const int colortex4Format = RGBA8; // Temp4
const int colortex5Format = RGBA8; // Temp5
const int colortex6Format = RGBA8; // Temp6
const int colortex8Format = RGBA32UI; // GBuffer Data 32UI
const int colortex9Format = R32UI; // GBuffer Data 8UN
const int colortex10Format = R32F; // GBuffer ViewZ
const int colortex11Format = RGB10_A2; // Translucent Color
const int colortex12Format = RGBA16F; // Translucent Data

const int shadowcolor0Format = R16F; // Depth offset
const int shadowcolor1Format = RGBA8_SNORM; // Normal
const int shadowcolor2Format = RGBA8; // Translucent color
const int shadowcolor3Format = RG16; // Unwarped UV
const int shadowcolor4Format = R32F; // Pixel area
const int shadowcolor5Format = RGBA8; // Water mask
const int shadowcolor6Format = RGB10_A2; // Water normal
*/

const bool colortex0Clear = false;

const bool colortex1Clear = false;
const vec4 colortex1ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex2Clear = false;
const vec4 colortex2ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool colortex3Clear = false;
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

const bool colortex11Clear = false;
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
const bool shadowcolor3Clear = true;
const vec4 shadowcolor3ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool shadowcolor4Clear = true;
const vec4 shadowcolor4ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool shadowcolor5Clear = true;
const vec4 shadowcolor5ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
const bool shadowcolor6Clear = true;
const vec4 shadowcolor6ClearColor = vec4(0.0, 1.0, 0.0, 0.0);
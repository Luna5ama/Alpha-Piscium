/*
const int colortex0Format = RGBA16F; // main
const int colortex1Format = RGBA16F; // viewcoord
const int colortex2Format = RGBA8; // albedo
const int colortex3Format = RGBA8; // lightMapCoord
const int colortex4Format = RGB10_A2; // normal
const int colortex5Format = R32UI; // material ID

const float sunPathRotation = -45.0;
*/

#define TEX_G_VIEWCOORD colortex1
#define TEX_G_ALBEDO colortex2
#define TEX_G_LIGHTMAPCOORD colortex3
#define TEX_G_NORMAL colortex4
#define TEX_G_MATERIALID colortex5
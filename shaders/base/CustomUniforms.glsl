uniform vec2 uval_rtwsmMin;

uniform vec3 uval_sunDirView;
uniform vec3 uval_sunDirWorld;

uniform vec3 uval_moonDirView;
uniform vec3 uval_moonDirWorld;

uniform vec3 uval_shadowLightDirView;
uniform vec3 uval_shadowLightDirWorld;

uniform vec3 uval_upDirView;

uniform bool uval_sunVisible;
uniform vec2 uval_sunNdcPos;
uniform bool uval_moonVisible;
uniform vec2 uval_moonNdcPos;

uniform float uval_dayNightTransition;

uniform vec3 uval_cuDetailWind;

uniform vec2 uval_mainImageSize;
uniform vec2 uval_mainImageSizeRcp;
uniform int uval_mainImageSizeIX;
uniform int uval_mainImageSizeIY;
ivec2 uval_mainImageSizeI = ivec2(uval_mainImageSizeIX, uval_mainImageSizeIY);

uniform vec3 uval_cameraDelta;
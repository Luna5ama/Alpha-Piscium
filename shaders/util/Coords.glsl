#ifndef INCLUDE_Coords.glsl
#define INCLUDE_Coords.glsl
#include "../_Base.glsl"

float coords_linearizeDepth(float depth, float near, float far) {
    return (near * far) / (depth * (near - far) + far);
}

#endif
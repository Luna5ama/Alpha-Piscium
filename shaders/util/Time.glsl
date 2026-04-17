#ifndef INCLUDE_util_Time_glsl
#define INCLUDE_util_Time_glsl a

#include "Math.glsl"

#define TIME_DAY_TOTAL 24000.0
#define TIME_SUNRISE 23000.0
#define TIME_MORNING 1000.0
#define TIME_NOON 6000.0
#define TIME_AFTERNOON 9000.0
#define TIME_SUNSET 12000.0
#define TIME_NIGHT 15000.0
#define TIME_MIDNIGHT 18000.0
#define TIME_EARLY_MORNING 21000.0

float time_interpolate(float currTime, float startTime, float midTime, float endTime) {
    float m = mod(midTime - startTime, TIME_DAY_TOTAL);
    float e = mod(endTime - startTime, TIME_DAY_TOTAL);
    float c = mod(currTime - startTime, TIME_DAY_TOTAL);
    return linearStep(0.0, m, c) * linearStep(e, m, c);
}

float time_interpolate(float startTime, float midTime, float endTime) {
    return time_interpolate(float(worldTime), startTime, midTime, endTime);
}

#endif
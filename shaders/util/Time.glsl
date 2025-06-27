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
    vec4 timePoints = vec4(currTime, startTime, midTime, endTime);
    float offset = 0.0;
    vec4 offsetTime = timePoints + offset;
    vec4 normalizedTime = fract(offsetTime / TIME_DAY_TOTAL);
    return linearStep(normalizedTime.y, normalizedTime.z, normalizedTime.x) *
    linearStep(normalizedTime.w, normalizedTime.z, normalizedTime.x);
}

float time_interpolate(float startTime, float midTime, float endTime) {
    return time_interpolate(float(worldTime), startTime, midTime, endTime);
}

#endif
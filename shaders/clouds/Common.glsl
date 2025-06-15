/*
    References:
        [HIL16] Hillaire, Sébastien. "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite".
            SIGGRAPH 2016. 2016.
            https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
        [SCH15] Schneider, Andrew. "The Real-Time Volumetric Cloudscapes Of Horizon: Zero Dawn"
            SIGGRAPH 2015. 2015.
            https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_clouds_Common_glsl
#define INCLUDE_clouds_Common_glsl a

#include "Constants.glsl"

struct CloudRaymarchParameters {
    vec3 rayStart;
    vec3 rayDir;
    vec3 rayEnd;
    float rayStartHeight;
    float rayEndHeight;
};

#endif
#ifndef INCLUDE_clouds_Common_glsl
#define INCLUDE_clouds_Common_glsl a

struct CloudRaymarchParameters {
    vec3 rayStart;
    vec3 rayDir;
    vec3 rayEnd;
    float rayStartHeight;
    float rayEndHeight;
};

#endif
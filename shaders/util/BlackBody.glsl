#ifndef INCLUDE_util_BlackBody_glsl
#define INCLUDE_util_BlackBody_glsl a

#include "Colors2.glsl"
#include "Math.glsl"

#define _LUMINOUS_EFFICACY 683.002 // lm/W

// See https://www.desmos.com/calculator/b3jsouh6iv
#define _blackBody_func1_v0 vec3(-37290175.3986, -40810912.9676, -47350510.215)
#define _blackBody_func1_v1 vec3(7055.3581938, 6758.95549072, 4118.25228444)
#define _blackBody_func1_v2 vec3(0.00484537801959, 0.00491276883428, 0.00506185958018)
#define _blackBody_func1_v3 vec3(-6.810769857e-7, -6.950213321e-7, -6.9539565265e-7)
#define _blackBody_func1_v4 vec3(5.6731538598e-11, 5.8002399769e-11, 5.6337103621e-11)
#define _blackBody_func1_v5 vec3(-2.5703246799e-15, -2.6264426188e-15, -2.4816463488e-15)
#define _blackBody_func1_v6 vec3(4.8909354617e-20, 4.9886030965e-20, 4.5966064285e-20)

#define _blackBody_func1(x) exp2( \
_blackBody_func1_v0 * rcp(pow2(x)) + \
_blackBody_func1_v1 * rcp(x) + \
_blackBody_func1_v2 * x + \
_blackBody_func1_v3 * pow2(x) + \
_blackBody_func1_v4 * pow3(x) + \
_blackBody_func1_v5 * pow4(x) + \
_blackBody_func1_v6 * pow5(x) \
)

#define _blackBody_func2_v0 vec3(-8005.4231085, -12668.5089522, 90350.9871613)
#define _blackBody_func2_v1 vec3(-13.8986440037, -13.5414550402, -53.296767586)
#define _blackBody_func2_v2 vec3(0.00322233921192, 0.00326336194057, 0.00762350264707)
#define _blackBody_func2_v3 vec3(-8.7157145477e-8, -8.8899800557e-8, -2.0127905863e-7)
#define _blackBody_func2_v4 vec3(1.2119660239e-12, 1.242334591e-12, 2.7501969304e-12)
#define _blackBody_func2_v5 vec3(-6.7641067868e-18, -6.9583618131e-18, -1.5149377815e-17)

#define _blackBody_func2(x) \
_blackBody_func2_v0 + \
_blackBody_func2_v1 * x + \
_blackBody_func2_v2 * pow2(x) + \
_blackBody_func2_v3 * pow3(x) + \
_blackBody_func2_v4 * pow4(x) + \
_blackBody_func2_v5 * pow5(x)

#define blackBody_evalRadiance_AP0(temperture) (_LUMINOUS_EFFICACY * mix( \
_blackBody_func1(temperture), \
_blackBody_func2(temperture), \
smoothstep(9000.0, 10000.0, float(temperture)) \
))


#endif
#ifndef INCLUDE_util_BlackBody_glsl
#define INCLUDE_util_BlackBody_glsl a

#include "Math.glsl"

#define _LUMINOUS_EFFICACY 683.002 // lm/W

// See https://www.desmos.com/calculator/wgbewiysfy
#define _blackBody_func1_v0 vec3(-35479281.5779, -38132429.4525, -52189491.3761)
#define _blackBody_func1_v1 vec3(5429.82333461, 2120.74345836, 1325.0766228)
#define _blackBody_func1_v2 vec3(0.0043129621406, 0.00438378352015, 0.00437328303002)
#define _blackBody_func1_v3 vec3(-6.3295475786e-7, -6.2109512439e-7, -5.8229422065e-7)
#define _blackBody_func1_v4 vec3(5.4725324266e-11, 5.1889379341e-11, 4.6054996404e-11)
#define _blackBody_func1_v5 vec3(-2.5563195331e-15, -2.3501107568e-15, -1.9888768394e-15)
#define _blackBody_func1_v6 vec3(4.9888152781e-20, 4.4623427601e-20, 3.6225328494e-20)

#define _blackBody_func1(x) exp2( \
_blackBody_func1_v0 * rcp(pow2(x)) + \
_blackBody_func1_v1 * rcp(x) + \
_blackBody_func1_v2 * x + \
_blackBody_func1_v3 * pow2(x) + \
_blackBody_func1_v4 * pow3(x) + \
_blackBody_func1_v5 * pow4(x) + \
_blackBody_func1_v6 * pow5(x) \
)

#define _blackBody_func2_v0 vec3(-9901.81766363, -3748.42608666, 23255.6934998)
#define _blackBody_func2_v1 vec3(0.341167270029, -2.67562462941, -12.5779410385)
#define _blackBody_func2_v2 vec3(0.000330155883666, 0.000686579274145, 0.00172900786342)
#define _blackBody_func2_v3 vec3(-9.2409488779e-9, -1.8795258032e-8, -4.5483910567e-8)
#define _blackBody_func2_v4 vec3(1.3160943449e-13, 2.6354142475e-13, 6.198046843e-13)
#define _blackBody_func2_v5 vec3(-7.4737814432e-19, -1.4796787691e-18, -3.4072528697e-18)

#define _blackBody_func2(x) \
_blackBody_func2_v0 + \
_blackBody_func2_v1 * x + \
_blackBody_func2_v2 * pow2(x) + \
_blackBody_func2_v3 * pow3(x) + \
_blackBody_func2_v4 * pow4(x) + \
_blackBody_func2_v5 * pow5(x)

#define blackBody_evalRadiance(temperture) _LUMINOUS_EFFICACY * mix( \
_blackBody_func1(temperture), \
_blackBody_func2(temperture), \
smoothstep(9000.0, 10000.0, float(temperture)) \
)

#endif
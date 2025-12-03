/*
    References:
        [GEI05] GEISA. "The Database and Associated Software OPAC". 2005. https://cds-espri.ipsl.upmc.fr/etherTypo/?id=989&L=0
*/

#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/PhaseFunc.glsl"

// Stratus (continental)
const vec3 CLOUDS_ST_SCATTERING = colors2_constants_toWorkSpace(vec3(0.23777277562949245, 0.23987629843161026, 0.2551375513806795));
const vec3 CLOUDS_ST_EXTINCTION = colors2_constants_toWorkSpace(vec3(0.23777304241545993, 0.2398763953030018, 0.25513757827543126));
const vec3 CLOUDS_ST_ASYM = colors2_constants_toWorkSpace(vec3(0.8629828250733059, 0.8727334366957631, 0.9334765415706695));
const vec3 CLOUDS_ST_E = colors2_constants_toWorkSpace(vec3(1091938.4155551041, 3341273.320789842, 5.699703416476822e12));
const float CLOUDS_ST_R_EFF = 7.33;

// Cumulus (cont., clean)
const vec3 CLOUDS_CU_SCATTERING = colors2_constants_toWorkSpace(vec3(0.18067349140471628, 0.18215551958414714, 0.19358579492341665));
const vec3 CLOUDS_CU_EXTINCTION = colors2_constants_toWorkSpace(vec3(0.18067367051236774, 0.18215559145291857, 0.19358580629452912));
const vec3 CLOUDS_CU_ASYM = colors2_constants_toWorkSpace(vec3(0.8615159687912013, 0.8732937077048064, 0.9375708300315341));
const vec3 CLOUDS_CU_E = colors2_constants_toWorkSpace(vec3(935502.5670106385, 3581719.839469918, 4.094123822571251E13));
const float CLOUDS_CU_R_EFF = 5.77;

// Cirrus 1: -25° C
const vec3 CLOUDS_CI_SCATTERING = colors2_constants_toWorkSpace(vec3(6.17017026101243, 6.239057230606717, 6.667813761472693));
const vec3 CLOUDS_CI_EXTINCTION = colors2_constants_toWorkSpace(vec3(6.170170260942012, 6.239057230606713, 6.667813761472693));
const vec3 CLOUDS_CI_ASYM = colors2_constants_toWorkSpace(vec3(0.7816707299139416, 0.7895764808723628, 0.8408420552232578));
const vec3 CLOUDS_CI_E = colors2_constants_toWorkSpace(vec3(4714.18358593152, 6665.142983584232, 143262.2082694281));
const float CLOUDS_CI_R_EFF = 91.7;

struct CloudParticpatingMedium {
    vec3 scattering;
    vec3 extinction;
    vec3 phase;
};

// See https://www.desmos.com/calculator/yerfmyqpuh
vec3 _clouds_samplePhaseLUT(float cosTheta, float type) {
    const float a0 = 0.672617934627;
    const float a1 = -0.0713555761181;
    const float a2 = 0.0299320735609;
    const float b = 0.264767018876;
    float x1 = acos(cosTheta);
    float x2 = x1 * x1;
    float u = saturate((a0 + a1 * x1 + a2 * x2) * pow(x1, b));
    float v = (type + 0.5) / 3.0;
    // It was encoded from AP0
    vec4 rgbm = texture(usam_cloudPhases, vec2(u, v));
    return colors2_constants_toWorkSpace(rgbm.rgb * rgbm.a);
}

vec3 _clouds_cirrusLUTPhase(float cosTheta) {
    return _clouds_samplePhaseLUT(cosTheta, 0.0);
}

vec3 _clouds_cumulusLUTPhase(float cosTheta) {
    return _clouds_samplePhaseLUT(cosTheta, 1.0);
}

vec3 _clouds_stratusLUTPhase(float cosTheta) {
    return _clouds_samplePhaseLUT(cosTheta, 2.0);
}

vec3 clouds_phase_cu(float cosTheta, float mixRatio) {
    return mix(vec3(phasefunc_HenyeyGreensteinDraine(cosTheta, CLOUDS_CU_R_EFF * 2.0)), _clouds_cumulusLUTPhase(cosTheta), mixRatio);
}

vec3 clouds_phase_ci(float cosTheta, float mixRatio) {
    // Not using Nishina phase to fill the gap between center and 22 degree halo
    return mix(phasefunc_HenyeyGreenstein(cosTheta, CLOUDS_CI_ASYM), _clouds_cirrusLUTPhase(cosTheta), mixRatio);
}

CloudParticpatingMedium clouds_cu_medium(float cosTheta) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CU_SCATTERING;
    medium.extinction = CLOUDS_CU_EXTINCTION;
    medium.phase = clouds_phase_cu(cosTheta, SETTING_CLOUDS_CU_PHASE_RATIO);
    return medium;
}

CloudParticpatingMedium clouds_ci_medium(float cosTheta) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CI_SCATTERING;
    medium.extinction = CLOUDS_CI_EXTINCTION;
    medium.phase = clouds_phase_ci(cosTheta, SETTING_CLOUDS_CI_PHASE_RATIO);
    return medium;
}
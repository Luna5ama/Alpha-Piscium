/*
    References:
        [GEI05] GEISA. "The Database and Associated Software OPAC". 2005. https://cds-espri.ipsl.upmc.fr/etherTypo/?id=989&L=0
*/

#include "/util/Colors.glsl"
#include "/util/PhaseFunc.glsl"

// Stratus (continental)
const vec3 CLOUDS_ST_SCATTERING = vec3(0.23977311964607095, 0.23773623658142246, 0.23611086956416943);
const vec3 CLOUDS_ST_EXTINCTION = vec3(0.23977368314514727, 0.2377362922769532, 0.23611087968268715);
const vec3 CLOUDS_ST_ASYM = vec3(0.8649476668256066, 0.8650419167426422, 0.8647112740133283);
const float CLOUDS_ST_R_EFF = 7.33;

// Cumulus (cont., clean)
const vec3 CLOUDS_CU_SCATTERING = vec3(0.18243438230318823, 0.18051322139189785, 0.17912261606289154);
const vec3 CLOUDS_CU_EXTINCTION = vec3(0.1824347555801482, 0.1805132687424516, 0.17912261461420664);
const vec3 CLOUDS_CU_ASYM = vec3(0.8589638232616821, 0.8658317471227941, 0.8690792504193543);
const float CLOUDS_CU_R_EFF = 5.77;

// Cirrus 1: -25° C
const vec3 CLOUDS_CI_SCATTERING = vec3(6.187642992926912, 6.184357289476479, 6.175761065554168);
const vec3 CLOUDS_CI_EXTINCTION = vec3(6.187642992746775, 6.184357289496035, 6.175761065555373);
const vec3 CLOUDS_CI_ASYM = vec3(0.7863210552363993, 0.7827084906574567, 0.7783142662154189);
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
    float x1 = acos(-cosTheta);
    float x2 = x1 * x1;
    float u = saturate((a0 + a1 * x1 + a2 * x2) * pow(x1, b));
    float v = (type + 0.5) / 3.0;
    return colors_LogLuv32ToSRGB(texture(usam_cloudPhases, vec2(u, v)));
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
    return mix(vec3(hgDrainePhase(cosTheta, CLOUDS_CU_R_EFF * 2.0)), _clouds_cumulusLUTPhase(cosTheta), mixRatio);
}

vec3 clouds_phase_ci(float cosTheta, float mixRatio) {
    return mix(cornetteShanksPhase(cosTheta, CLOUDS_CI_ASYM), _clouds_cirrusLUTPhase(cosTheta), mixRatio);
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
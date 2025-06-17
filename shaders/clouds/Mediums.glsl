/*
    References:
        [GEI05] GEISA. "The Database and Associated Software OPAC". 2005. https://cds-espri.ipsl.upmc.fr/etherTypo/?id=989&L=0
*/

// See https://www.desmos.com/calculator/m39bzkcevu
// Stratus (continental)
const vec3 CLOUDS_ST_SCATTERING = vec3(0.239422239197, 0.237947659001, 0.236198745092);
const vec3 CLOUDS_ST_EXTINCTION = vec3(0.239422523318, 0.237947741995, 0.236198789682);
const vec3 CLOUDS_ST_ASYM = vec3(0.864613854108, 0.865550699239, 0.865188695119);
const float CLOUDS_ST_R_EFF = 7.33;

// Cumulus (cont., clean)
const vec3 CLOUDS_CU_SCATTERING = vec3(0.18214483489, 0.180835685649, 0.179274244451);
const vec3 CLOUDS_CU_EXTINCTION = vec3(0.182145119011, 0.180835768642, 0.179274289041);
const vec3 CLOUDS_CU_ASYM = vec3(0.861247898372, 0.865190327068, 0.865986164932);
const float CLOUDS_CU_R_EFF = 5.77;

// Cirrus 1: -25° C
const vec3 CLOUDS_CI_SCATTERING = vec3(6.18410168687, 6.18700464438, 6.19390902499);
const vec3 CLOUDS_CI_EXTINCTION = vec3(6.18410168687, 6.18700464438, 6.19390902499);
const vec3 CLOUDS_CI_ASYM = vec3(0.785425549428, 0.783514530696, 0.779705038777);
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

vec3 clouds_phase_cu(float cosTheta) {
    return mix(vec3(hgDrainePhase(cosTheta, CLOUDS_CU_R_EFF * 2.0)), _clouds_cumulusLUTPhase(cosTheta), SETTING_CLOUDS_CU_PHASE_RATIO);
}

vec3 clouds_phase_ci(float cosTheta) {
    return mix(cornetteShanksPhase(cosTheta, CLOUDS_CI_ASYM), _clouds_cirrusLUTPhase(cosTheta), SETTING_CLOUDS_CI_PHASE_RATIO);
}

CloudParticpatingMedium clouds_cu_medium(float cosTheta) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CU_SCATTERING;
    medium.extinction = CLOUDS_CU_EXTINCTION;
    medium.phase = clouds_phase_cu(cosTheta);
    return medium;
}

CloudParticpatingMedium clouds_ci_medium(float cosTheta) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CI_SCATTERING;
    medium.extinction = CLOUDS_CI_EXTINCTION;
    medium.phase = clouds_phase_ci(cosTheta);
    return medium;
}
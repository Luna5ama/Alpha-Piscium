#extension GL_KHR_shader_subgroup_ballot : enable

#define GLOBAL_DATA_MODIFIER
#include "../_Util.glsl"

layout(local_size_x = 32, local_size_y = 32) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

shared uint shared_topBinCount = 0u;

layout(rgba16f) restrict uniform image2D uimg_main;
layout(rgba16f) readonly uniform image2D uimg_temp1;
layout(rgba16f) writeonly uniform image2D uimg_temp2;

vec3 uchimura(vec3 x, float P, float a, float m, float l, float c, float b) {
    // Uchimura 2017, "HDR theory and practice"
    // Math: https://www.desmos.com/calculator/gslcdxvipg
    // Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    vec3 w0 = 1.0 - smoothstep(0.0, m, x);
    vec3 w2 = step(m + l0, x);
    vec3 w1 = 1.0 - w0 - w2;

    vec3 T = m * pow(x / m, vec3(c)) + b;
    vec3 S = P - (P - S1) * exp(CP * (x - S0));
    vec3 L = m + a * (x - m);

    return T * w0 + L * w1 + S * w2;
}

vec3 uchimura(vec3 x) {
    const float P = 1.0;// max display brightness
    const float a = 1.05;// contrast
    const float m = 0.2;// linear section start
    const float l = 0.4;// linear section length
    const float c = 1.35;// black
    const float b = 0.0;// pedestal
    return uchimura(x, P, a, m, l, c, b);
}

uint histoIndex(float x) {
    const float EPSILON = 0.001;
    uint binIndex;
    if (x < EPSILON) {
        binIndex = 0;
    } else {
        float lumMapped = clamp(log2(x) + 1.0, 0.0, 1.0);
        binIndex = uint(lumMapped * 254.0 + 1.0);
    }
    return binIndex;
}

void main() {
    ivec2 imgSize = imageSize(uimg_main);
    ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);


    if (all(lessThan(pixelPos, imgSize))) {
        vec4 color = imageLoad(uimg_main, pixelPos);
//        color.rgb *= global_exposure;
        color.rgb *= 0.001;
        float lumanance = colors_linearSRGBToLuminance(max(color.rgb, 0.0));

        uint binIndex = histoIndex(lumanance);
        uvec4 topBinBallot = subgroupBallot(binIndex == 255u);
        if (subgroupElect()) {
            uint topBinCount = subgroupBallotBitCount(topBinBallot);
            atomicAdd(shared_topBinCount, topBinCount);
        }

        color.rgb = uchimura(color.rgb);
        color.rgb = pow(color.rgb, vec3(1.0 / SETTING_OUTPUT_GAMMA));
        imageStore(uimg_main, pixelPos, color);

        barrier();

        if (gl_LocalInvocationID.x == 0) {
            atomicAdd(global_topBinCount, shared_topBinCount);
        }
    }
}
#include "/util/BitPacking.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"
#include "/util/GBuffers.glsl"
#include "/util/Dither.glsl"
uniform sampler2D usam_temp6;

ivec2 denoiser_getImageSize();
void denoiser_input(ivec2 coord, out vec4 data, out vec3 normal, out float viewZ);
void denoiser_output(ivec2 coord, vec4 data);

#define WORKGROUP_SIZE 128

#if defined(DENOISER_HORIZONTAL)
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
#elif defined(DENOISER_VERTICAL)
layout(local_size_x = 1, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;
#else
#error "Either DENOISER_HORIZONTAL or DENOISER_VERTICAL must be defined"
#endif

#ifdef DATA_32
shared vec4 shared_data_32[WORKGROUP_SIZE + DENOISER_KERNEL_RADIUS * 2];

vec4 readSharedData(uint idx) {
    return shared_data_32[idx];
}

void writeSharedData(uint idx, vec4 value) {
    shared_data_32[idx] = value;
}
#else
shared uvec2 shared_data_16[WORKGROUP_SIZE + DENOISER_KERNEL_RADIUS * 2];

vec4 readSharedData(uint idx) {
    uvec2 data = shared_data_16[idx];
    return vec4(unpackHalf2x16(data.x), unpackHalf2x16(data.y));
}

void writeSharedData(uint idx, vec4 value) {
    shared_data_16[idx] = uvec2(packHalf2x16(value.xy), packHalf2x16(value.zw));
}
#endif

shared uvec4 shared_posNormal[WORKGROUP_SIZE + DENOISER_KERNEL_RADIUS * 2];

#if DENOISER_KERNEL_RADIUS == 64
const float WEIGHTS[] = { 0.07038609217001514, 0.06930322921355336, 0.06615308243111911, 0.061216285234766944, 0.05491460881354093, 0.04775183375090516, 0.040247974161477205, 0.03287862677979828, 0.026028912867340305, 0.019967385213302154, 0.01484062414502187, 0.010685249384415747, 0.007451555491763613, 0.0050322192931390635, 0.0032902972301293875, 0.0020824666013477136, 0.0012755107933254746, 7.558582478965775E-4, 4.33235825013892E-4, 2.401066018149281E-4, 1.2862853668656863E-4, 6.658418369657669E-5, 3.3292091848288346E-5, 1.6072044340552996E-5, 7.488111567757646E-6, 3.3654434012393916E-6, 1.4583588072037363E-6, 6.08984996414747E-7, 2.449178789928874E-7, 9.480692090047253E-8, 3.530044927145254E-8, 1.2633845002414594E-8, 4.342884219580017E-9, 1.4327040724387683E-9, 4.5320230862858996E-10, 1.3733403291775454E-10, 3.982686954614882E-11, 1.1041112349427394E-11, 2.922647386613134E-12, 7.377556509897231E-13, 1.7734510841099112E-13, 4.053602477965512E-14, 8.79555254652894E-15, 1.8084313647068848E-15, 3.5163943202633876E-16, 6.45209967020805E-17, 1.114453579399572E-17, 1.80722202064796E-18, 2.7431048527692E-19, 3.884042269408E-20, 5.11058193343E-21, 6.2215780059E-22, 6.97245811E-23, 7.15123909E-24, 6.6664093E-25, 5.602025E-26, 4.20152E-27, 2.7779E-28, 1.594E-29, 7.8E-31, 3.0E-32, 0.0, 0.0, 0.0, 0.0 };
#elif DENOISER_KERNEL_RADIUS == 32
const float WEIGHTS[] = { 0.09934675374796689, 0.09633624605863457, 0.08783598905346093, 0.07528799061725222, 0.06064865910834207, 0.04589628256847508, 0.03261051656181124, 0.021740344374540827, 0.013587715234088017, 0.007953784527271034, 0.004355643907791281, 0.00222846897607926, 0.0010635874658560104, 4.7270554038044907E-4, 1.9524794059192462E-4, 7.477580703520517E-5, 2.64830983249685E-5, 8.647542310193795E-6, 2.5942626930581386E-6, 7.121505431924302E-7, 1.7803763579810755E-7, 4.03104081052319E-8, 8.211379428843535E-9, 1.4929780779715518E-9, 2.399429053882851E-10, 3.367619724747861E-11, 4.064368633316384E-12, 4.133256237270899E-13, 3.4443801977257493E-14, 2.258609965721803E-15, 1.0928757898653885E-16, 3.46944695195361E-18, 5.421010862428E-20 };
#elif DENOISER_KERNEL_RADIUS == 16
const float WEIGHTS[] = { 0.10148350230345303, 0.10745312008600909, 0.10148350230345303, 0.08545979141343413, 0.0640948435600756, 0.042729895706717064, 0.025249483826696447, 0.013173643735667713, 0.006037920045514368, 0.0024151680182057473, 8.360196986096817E-4, 2.477095403287946E-4, 6.192738508219865E-5, 1.281256243079972E-5, 2.1354270717999536E-6, 2.755389770064456E-7, 2.5831779094354274E-8 };
#elif DENOISER_KERNEL_RADIUS == 8
const float WEIGHTS[] = { 0.1235480464625132, 0.13727560718057022, 0.1235480464625132, 0.0898531247000096, 0.052414322741672265, 0.024191225880771817, 0.008639723528847077, 0.0023039262743592206, 4.3198617644235387E-4 };
#elif DENOISER_KERNEL_RADIUS == 4
const float WEIGHTS[] = { 0.15283842794759825, 0.18340611353711792, 0.15283842794759825, 0.08733624454148471, 0.03275109170305677 };
#elif DENOISER_KERNEL_RADIUS == 2
const float WEIGHTS[] = { 0.42857142857142855, 0.2857142857142857, 0.07142857142857142 };
#else
#error "Unsupported kernel radius"
#endif

ivec2 imgSize = denoiser_getImageSize();

void readSharedPosNormal(uint index, out vec3 normal, out vec3 pos) {
    uvec4 data = shared_posNormal[index];
    pos = uintBitsToFloat(data.xyz);
    normal.x = unpackS10(bitfieldExtract(data.w, 0, 10));
    normal.y = unpackS10(bitfieldExtract(data.w, 10, 10));
    normal.z = unpackS10(bitfieldExtract(data.w, 20, 10));
}

void writeSharedPosNormal(uint index, vec3 normal, vec3 pos) {
    uvec4 data;
    data.xyz = floatBitsToUint(pos);
    data.w = bitfieldInsert(data.w, packS10(normal.x), 0, 10);
    data.w = bitfieldInsert(data.w, packS10(normal.y), 10, 10);
    data.w = bitfieldInsert(data.w, packS10(normal.z), 20, 10);
    shared_posNormal[index] = data;
}

void readData(uint writeIndex, ivec2 coord, out vec3 normal, out vec3 pos) {
    coord = clamp(coord, ivec2(0), imgSize - 1);
    float viewZ;
    vec4 data;
    denoiser_input(coord, data, normal, viewZ);

    writeSharedData(writeIndex, data);

    vec2 texCoord = (vec2(coord) + 0.5) / vec2(imgSize);
    pos = coords_toViewCoord(texCoord, viewZ, gbufferProjectionInverse);

    writeSharedPosNormal(writeIndex, normal, pos);
}

void readData(uint writeIndex, ivec2 coord) {
    coord = clamp(coord, ivec2(0), imgSize - 1);
    float viewZ;
    vec3 normal;
    vec4 data;
    denoiser_input(coord, data, normal, viewZ);

    writeSharedData(writeIndex, data);

    vec2 texCoord = (vec2(coord) + 0.5) / vec2(imgSize);
    vec3 pos = coords_toViewCoord(texCoord, viewZ, gbufferProjectionInverse);

    writeSharedPosNormal(writeIndex, normal, pos);
}

#if defined(DENOISER_HORIZONTAL)
void setup(out ivec2 icoord, out vec3 normalC, out vec3 posC) {
    int intLocalIDX = int(gl_LocalInvocationID.x);
    icoord = ivec2(gl_GlobalInvocationID.xy);

    readData(gl_LocalInvocationIndex + DENOISER_KERNEL_RADIUS, icoord, normalC, posC);
    if (gl_LocalInvocationIndex < DENOISER_KERNEL_RADIUS * 2u) {
        if (gl_LocalInvocationIndex < DENOISER_KERNEL_RADIUS) {
            ivec2 coord2 = icoord;
            coord2.x -= DENOISER_KERNEL_RADIUS;
            readData(gl_LocalInvocationIndex, coord2);
        } else {
            ivec2 coord2 = icoord;
            coord2.x += int(WORKGROUP_SIZE) - DENOISER_KERNEL_RADIUS;
            readData(gl_LocalInvocationIndex + WORKGROUP_SIZE, coord2);
        }
    }
}
#elif defined(DENOISER_VERTICAL)
void setup(out ivec2 icoord, out vec3 normalC, out vec3 posC) {
    int intLocalIDX = int(gl_LocalInvocationIndex);
    icoord = ivec2(gl_GlobalInvocationID.xy);

    readData(gl_LocalInvocationIndex + DENOISER_KERNEL_RADIUS, icoord, normalC, posC);
    if (gl_LocalInvocationIndex < DENOISER_KERNEL_RADIUS * 2u) {
        if (gl_LocalInvocationIndex < DENOISER_KERNEL_RADIUS) {
            ivec2 coord2 = icoord;
            coord2.y -= DENOISER_KERNEL_RADIUS;
            readData(gl_LocalInvocationIndex, coord2);
        } else {
            ivec2 coord2 = icoord;
            coord2.y += int(WORKGROUP_SIZE) - DENOISER_KERNEL_RADIUS;
            readData(gl_LocalInvocationIndex + int(WORKGROUP_SIZE), coord2);
        }
    }
}
#else
#error "Either DENOISER_HORIZONTAL or DENOISER_VERTICAL must be defined"
#endif

#if defined(DENOISER_BOX)
#define baseWeight(i) 1.0
#elif defined(DENOISER_GAUSSIAN)
#define baseWeight(i) WEIGHTS[i]
#else
#error "Either DENOISER_BOX or DENOISER_GAUSSIAN must be defined"
#endif

float hlenC;
const float BASE_LUMA_ALPHA = 1.0 / SETTING_DENOISER_FILTER_COLOR_STRICTNESS;

float computeWeight(vec3 normalC, vec3 posC, float lumaC, vec3 normal, vec3 pos, float luma) {
    vec3 diff = pos - posC;
    float posA = -posC.z * 0.1;
    float distSq = dot(diff, diff);

    float normalDot = dot(normalC, normal);
    float diffNormalDot = dot(normalize(normalC), normalize(diff));
    float lumaDiff = luma - lumaC;

    float weight = 1.0;
    weight *= pow(saturate(normalDot * normalDot), SETTING_DENOISER_FILTER_NORMAL_STRICTNESS);
    weight *= pow(saturate(1.0 - diffNormalDot * diffNormalDot), SETTING_DENOISER_FILTER_DEPTH_STRICTNESS);

    float la = BASE_LUMA_ALPHA / hlenC;
    weight *= max(la / saturate(la + lumaDiff * lumaDiff), BASE_LUMA_ALPHA);

    return weight;
}

void main() {
    ivec2 icoord;
    vec3 normalC;
    vec3 posC;
    setup(icoord, normalC, posC);

    if (any(greaterThanEqual(icoord, imgSize))) {
        return;
    }

    barrier();

    hlenC = texelFetch(usam_temp6, icoord, 0).r * 255.0 + 1.0;

    vec4 sum = vec4(0.0);
    uint centerIdx = gl_LocalInvocationIndex + DENOISER_KERNEL_RADIUS;

    float weight0 = baseWeight(0);
    vec4 colorC = readSharedData(centerIdx);
    sum += colorC * weight0;
    float weightSum = weight0;

    for (uint i = 1; i <= DENOISER_KERNEL_RADIUS; i++) {
        vec3 pos;
        vec3 normal;
        readSharedPosNormal(centerIdx - i, normal, pos);
        vec4 color = readSharedData(centerIdx - i);
        float weight = baseWeight(i) * computeWeight(normalC, posC, colorC.a, normal, pos, color.a);
        sum += color * weight;
        weightSum += weight;
    }
    for (uint i = 1; i <= DENOISER_KERNEL_RADIUS; i++) {
        vec3 pos;
        vec3 normal;
        readSharedPosNormal(centerIdx + i, normal, pos);
        vec4 color = readSharedData(centerIdx + i);
        float weight = baseWeight(i) * computeWeight(normalC, posC, colorC.a, normal, pos, color.a);
        sum += color * weight;
        weightSum += weight;
    }

    sum /= weightSum;
    sum = dither_fp16(sum, rand_IGN(icoord, frameCounter + 1));

    denoiser_output(icoord, sum);
}
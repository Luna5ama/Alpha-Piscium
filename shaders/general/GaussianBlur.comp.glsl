layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

#ifndef GAUSSIAN_BLUR_INPUT
#error "GAUSSIAN_BLUR_INPUT must be defined"
#endif

#ifndef GAUSSIAN_BLUR_OUTPUT
#error "GAUSSIAN_BLUR_OUTPUT must be defined"
#endif

#ifndef GAUSSIAN_BLUR_CHANNELS
#error "GAUSSIAN_BLUR_CHANNELS must be defined"
#endif

#ifndef GAUSSIAN_BLUR_KERNEL_RADIUS
#error "GAUSSIAN_BLUR_KERNEL_RADIUS must be defined"
#endif

#if GAUSSIAN_BLUR_CHANNELS == 4
#define READ_INPUT(coord) imageLoad(GAUSSIAN_BLUR_INPUT, coord)
#define DATA_TYPE vec4
#elif GAUSSIAN_BLUR_CHANNELS == 3
#define READ_INPUT(coord) imageLoad(GAUSSIAN_BLUR_INPUT, coord).rgb
#define DATA_TYPE vec3
#elif GAUSSIAN_BLUR_CHANNELS == 2
#define READ_INPUT(coord) imageLoad(GAUSSIAN_BLUR_INPUT, coord).rg
#define DATA_TYPE vec2
#elif GAUSSIAN_BLUR_CHANNELS == 1
#define READ_INPUT(coord) imageLoad(GAUSSIAN_BLUR_INPUT, coord).r
#define DATA_TYPE float
#else
#error "Unsupported channel count"
#endif

#if GAUSSIAN_BLUR_KERNEL_RADIUS == 64
const float WEIGHTS[] = { 0.07038609217001514, 0.06930322921355336, 0.06615308243111911, 0.061216285234766944, 0.05491460881354093, 0.04775183375090516, 0.040247974161477205, 0.03287862677979828, 0.026028912867340305, 0.019967385213302154, 0.01484062414502187, 0.010685249384415747, 0.007451555491763613, 0.0050322192931390635, 0.0032902972301293875, 0.0020824666013477136, 0.0012755107933254746, 7.558582478965775E-4, 4.33235825013892E-4, 2.401066018149281E-4, 1.2862853668656863E-4, 6.658418369657669E-5, 3.3292091848288346E-5, 1.6072044340552996E-5, 7.488111567757646E-6, 3.3654434012393916E-6, 1.4583588072037363E-6, 6.08984996414747E-7, 2.449178789928874E-7, 9.480692090047253E-8, 3.530044927145254E-8, 1.2633845002414594E-8, 4.342884219580017E-9, 1.4327040724387683E-9, 4.5320230862858996E-10, 1.3733403291775454E-10, 3.982686954614882E-11, 1.1041112349427394E-11, 2.922647386613134E-12, 7.377556509897231E-13, 1.7734510841099112E-13, 4.053602477965512E-14, 8.79555254652894E-15, 1.8084313647068848E-15, 3.5163943202633876E-16, 6.45209967020805E-17, 1.114453579399572E-17, 1.80722202064796E-18, 2.7431048527692E-19, 3.884042269408E-20, 5.11058193343E-21, 6.2215780059E-22, 6.97245811E-23, 7.15123909E-24, 6.6664093E-25, 5.602025E-26, 4.20152E-27, 2.7779E-28, 1.594E-29, 7.8E-31, 3.0E-32, 0.0, 0.0, 0.0, 0.0 };
#elif GAUSSIAN_BLUR_KERNEL_RADIUS == 32
const float WEIGHTS[] = { 0.09934675374796689, 0.09633624605863457, 0.08783598905346093, 0.07528799061725222, 0.06064865910834207, 0.04589628256847508, 0.03261051656181124, 0.021740344374540827, 0.013587715234088017, 0.007953784527271034, 0.004355643907791281, 0.00222846897607926, 0.0010635874658560104, 4.7270554038044907E-4, 1.9524794059192462E-4, 7.477580703520517E-5, 2.64830983249685E-5, 8.647542310193795E-6, 2.5942626930581386E-6, 7.121505431924302E-7, 1.7803763579810755E-7, 4.03104081052319E-8, 8.211379428843535E-9, 1.4929780779715518E-9, 2.399429053882851E-10, 3.367619724747861E-11, 4.064368633316384E-12, 4.133256237270899E-13, 3.4443801977257493E-14, 2.258609965721803E-15, 1.0928757898653885E-16, 3.46944695195361E-18, 5.421010862428E-20 };
#elif GAUSSIAN_BLUR_KERNEL_RADIUS == 16
const float WEIGHTS[] = { 0.10148350230345303, 0.10745312008600909, 0.10148350230345303, 0.08545979141343413, 0.0640948435600756, 0.042729895706717064, 0.025249483826696447, 0.013173643735667713, 0.006037920045514368, 0.0024151680182057473, 8.360196986096817E-4, 2.477095403287946E-4, 6.192738508219865E-5, 1.281256243079972E-5, 2.1354270717999536E-6, 2.755389770064456E-7, 2.5831779094354274E-8 };
#elif GAUSSIAN_BLUR_KERNEL_RADIUS == 8
const float WEIGHTS[] = { 0.1235480464625132, 0.13727560718057022, 0.1235480464625132, 0.0898531247000096, 0.052414322741672265, 0.024191225880771817, 0.008639723528847077, 0.0023039262743592206, 4.3198617644235387E-4 };
#elif GAUSSIAN_BLUR_KERNEL_RADIUS == 4
const float WEIGHTS[] = { 0.15283842794759825, 0.18340611353711792, 0.15283842794759825, 0.08733624454148471, 0.03275109170305677 };
#elif GAUSSIAN_BLUR_KERNEL_RADIUS == 2
const float WEIGHTS[] = { 0.42857142857142855, 0.2857142857142857, 0.07142857142857142 };
#else
#error "Unsupported kernel radius"
#endif

ivec2 imgSize = imageSize(GAUSSIAN_BLUR_OUTPUT);
shared DATA_TYPE dataCache[gl_WorkGroupSize.x + GAUSSIAN_BLUR_KERNEL_RADIUS * 2];

void sampleH(out ivec2 icoord) {
    int intLocalIDX = int(gl_LocalInvocationID.x);
    icoord = ivec2(gl_GlobalInvocationID.xy);

    dataCache[gl_LocalInvocationID.x + GAUSSIAN_BLUR_KERNEL_RADIUS] = READ_INPUT(icoord);
    if (gl_LocalInvocationID.x < GAUSSIAN_BLUR_KERNEL_RADIUS * 2u) {
        if (gl_LocalInvocationID.x < GAUSSIAN_BLUR_KERNEL_RADIUS) {
            ivec2 coord2 = icoord;
            coord2.x -= GAUSSIAN_BLUR_KERNEL_RADIUS;
            coord2 = max(coord2, ivec2(0));
            dataCache[gl_LocalInvocationID.x] = READ_INPUT(coord2);
        } else {
            ivec2 coord2 = icoord;
            coord2.x += int(gl_WorkGroupSize.x) - GAUSSIAN_BLUR_KERNEL_RADIUS;
            coord2 = min(coord2, imgSize.x - 1);
            dataCache[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = READ_INPUT(coord2);
        }
    }
}

void sampleV(out ivec2 icoord) {
    int intLocalIDX = int(gl_LocalInvocationID.x);
    icoord = ivec2(gl_WorkGroupID.yx);
    icoord.y = int(icoord.y) * int(gl_WorkGroupSize.x) + intLocalIDX;

    dataCache[gl_LocalInvocationID.x + GAUSSIAN_BLUR_KERNEL_RADIUS] = READ_INPUT(icoord);
    if (gl_LocalInvocationID.x < GAUSSIAN_BLUR_KERNEL_RADIUS * 2u) {
        if (gl_LocalInvocationID.x < GAUSSIAN_BLUR_KERNEL_RADIUS) {
            ivec2 coord2 = icoord;
            coord2.y -= GAUSSIAN_BLUR_KERNEL_RADIUS;
            coord2 = max(coord2, ivec2(0));
            dataCache[gl_LocalInvocationID.x] = READ_INPUT(coord2);
        } else {
            ivec2 coord2 = icoord;
            coord2.y += int(gl_WorkGroupSize.x) - GAUSSIAN_BLUR_KERNEL_RADIUS;
            coord2 = min(coord2, imgSize.y - 1);
            dataCache[gl_LocalInvocationID.x + int(gl_WorkGroupSize.x)] = READ_INPUT(coord2);
        }
    }
}

void main() {
    ivec2 icoord;
    #if defined(GAUSSIAN_BLUR_VERTICAL)
    sampleV(icoord);
    #elif defined(GAUSSIAN_BLUR_HORIZONTAL)
    sampleH(icoord);
    #else
    #error "Either GAUSSIAN_BLUR_VERTICAL or GAUSSIAN_BLUR_HORIZONTAL must be defined"
    #endif

    if (any(greaterThanEqual(icoord, imgSize))) {
        return;
    }

    barrier();

    DATA_TYPE sum = DATA_TYPE(0.0);
    uint centerIdx = gl_LocalInvocationID.x + GAUSSIAN_BLUR_KERNEL_RADIUS;
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 64
    sum += dataCache[centerIdx - 64] * WEIGHTS[64];
    sum += dataCache[centerIdx - 63] * WEIGHTS[63];
    sum += dataCache[centerIdx - 62] * WEIGHTS[62];
    sum += dataCache[centerIdx - 61] * WEIGHTS[61];
    sum += dataCache[centerIdx - 60] * WEIGHTS[60];
    sum += dataCache[centerIdx - 59] * WEIGHTS[59];
    sum += dataCache[centerIdx - 58] * WEIGHTS[58];
    sum += dataCache[centerIdx - 57] * WEIGHTS[57];
    sum += dataCache[centerIdx - 56] * WEIGHTS[56];
    sum += dataCache[centerIdx - 55] * WEIGHTS[55];
    sum += dataCache[centerIdx - 54] * WEIGHTS[54];
    sum += dataCache[centerIdx - 53] * WEIGHTS[53];
    sum += dataCache[centerIdx - 52] * WEIGHTS[52];
    sum += dataCache[centerIdx - 51] * WEIGHTS[51];
    sum += dataCache[centerIdx - 50] * WEIGHTS[50];
    sum += dataCache[centerIdx - 49] * WEIGHTS[49];
    sum += dataCache[centerIdx - 48] * WEIGHTS[48];
    sum += dataCache[centerIdx - 47] * WEIGHTS[47];
    sum += dataCache[centerIdx - 46] * WEIGHTS[46];
    sum += dataCache[centerIdx - 45] * WEIGHTS[45];
    sum += dataCache[centerIdx - 44] * WEIGHTS[44];
    sum += dataCache[centerIdx - 43] * WEIGHTS[43];
    sum += dataCache[centerIdx - 42] * WEIGHTS[42];
    sum += dataCache[centerIdx - 41] * WEIGHTS[41];
    sum += dataCache[centerIdx - 40] * WEIGHTS[40];
    sum += dataCache[centerIdx - 39] * WEIGHTS[39];
    sum += dataCache[centerIdx - 38] * WEIGHTS[38];
    sum += dataCache[centerIdx - 37] * WEIGHTS[37];
    sum += dataCache[centerIdx - 36] * WEIGHTS[36];
    sum += dataCache[centerIdx - 35] * WEIGHTS[35];
    sum += dataCache[centerIdx - 34] * WEIGHTS[34];
    sum += dataCache[centerIdx - 33] * WEIGHTS[33];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 32
    sum += dataCache[centerIdx - 32] * WEIGHTS[32];
    sum += dataCache[centerIdx - 31] * WEIGHTS[31];
    sum += dataCache[centerIdx - 30] * WEIGHTS[30];
    sum += dataCache[centerIdx - 29] * WEIGHTS[29];
    sum += dataCache[centerIdx - 28] * WEIGHTS[28];
    sum += dataCache[centerIdx - 27] * WEIGHTS[27];
    sum += dataCache[centerIdx - 26] * WEIGHTS[26];
    sum += dataCache[centerIdx - 25] * WEIGHTS[25];
    sum += dataCache[centerIdx - 24] * WEIGHTS[24];
    sum += dataCache[centerIdx - 23] * WEIGHTS[23];
    sum += dataCache[centerIdx - 22] * WEIGHTS[22];
    sum += dataCache[centerIdx - 21] * WEIGHTS[21];
    sum += dataCache[centerIdx - 20] * WEIGHTS[20];
    sum += dataCache[centerIdx - 19] * WEIGHTS[19];
    sum += dataCache[centerIdx - 18] * WEIGHTS[18];
    sum += dataCache[centerIdx - 17] * WEIGHTS[17];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 16
    sum += dataCache[centerIdx - 16] * WEIGHTS[16];
    sum += dataCache[centerIdx - 15] * WEIGHTS[15];
    sum += dataCache[centerIdx - 14] * WEIGHTS[14];
    sum += dataCache[centerIdx - 13] * WEIGHTS[13];
    sum += dataCache[centerIdx - 12] * WEIGHTS[12];
    sum += dataCache[centerIdx - 11] * WEIGHTS[11];
    sum += dataCache[centerIdx - 10] * WEIGHTS[10];
    sum += dataCache[centerIdx - 9] * WEIGHTS[9];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 8
    sum += dataCache[centerIdx - 8] * WEIGHTS[8];
    sum += dataCache[centerIdx - 7] * WEIGHTS[7];
    sum += dataCache[centerIdx - 6] * WEIGHTS[6];
    sum += dataCache[centerIdx - 5] * WEIGHTS[5];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 4
    sum += dataCache[centerIdx - 4] * WEIGHTS[4];
    sum += dataCache[centerIdx - 3] * WEIGHTS[3];
    #endif

    sum += dataCache[centerIdx - 2] * WEIGHTS[2];
    sum += dataCache[centerIdx - 1] * WEIGHTS[1];

    sum += dataCache[centerIdx] * WEIGHTS[0];

    sum += dataCache[centerIdx + 1] * WEIGHTS[1];
    sum += dataCache[centerIdx + 2] * WEIGHTS[2];

    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 4
    sum += dataCache[centerIdx + 3] * WEIGHTS[3];
    sum += dataCache[centerIdx + 4] * WEIGHTS[4];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 8
    sum += dataCache[centerIdx + 5] * WEIGHTS[5];
    sum += dataCache[centerIdx + 6] * WEIGHTS[6];
    sum += dataCache[centerIdx + 7] * WEIGHTS[7];
    sum += dataCache[centerIdx + 8] * WEIGHTS[8];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 16
    sum += dataCache[centerIdx + 9] * WEIGHTS[9];
    sum += dataCache[centerIdx + 10] * WEIGHTS[10];
    sum += dataCache[centerIdx + 11] * WEIGHTS[11];
    sum += dataCache[centerIdx + 12] * WEIGHTS[12];
    sum += dataCache[centerIdx + 13] * WEIGHTS[13];
    sum += dataCache[centerIdx + 14] * WEIGHTS[14];
    sum += dataCache[centerIdx + 15] * WEIGHTS[15];
    sum += dataCache[centerIdx + 16] * WEIGHTS[16];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 32
    sum += dataCache[centerIdx + 17] * WEIGHTS[17];
    sum += dataCache[centerIdx + 18] * WEIGHTS[18];
    sum += dataCache[centerIdx + 19] * WEIGHTS[19];
    sum += dataCache[centerIdx + 20] * WEIGHTS[20];
    sum += dataCache[centerIdx + 21] * WEIGHTS[21];
    sum += dataCache[centerIdx + 22] * WEIGHTS[22];
    sum += dataCache[centerIdx + 23] * WEIGHTS[23];
    sum += dataCache[centerIdx + 24] * WEIGHTS[24];
    sum += dataCache[centerIdx + 25] * WEIGHTS[25];
    sum += dataCache[centerIdx + 26] * WEIGHTS[26];
    sum += dataCache[centerIdx + 27] * WEIGHTS[27];
    sum += dataCache[centerIdx + 28] * WEIGHTS[28];
    sum += dataCache[centerIdx + 29] * WEIGHTS[29];
    sum += dataCache[centerIdx + 30] * WEIGHTS[30];
    sum += dataCache[centerIdx + 31] * WEIGHTS[31];
    sum += dataCache[centerIdx + 32] * WEIGHTS[32];
    #endif
    #if GAUSSIAN_BLUR_KERNEL_RADIUS >= 64
    sum += dataCache[centerIdx + 33] * WEIGHTS[33];
    sum += dataCache[centerIdx + 34] * WEIGHTS[34];
    sum += dataCache[centerIdx + 35] * WEIGHTS[35];
    sum += dataCache[centerIdx + 36] * WEIGHTS[36];
    sum += dataCache[centerIdx + 37] * WEIGHTS[37];
    sum += dataCache[centerIdx + 38] * WEIGHTS[38];
    sum += dataCache[centerIdx + 39] * WEIGHTS[39];
    sum += dataCache[centerIdx + 40] * WEIGHTS[40];
    sum += dataCache[centerIdx + 41] * WEIGHTS[41];
    sum += dataCache[centerIdx + 42] * WEIGHTS[42];
    sum += dataCache[centerIdx + 43] * WEIGHTS[43];
    sum += dataCache[centerIdx + 44] * WEIGHTS[44];
    sum += dataCache[centerIdx + 45] * WEIGHTS[45];
    sum += dataCache[centerIdx + 46] * WEIGHTS[46];
    sum += dataCache[centerIdx + 47] * WEIGHTS[47];
    sum += dataCache[centerIdx + 48] * WEIGHTS[48];
    sum += dataCache[centerIdx + 49] * WEIGHTS[49];
    sum += dataCache[centerIdx + 50] * WEIGHTS[50];
    sum += dataCache[centerIdx + 51] * WEIGHTS[51];
    sum += dataCache[centerIdx + 52] * WEIGHTS[52];
    sum += dataCache[centerIdx + 53] * WEIGHTS[53];
    sum += dataCache[centerIdx + 54] * WEIGHTS[54];
    sum += dataCache[centerIdx + 55] * WEIGHTS[55];
    sum += dataCache[centerIdx + 56] * WEIGHTS[56];
    sum += dataCache[centerIdx + 57] * WEIGHTS[57];
    sum += dataCache[centerIdx + 58] * WEIGHTS[58];
    sum += dataCache[centerIdx + 59] * WEIGHTS[59];
    sum += dataCache[centerIdx + 60] * WEIGHTS[60];
    sum += dataCache[centerIdx + 61] * WEIGHTS[61];
    sum += dataCache[centerIdx + 62] * WEIGHTS[62];
    sum += dataCache[centerIdx + 63] * WEIGHTS[63];
    #endif

    imageStore(GAUSSIAN_BLUR_OUTPUT, icoord, vec4(sum));
}
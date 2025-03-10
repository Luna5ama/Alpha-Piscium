// Contains code adopted from:
// https://www.shadertoy.com/view/XcdBWf
// MIT License
//
// You can find full license texts in /licenses
#include "/general/EnvProbe.glsl"
#include "/util/Coords.glsl"
#include "/util/Dither.glsl"
#include "/util/NZPacking.glsl"
#include "/util/BSDF.glsl"
#include "/util/FastMathLib.glsl"
#include "/util/Math.glsl"

uniform usampler2D usam_gbufferData;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_temp1;
uniform usampler2D usam_packedNZ;
uniform sampler2D usam_temp7;
uniform sampler2D usam_skyLUT;
uniform usampler2D usam_envProbe;

ivec2 vbgi_texelPos1x1;
ivec2 vbgi_texelPos2x2;

// Inverse function approximation
// See https://www.desmos.com/calculator/gs3clmp5hj
float radiusToLodStep(float y) {
    #if SSVBIL_SAMPLE_STEPS == 8
    const float a0 = 0.163523912451;
    const float a1 = 0.266941582847;
    const float a2 = -1.22024640462;
    #elif SSVBIL_SAMPLE_STEPS == 12
    const float a0 = 0.106829556893;
    const float a1 = 0.171480940545;
    const float a2 = -1.13391617357;
    #elif SSVBIL_SAMPLE_STEPS == 16
    const float a0 = 0.0795356209197;
    const float a1 = 0.125400926129;
    const float a2 = -1.07532664373;
    #elif SSVBIL_SAMPLE_STEPS == 24
    const float a0 = 0.0527947221187;
    const float a1 = 0.0808744790024;
    const float a2 = -0.999305945929;
    #elif SSVBIL_SAMPLE_STEPS == 32
    const float a0 = 0.0396467304144;
    const float a1 = 0.0590825497733;
    const float a2 = -0.939977972788;
    #elif SSVBIL_SAMPLE_STEPS == 64
    const float a0 = 0.0199939489785;
    const float a1 = 0.0279244889461;
    const float a2 = -0.817485681694;
    #else
    #error "Invalid SSVBIL_SAMPLE_STEPS"
    #endif
    y = clamp(y, SSVBIL_SAMPLE_STEPS, 32768.0);
    return max(a0 * log2(a1 * y + a2), 0.0);
}

float lodTexelSize(float lod) {
    return exp2(lod);
}


vec3 view2screen(vec3 vpos) {
    vec4 ppos = gbufferProjection * vec4(vpos, 1.0);
    vec2 tc21 = ppos.xy / ppos.w;
    vec2 uv0 = (tc21 * 0.5 + 0.5) * global_mainImageSize;
    return vec3(uv0, vpos.z);
}

#define NOISE_FRAME uint(frameCounter)

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////// partial slice sampling
//==================================================================================//

vec2 cmul(vec2 c0, vec2 c1) {
    return vec2(c0.x * c1.x - c0.y * c1.y,
        c0.y * c1.x + c0.x * c1.y);
}

float SamplePartialSlice(float x, float sin_thVN) {
    if (x == 0.0 || abs(x) >= 1.0) return x;

    bool sgn = x < 0.0;
    x = abs(x);

    float s = sin_thVN;

    float o = s - s * s;
    float slp0 = 1.0 / (1.0 + (PI  - 1.0) * (s - o * 0.30546));
    float slp1 = 1.0 / (1.0 - (1.0 - exp2(-20.0)) * (s + o * mix(0.5, 0.785, s)));
    float k = mix(0.1, 0.25, s);

    float a = 1.0 - (PI - 2.0) / (PI - 1.0);
    float b = 1.0 / (PI - 1.0);

    float d0 =   a - slp0 * b;
    float d1 = 1.0 - slp1;

    float f0 = d0 * (PI * x - asinFast4(clamp(x, -1.0, 1.0)));
    float f1 = d1 * (x - 1.0);

    float kk = k * k;

    float h0 = sqrt(f0*f0 + kk) - k;
    float h1 = sqrt(f1*f1 + kk) - k;

    float hh = (h0 * h1) / (h0 + h1);

    float y = x - sqrt(hh*(hh + 2.0*k));

    return sgn ? -y : y;
}

// vvsN: view vec space normal | rnd01: [0, 1]
vec2 SamplePartialSliceDir(vec3 vvsN, float rnd01) {
    float ang0 = rnd01 * PI_2;
    vec2 dir0 = vec2(cos(ang0), sin(ang0));
    float l = length(vvsN.xy);
    if (l == 0.0) return dir0;

    vec2 n = vvsN.xy / l;
    // align n with x-axis
    dir0 = cmul(dir0, n * vec2(1.0, -1.0));

    // sample slice angle
    float ang;
    {
        float x = atan(dir0.x, dir0.y) / PI;
        float sinNV = l;
        ang = SamplePartialSlice(x, sinNV) * PI;
    }

    // ray space slice direction
    vec2 dir = vec2(cos(ang), sin(ang));
    // align x-axis with n
    dir = cmul(dir, n);

    return dir;
}

//==================================================================================//
//////////////////////////////////////////////////////////////////////////////////////

const float MAX_RADIUS_SQ = SETTING_VBGI_MAX_RADIUS * SETTING_VBGI_MAX_RADIUS;

////////////////////////////////////////////////////////////////////////////////////// quaternion utils
//==================================================================================//

vec4 GetQuaternion(vec3 from, vec3 to) {
    vec3 xyz = cross(from, to);
    float s  =   dot(from, to);

    float u = inversesqrt(max(0.0, s * 0.5 + 0.5));// rcp(cosine half-angle formula)

    s    = 1.0 / u;
    xyz *= u * 0.5;

    return vec4(xyz, s);
}

vec4 GetQuaternion(vec3 to) {
    //vec3 from = vec3(0.0, 0.0, 1.0);

    vec3 xyz = vec3(-to.y, to.x, 0.0);// cross(from, to);
    float s  =                   to.z;//   dot(from, to);

    float u = inversesqrt(max(0.0, s * 0.5 + 0.5));// rcp(cosine half-angle formula)

    s    = 1.0 / u;
    xyz *= u * 0.5;

    return vec4(xyz, s);
}

// transform v by unit quaternion q.xyzs
vec3 Transform(vec3 v, vec4 q) {
    vec3 k = cross(q.xyz, v);

    return v + 2.0 * vec3(dot(vec3(q.wy, -q.z), k.xzy),
        dot(vec3(q.wz, -q.x), k.yxz),
        dot(vec3(q.wx, -q.y), k.zyx));
}

// transform v by unit quaternion q.xy0s
vec3 Transform_Qz0(vec3 v, vec4 q) {
    float k = v.y * q.x - v.x * q.y;
    float g = 2.0 * (v.z * q.w + k);

    vec3 r;
    r.xy = v.xy + q.yx * vec2(g, -g);
    r.z  = v.z  + 2.0 * (q.w * k - v.z * dot(q.xy, q.xy));

    return r;
}

// transform v.xy0 by unit quaternion q.xy0s
vec3 Transform_Vz0Qz0(vec2 v, vec4 q) {
    float o = q.x * v.y;
    float c = q.y * v.x;

    vec3 b = vec3(o - c,
        -o + c,
        o - c);

    return vec3(v, 0.0) + 2.0 * (b * q.yxw);
}

float sliceRelCDF(float x, float angN, float cosN) {
    if (x <= 0.0 || x >= 1.0) return x;

    float phi = x * PI - PI_HALF;

    const float n0 = 3.0;
    const float n1 = -1.0;
    const float n2 = 4.0;

    float t0 = n0 * cosN + n1 * cos(angN - 2.0 * phi) + (n2 * angN + (n1 * 2.0) * phi + PI) * sin(angN);
    float t1 = 4.0 * (cosN + angN * sin(angN));

    return t0 / t1;
}

const vec2 RADIUS_SQ = vec2(SETTING_VBGI_RADIUS * SETTING_VBGI_RADIUS, SETTING_VBGI_MAX_RADIUS * SETTING_VBGI_MAX_RADIUS);

uint toBitMask(vec2 h01) {
    uint bitmask;// turn arc into bit mask
    {
        uvec2 horInt = uvec2(floor(h01 * 32.0));

        uint OxFFFFFFFFu = 0xFFFFFFFFu;// don't inline here! ANGLE bug: https://issues.angleproject.org/issues/353039526

        uint mX = horInt.x < 32u ? OxFFFFFFFFu <<        horInt.x  : 0u;
        uint mY = horInt.y != 0u ? OxFFFFFFFFu >> (32u - horInt.y) : 0u;

        bitmask = mX & mY;
    }
    return bitmask;
}

void uniGTVBGI(vec3 viewPos, vec3 viewNormal, inout vec4 result) {
    vec3 viewDir = -normalize(viewPos);
    vec2 rayStart = view2screen(viewPos).xy;

    ////////////////////////////////////////////////// slice direction sampling
    vec3 smplDirVS;// view space sampling vector
    vec2 dir;// screen space sampling vector
    {
        // approximate partial slice dir importance sampling via 1 rnd number

        // set up View Vec Space <-> View Space mapping
        vec4   Q_toV = GetQuaternion(viewDir);
        vec4 Q_fromV = Q_toV * vec4(vec3(-1.0), 1.0);// conjugate
        vec3 normalVVS = viewNormal;
        normalVVS = Transform_Qz0(viewNormal, Q_fromV);

        float dirRand = rand_IGN(vbgi_texelPos2x2, frameCounter);
        dir = SamplePartialSliceDir(normalVVS, dirRand);

        smplDirVS = vec3(dir.xy, 0.0);
        smplDirVS = Transform_Vz0Qz0(dir, Q_toV);

        vec3 rayStart = view2screen(viewPos);
        vec3 rayEnd = view2screen(viewPos + smplDirVS*(near*0.5));
        vec3 rayDir = rayEnd - rayStart;

        rayDir /= length(rayDir.xy);
        dir = rayDir.xy;
    }
    //////////////////////////////////////////////////

    ////////////////////////////////////////////////// construct slice
    vec3 sliceN, projN, T;
    float cosN, angN, projNRcpLen;
    float sgn;
    {
        sliceN = cross(viewDir, smplDirVS);
        projN = viewNormal - sliceN * dot(viewNormal, sliceN);

        float projNSqrLen = dot(projN, projN);
        if (projNSqrLen == 0.0) return;

        projNRcpLen = inversesqrt(projNSqrLen);
        cosN = dot(projN, viewDir) * projNRcpLen;
        T = cross(sliceN, projN);

        sgn = dot(viewDir, T) < 0.0 ? -1.0 : 1.0;
        angN = sgn * acosFast4(clamp(cosN, -1.0, 1.0));
    }
    //////////////////////////////////////////////////

    float angOff = angN * RCP_PI + 0.5;

    // percentage of the slice we don't use ([0, angN]-integrated slice-relative pdf)
    float w0 = clamp((sin(angN) / (cos(angN) + angN * sin(angN))) * (PI/4.0) + 0.5, 0.0, 1.0);

    // partial slice re-mapping constants
    float w0_remap_mul = 1.0 / (1.0 - w0);
    float w0_remap_add = -w0 * w0_remap_mul;

    vec2 rayDir = dir.xy;

    vec2 texelCenterPos = vec2(vbgi_texelPos1x1) + 0.5;
    float maxDistX = rayDir.x != 0.0 ? (rayDir.x >= 0.0 ? (global_mainImageSize.x - texelCenterPos.x) / rayDir.x : -texelCenterPos.x / rayDir.x) : 1e6;
    float maxDistY = rayDir.y != 0.0 ? (rayDir.y < 0.0 ? (-texelCenterPos.y / rayDir.y) : (global_mainImageSize.y - texelCenterPos.y) / rayDir.y) : 1e6;
    float maxDist = min(maxDistX, maxDistY);

    uvec2 hashKey = (uvec2(vbgi_texelPos2x2 >> 1) & uvec2(31u)) ^ (NOISE_FRAME & 0xFFFFFFF0u);
    uint r2Index = (rand_hash21(hashKey) & 65535u) + NOISE_FRAME;
    float jitter = rand_r2Seq1(r2Index);

    float lodStep = radiusToLodStep(maxDist);
    float sampleLod = lodStep * jitter;

    float sampleTexelDist = 0.5;

    uint occBits = 0u;

    float NDotV = dot(viewNormal, viewDir);
    GBufferData gData;
    gbuffer_unpack(texelFetch(usam_gbufferData, vbgi_texelPos1x1, 0), gData);
    Material material = material_decode(gData);
    material.roughness = max(material.roughness, 0.01);

    float diffuseBase = (1.0 - material.metallic) * SETTING_VBGI_DGI_STRENGTH;
    float specularBase = PI * SETTING_VBGI_SGI_STRENGTH;

    float bitmaskJitter = jitter * (1.0 / 32.0);

    for (uint stepIndex = 0; stepIndex < SSVBIL_SAMPLE_STEPS; ++stepIndex) {
        float sampleLodTexelSize = lodTexelSize(sampleLod);
        float stepTexelSize = sampleLodTexelSize * 0.5;
        sampleTexelDist += stepTexelSize;

        int index = frameCounter & 3;
        ivec2 downSampleOffset = ivec2(index, index >> 1) & ivec2(1);
        vec2 sampleTexelPos = floor(rayDir * sampleTexelDist + rayStart);

        ivec2 sampleTexelPosHalf = clamp(ivec2(sampleTexelPos * 0.5), ivec2(0), ivec2(global_mipmapSizesI[1] - 1));
        vec2 sampleUV = saturate((sampleTexelPos + 0.5) / textureSize(usam_gbufferViewZ, 0).xy);

        float sampleViewZ;
        vec3 sampleWorldNormal;
        nzpacking_unpack(texelFetch(usam_packedNZ, sampleTexelPosHalf, 0).xy, sampleWorldNormal, sampleViewZ);

        vec3 samplePosVS = coords_toViewCoord(sampleUV, sampleViewZ, gbufferProjectionInverse);
        vec3 frontDiff = samplePosVS - viewPos;
        float frontDistSq = dot(frontDiff, frontDiff);

        if (frontDistSq < RADIUS_SQ.y) {
            vec3 backDiff = coords_toViewCoord(sampleUV, sampleViewZ - SETTING_VBGI_THICKNESS, gbufferProjectionInverse) - viewPos;

            float frontDiffRcpLen = fastRcpSqrtNR0(frontDistSq);
            float backDiffRcpLen = fastRcpSqrtNR0(dot(backDiff, backDiff));
            vec3 thisToSample = frontDiff * frontDiffRcpLen;

            // project samples onto unit circle and compute angles relative to viewDir
            vec2 horCos = vec2(dot(frontDiff * frontDiffRcpLen, viewDir), dot(backDiff * backDiffRcpLen, viewDir));

            vec2 horAng = acosFast4(clamp(horCos, -1.0, 1.0));

            // shift relative angles from viewDir to N + map to [0,1]
            vec2 hor01 = saturate(horAng * RCP_PI + angOff);

            // map to slice relative distribution
            hor01.x = sliceRelCDF(hor01.x, angN, cosN);
            hor01.y = sliceRelCDF(hor01.y, angN, cosN);

            // partial slice re-mapping
            hor01 = hor01 * w0_remap_mul + w0_remap_add;

            // jitter sample locations + clamp01
            hor01 = saturate(hor01 + bitmaskJitter);
            uint occBits0 = toBitMask(hor01);

            // compute gi contribution
            {
                uint visBits0 = occBits0 & (~occBits);

                if (visBits0 != 0u) {
                    vec4 sample2 = texelFetch(usam_temp1, sampleTexelPosHalf, 0);
                    float emissive = float(sample2.a > 0.0);
                    vec3 sampleViewNormal = mat3(gbufferModelView) * sampleWorldNormal;
                    float emitterCos = mix(saturate(dot(sampleViewNormal, -thisToSample)), 1.0, emissive);

                    float falloff = linearStep(RADIUS_SQ.y, RADIUS_SQ.x, frontDistSq);
                    vec3 sampleRad = sample2.rgb * falloff;
                    float bitV = float(bitCount(visBits0)) * (1.0 / 32.0);

                    vec3 N = viewNormal;
                    vec3 L = thisToSample;
                    float halfWayLen = sqrt(2.0 * dot(L, viewDir) + 2.0);
                    float NDotL = dot(N, L);
                    float NDotH = (NDotL + NDotV) / halfWayLen;
                    float LDotH = 0.5 * halfWayLen;
                    vec3 fresnel = bsdf_fresnel(material, saturate(LDotH));
                    float ggx = bsdf_ggx(material, NDotL, NDotV, NDotH);

                    vec3 indirectBounce = (vec3(1.0) - fresnel) * (diffuseBase);
                    indirectBounce += fresnel * (ggx * specularBase);
                    result.rgb += sampleRad * indirectBounce * (bitV * emitterCos * (PI * SETTING_VGBI_IB_STRENGTH));
                }
            }

            occBits = occBits | occBits0;
        }

        sampleLod = sampleLod + lodStep;
        sampleTexelDist += stepTexelSize;
    }

    // compute AO
    result.a = float(bitCount(occBits)) * (1.0 / 32.0);
    result.a = saturate(1.0 - result.a);
    result.a = pow(result.a, SETTING_VBGI_AO_STRENGTH);

    {
        mat3 viewToScene = mat3(gbufferModelViewInverse);

        uint unoccluedBits = ~occBits;
        vec3 realTangent = normalize(T);

        float lmCoordSky = texelFetch(usam_temp1, vbgi_texelPos2x2 + ivec2(global_mipmapSizesI[1].x, 0), 0).a;
        float skyLightingBase = SETTING_VBGI_SKYLIGHT_STRENGTH;

        #if SETTING_VBGI_FALLBACK_SAMPLES == 4
        const float w5 = 0.125;
        const float w1 = 0.25;
        #elif SETTING_VBGI_FALLBACK_SAMPLES == 8
        const float w5 = 0.0625;
        const float w1 = 0.125;
        #elif SETTING_VBGI_FALLBACK_SAMPLES == 16
        const float w5 = 0.03125;
        const float w1 = 0.0625;
        #elif SETTING_VBGI_FALLBACK_SAMPLES == 32
        const float w5 = 0.015625;
        const float w1 = 0.03125;
        #endif

        for (uint i = 0u; i < SETTING_VBGI_FALLBACK_SAMPLES; i++) {
            float fi = float(i) + jitter - 0.5;

            float ang0 = w1 * fi;
            vec2 hor01 = saturate(vec2(ang0, ang0 + w1));
            hor01.x = sliceRelCDF(hor01.x, angN, cosN);
            hor01.y = sliceRelCDF(hor01.y, angN, cosN);
            hor01 = hor01 * w0_remap_mul + w0_remap_add;
            hor01 = clamp(hor01 + bitmaskJitter, 0.0, 1.0);

            uint sectorBitMask = toBitMask(hor01);
            uint sectorBits = (unoccluedBits & sectorBitMask);
            float bitV = float(bitCount(sectorBits)) * (1.0 / 32.0);

            float angC = saturate(w5 + w1 * fi) * PI - PI_HALF;
            float cosC = cos(angC);
            float sinC = sin(angC);

            vec3 sampleDirView = normalize((viewNormal * cosC + realTangent * sinC));
            vec3 sampleDirWorld = viewToScene * sampleDirView;

            vec2 skyLUTUV = coords_octEncode01(sampleDirWorld);
            vec3 skyRad = skyLightingBase * texture(usam_skyLUT, skyLUTUV).rgb;

            vec2 envUV = coords_mercatorForward(sampleDirWorld);
            ivec2 envTexel = ivec2(envUV * ENV_PROBE_SIZE);
            EnvProbeData envData = envProbe_decode(texelFetch(usam_envProbe, envTexel, 0));
            vec3 envRad = envData.radiance * (2.0 * PI * SETTING_VGBI_ENV_STRENGTH);

            bool probeIsSky = envProbe_isSky(envData);
            vec3 sampleRad = probeIsSky ? skyRad : envRad;
            float emitterCos = probeIsSky ? 1.0 : saturate(dot(envData.normal, -sampleDirWorld));

            vec3 N = viewNormal;
            vec3 L = sampleDirView;
            float halfWayLen = sqrt(2.0 * dot(L, viewDir) + 2.0);
            float NDotL = dot(N, L);
            float NDotH = (NDotL + NDotV) / halfWayLen;
            float LDotH = 0.5 * halfWayLen;
            vec3 fresnel = bsdf_fresnel(material, saturate(LDotH));
            float ggx = bsdf_ggx(material, NDotL, NDotV, NDotH);

            vec3 fallbackLighting = (vec3(1.0) - fresnel) * (diffuseBase);
            fallbackLighting += fresnel * (ggx * specularBase);
            result.rgb += sampleRad * fallbackLighting * (bitV * emitterCos * result.a);
        }
    }

    result = dither_fp16(result, rand_IGN(vbgi_texelPos2x2, frameCounter));
}

vec4 gtvbgi(ivec2 texelPos2x2) {
    vbgi_texelPos2x2 = texelPos2x2;
    vbgi_texelPos1x1 = texelPos2x2 << 1;

    float centerViewZ = texelFetch(usam_gbufferViewZ, vbgi_texelPos1x1, 0).r;

    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);
    if (centerViewZ < 0.0) {
        vec2 screenPos = (vec2(vbgi_texelPos1x1) + 0.5) * global_mainImageSizeRcp;
        float centerZ;
        vec3 centerWorldNormal;
        nzpacking_unpack(texelFetch(usam_packedNZ, vbgi_texelPos2x2, 0).xy, centerWorldNormal, centerZ);

        vec3 centerViewNormal = mat3(gbufferModelView) * centerWorldNormal;
        vec3 centerViewPos = coords_toViewCoord(screenPos, centerZ, gbufferProjectionInverse);
        centerViewPos += centerViewNormal * (1.0 / 1024.0);

        uniGTVBGI(centerViewPos, centerViewNormal, result);
    }

    return result;
}
#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"
#include "/util/Sampling.glsl"


in vec2 frag_texCoord;

/* RENDERTARGETS:0,15 */
layout(location = 0) out vec4 rt_out;
layout(location = 1) out vec4 rt_taaLast;

// from https://github.com/GameTechDev/TAA
vec4 BicubicSampling5(sampler2D samplerV, vec2 inHistoryST){
    const vec2 rcpResolution = uval_mainImageSizeRcp;
    const vec2 fractional = fract(inHistoryST - 0.5);
    const vec2 uv = (floor(inHistoryST - 0.5) + vec2(0.5f, 0.5f)) * rcpResolution;

    // 5-tap bicubic sampling (for Hermite/Carmull-Rom filter) -- (approximate from original 16->9-tap bilinear fetching)
    const vec2 t = vec2(fractional);
    const vec2 t2 = vec2(fractional * fractional);
    const vec2 t3 = vec2(fractional * fractional * fractional);
    const float s = float(0.5);
    const vec2 w0 = -s * t3 + float(2.f) * s * t2 - s * t;
    const vec2 w1 = (float(2.f) - s) * t3 + (s - float(3.f)) * t2 + float(1.f);
    const vec2 w2 = (s - float(2.f)) * t3 + (3 - float(2.f) * s) * t2 + s * t;
    const vec2 w3 = s * t3 - s * t2;
    const vec2 s0 = w1 + w2;
    const vec2 f0 = w2 / (w1 + w2);
    const vec2 m0 = uv + f0 * rcpResolution;
    const vec2 tc0 = uv - 1.f * rcpResolution;
    const vec2 tc3 = uv + 2.f * rcpResolution;

    const vec4 A = vec4(texture(samplerV, vec2(m0.x, tc0.y)));
    const vec4 B = vec4(texture(samplerV, vec2(tc0.x, m0.y)));
    const vec4 C = vec4(texture(samplerV, vec2(m0.x, m0.y)));
    const vec4 D = vec4(texture(samplerV, vec2(tc3.x, m0.y)));
    const vec4 E = vec4(texture(samplerV, vec2(m0.x, tc3.y)));
    const vec4 color = (float(0.5f) * (A + B) * w0.x + A * s0.x + float(0.5f) * (A + B) * w3.x) * w0.y + (B * w0.x + C * s0.x + D * w3.x) * s0.y + (float(0.5f) * (B + E) * w0.x + E * s0.x + float(0.5f) * (D + E) * w3.x) * w3.y;
    return color;
}

struct ColorAABB {
    vec3 minVal;
    vec3 maxVal;
    vec3 moment1;
    vec3 moment2;
};

ColorAABB initAABB(vec3 colorYCoCg) {
    ColorAABB box;
    box.minVal = colorYCoCg;
    box.maxVal = colorYCoCg;
    box.moment1 = colorYCoCg;
    box.moment2 = colorYCoCg * colorYCoCg;
    return box;
}

void updateAABB(vec3 colorSRGB, inout ColorAABB box) {
    vec3 colorYCoCg = colors_SRGBToYCoCg(colorSRGB);
    box.minVal = min(box.minVal, colorYCoCg);
    box.maxVal = max(box.maxVal, colorYCoCg);
    box.moment1 += colorYCoCg;
    box.moment2 += colorYCoCg * colorYCoCg;
}

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

    GBufferData gData = gbufferData_init();
    gbufferData2_unpack(texelFetch(usam_gbufferData2, intTexCoord, 0), gData);

    float viewZ = texelFetch(usam_gbufferViewZ, intTexCoord, 0).r;
    vec3 currViewPos = coords_toViewCoord(frag_texCoord, viewZ, global_camProjInverse);
    vec4 prevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
    vec4 prevClipPos = global_prevCamProj * prevViewPos;
    prevClipPos /= prevClipPos.w;
    vec2 prevScreenPos = prevClipPos.xy * 0.5 + 0.5;

    vec3 currColor = texture(usam_main, frag_texCoord).rgb;
    vec4 prevResult = sampling_catmullBicubic5Tap(usam_taaLast, prevScreenPos * uval_mainImageSize, 0.5, uval_mainImageSizeRcp);
    vec3 prevColor = saturate(prevResult.rgb);

    vec2 pixelPosDiff = (frag_texCoord - prevScreenPos) * textureSize(usam_main, 0).xy;
    vec3 cameraDelta = uval_cameraDelta;
    float cameraSpeed = length(cameraDelta);
    float prevCameraSpeed = length(global_prevCameraDelta);
    float cameraSpeedDiff = abs(cameraSpeed - prevCameraSpeed);
    float pixelSpeed = length(pixelPosDiff);

    float lastFrameAccum = texture(usam_taaLast, frag_texCoord).a;
    float newFrameAccum = lastFrameAccum + 1.0;

    float speedSum = 0.0;
    speedSum += cameraSpeedDiff * 4.0;
    speedSum += cameraSpeed * 0.125;
    speedSum += pixelSpeed * 0.25;

    float extraReset = 1.0;
    #ifdef SETTING_SCREENSHOT_MODE
    extraReset *= (1.0 - saturate(cameraSpeedDiff * 114514.0));
    extraReset *= (1.0 - saturate(cameraSpeed * 114514.0));
    extraReset *= (1.0 - saturate(pixelSpeed * 114.0));
    #if SETTING_SCREENSHOT_MODE_SKIP_INITIAL
    extraReset *= float(frameCounter > SETTING_SCREENSHOT_MODE_SKIP_INITIAL);
    #endif
    #endif

    {
        vec3 currColorYCoCg = colors_SRGBToYCoCg(currColor);
        ColorAABB box = initAABB(currColorYCoCg);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(-1, 0)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(1, 0)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(0, -1)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(0, 1)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(-1, -1)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(1, -1)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(-1, 1)).rgb, box);
        updateAABB(textureOffset(usam_main, frag_texCoord, ivec2(1, 1)).rgb, box);

        vec3 mean = box.moment1 / 9.0;
        vec3 mean2 = box.moment2 / 9.0;
        vec3 variance = mean2 - mean * mean;
        vec3 stddev = sqrt(abs(variance));

        vec3 prevColorYCoCg = colors_SRGBToYCoCg(prevColor);
        vec3 varianceAABBMin = mean - stddev * 1.0;
        vec3 varianceAABBMax = mean + stddev * 1.0;

        const float clippingEps = FLT_MIN;
        vec3 delta = prevColorYCoCg - mean;
        delta /= max(1.0, length(delta / stddev));

        vec3 prevColorYCoCgEllipsoid = mean + delta;
        vec3 prevColorYCoCgAABBClamped = clamp(prevColorYCoCg, box.minVal, box.maxVal);

        float clampWeight = exp2(-speedSum);
        clampWeight *= extraReset;
        vec3 prevColorYCoCgClamped = mix(prevColorYCoCgEllipsoid, prevColorYCoCgAABBClamped, clampWeight);

        #ifdef SETTING_SCREENSHOT_MODE
        prevColor = colors_YCoCgToSRGB(mix(prevColorYCoCgClamped, prevColorYCoCg, extraReset));
        #else
        prevColor = colors_YCoCgToSRGB(prevColorYCoCgClamped);
        #endif
    }

    float frameReset = exp2(-0.25 * log2(1.0 + speedSum));
    newFrameAccum *= frameReset;
    #ifdef SETTING_SCREENSHOT_MODE
    float MIN_ACCUM_FRAMES = 1.0;
    float MAX_ACCUM_FRAMES = 100.0;
    #else
    float MIN_ACCUM_FRAMES = 2.0;
    float MAX_ACCUM_FRAMES = 100.0;
    if (gData.isHand) {
        MAX_ACCUM_FRAMES *= 0.1;
    }
    #endif

    newFrameAccum = clamp(newFrameAccum, MIN_ACCUM_FRAMES, MAX_ACCUM_FRAMES);

    float finalCurrWeight = 1.0 / newFrameAccum;
    #ifndef SETTING_TAA
    finalCurrWeight = 1.0;
    #endif

    rt_out.rgb = mix(prevColor, currColor, finalCurrWeight);
    rt_out.a = 1.0;

    rt_taaLast = vec4(rt_out.rgb, newFrameAccum);

    float ditherNoise = rand_IGN(intTexCoord, frameCounter);
    rt_taaLast.rgb = dither_fp16(rt_taaLast.rgb, ditherNoise);
}
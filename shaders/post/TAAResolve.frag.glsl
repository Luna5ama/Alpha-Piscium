#include "/util/Coords.glsl"
#include "/util/Colors.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"

uniform sampler2D usam_main;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;
uniform sampler2D usam_taaLast;

in vec2 frag_texCoord;

/* RENDERTARGETS:0,15 */
layout(location = 0) out vec4 rt_out;
layout(location = 1) out vec4 rt_taaLast;

// from https://github.com/GameTechDev/TAA
vec4 BicubicSampling5(sampler2D samplerV, vec2 inHistoryST){
    const vec2 rcpResolution = global_mainImageSizeRcp;
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

void updateMoments(vec3 colorSRGB, inout vec3 sum, inout vec3 sqSum) {
    vec3 color = colors_SRGBToYCoCg(colorSRGB);
    sum += color;
    sqSum += color * color;
}

void main() {
    ivec2 intTexCoord = ivec2(gl_FragCoord.xy);

    GBufferData gData;
    gbufferData2_unpack(texelFetch(usam_gbufferData8UN, intTexCoord, 0), gData);

    float viewZ = texelFetch(usam_gbufferViewZ, intTexCoord, 0).r;
    vec3 currViewPos = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
    vec4 prevViewPos = coord_viewCurrToPrev(vec4(currViewPos, 1.0), gData.isHand);
    vec4 prevClipPos = gbufferProjection * prevViewPos;
    prevClipPos /= prevClipPos.w;
    vec2 prevScreenPos = prevClipPos.xy * 0.5 + 0.5;

    vec3 currColor = texture(usam_main, frag_texCoord).rgb;

    vec4 prevResult = BicubicSampling5(usam_taaLast, prevScreenPos * global_mainImageSize);
    vec3 prevColor = saturate(prevResult.rgb);

    vec2 pixelPosDiff = (frag_texCoord - prevScreenPos) * textureSize(usam_main, 0).xy;
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    float cameraSpeed = length(cameraDelta);
    float prevCameraSpeed = length(global_prevCameraDelta);
    float cameraSpeedDiff = abs(cameraSpeed - prevCameraSpeed);
    float pixelSpeed = length(pixelPosDiff);

    #ifndef SETTING_SCREENSHOT_MODE
    vec3 curr3x3Avg = vec3(0.0);
    vec3 curr3x3SqAvg = vec3(0.0);
    updateMoments(currColor, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(-1, 0)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(1, 0)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(0, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(0, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(-1, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(1, -1)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(-1, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
    updateMoments(textureOffset(usam_main, frag_texCoord, ivec2(1, 1)).rgb, curr3x3Avg, curr3x3SqAvg);
    curr3x3Avg /= 9.0;
    curr3x3SqAvg /= 9.0;

    // Ellipsoid intersection clipping by Marty
    const float clippingEps = 0.00001;
    vec3 prevColorYCoCg = colors_SRGBToYCoCg(prevColor);
    vec3 stddev = sqrt(max(curr3x3SqAvg - curr3x3Avg * curr3x3Avg, clippingEps));
    vec3 delta = prevColorYCoCg - curr3x3Avg;
    delta /= max(1.0, length(delta / stddev));

    float clipWeight = 0.5;
    clipWeight += saturate(1.0 - prevResult.a);
    clipWeight += pixelSpeed * 0.05;
    clipWeight += cameraSpeed * 0.1;
    clipWeight += cameraSpeedDiff * 4.0;
    clipWeight = saturate(clipWeight);
    prevColorYCoCg = mix(prevColorYCoCg, curr3x3Avg + delta, clipWeight);

    prevColor = colors_YCoCgToSRGB(prevColorYCoCg);
    #endif

    float lastMixWeight = texture(usam_taaLast, frag_texCoord).a;

    float mixWeight = 0.95;
    mixWeight = mix(lastMixWeight, mixWeight, 0.5);

    #ifdef SETTING_SCREENSHOT_MODE
    float mixDecrease = 1.0;
    mixDecrease *= (1.0 - saturate(cameraSpeedDiff * 114.0));
    mixDecrease *= (1.0 - saturate(cameraSpeed * 69.0));
    mixDecrease *= (1.0 - saturate(pixelSpeed * 69.0));
    #else
    float mixDecrease = 1.0;
    mixDecrease *= (1.0 - saturate(cameraSpeedDiff * 4.0));
    mixDecrease *= (1.0 - saturate(cameraSpeed * 0.02));
    mixDecrease *= (1.0 - saturate(pixelSpeed * 0.01));
    mixDecrease = max(mixDecrease, 0.75);
    #endif

    mixWeight = mixWeight * mixDecrease;

    float finalMixWeight = mixWeight;
    finalMixWeight *= (1.0 - min(cameraSpeedDiff * 1.0, 0.5));

    #ifdef SETTING_SCREENSHOT_MODE
    finalMixWeight = clamp(finalMixWeight, 0.0, 0.99);
    #else
    finalMixWeight = clamp(finalMixWeight, 0.5, 0.99);
    #endif

    mixWeight = mix(lastMixWeight + 0.01, mixWeight, 0.05);

    #ifndef SETTING_SCREENSHOT_MODE
    mixWeight = saturate(mixWeight - float(gData.isHand) * 0.2);
    #endif

    rt_out.rgb = mix(currColor, prevColor, finalMixWeight);
    rt_out.a = 1.0;
    rt_taaLast = vec4(rt_out.rgb, mixWeight);

    float ditherNoise = rand_IGN(intTexCoord, frameCounter);
    rt_taaLast.rgb = dither_fp16(rt_taaLast.rgb, ditherNoise);
    rt_out.rgb = dither_u8(rt_out.rgb, ditherNoise);
}
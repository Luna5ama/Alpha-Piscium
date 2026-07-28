#ifndef PARALLAX_TRACE_GLSL
#define PARALLAX_TRACE_GLSL

#include "/techniques/parallax/Common.glsl"

uniform sampler2D usam_blocksNormal;
uniform sampler2D usam_materialDepthMip;

float materialDepthMaxAlpha(ivec2 atlasTexel, int level, ivec2 atlasSize) {
    if (level == 0) {
        return texelFetch(usam_blocksNormal, clamp(atlasTexel, ivec2(0), atlasSize - 1), 0).a;
    }

    ivec2 mipSize = mipPackedSize(atlasSize, level);
    ivec2 mipTexel = clamp(atlasTexel, ivec2(0), mipSize - 1);
    return texelFetch(usam_materialDepthMip, mipPackedOffset(atlasSize, level) + mipTexel, 0).r;
}

#if SETTING_PARALLAX_MODE != 0
ivec2 wrapParallaxCell(ivec2 cell, ivec2 cellMin, ivec2 cellMax) {
    ivec2 cellExtent = max(cellMax - cellMin, ivec2(1));
    return cellMin + ivec2(mod(vec2(cell - cellMin), vec2(cellExtent)));
}
#endif

#if SETTING_PARALLAX_MODE == 2
float smoothstepBilinearDepth(vec4 depths, vec2 position) {
    vec2 weight = position * position * (3.0 - 2.0 * position);
    return mix(mix(depths.x, depths.y, weight.x), mix(depths.z, depths.w, weight.x), weight.y);
}

vec2 smoothstepBilinearGradient(vec4 depths, vec2 position) {
    vec2 weight = position * position * (3.0 - 2.0 * position);
    vec2 weightGradient = 6.0 * position * (1.0 - position);
    return weightGradient * vec2(
        mix(depths.y - depths.x, depths.w - depths.z, weight.y),
        mix(depths.z - depths.x, depths.w - depths.y, weight.x)
    );
}
#endif

bool traceParallax(
    vec2 atlasTexCoord,
    vec4 spriteBounds,
    vec2 rayDeltaTexels,
    out vec2 hitTexCoord,
    out float hitT,
    out vec3 hitSurfaceNormal
) {
    const float texelEpsilon = 1e-3;
    float tEpsilon = texelEpsilon / max(max(abs(rayDeltaTexels.x), abs(rayDeltaTexels.y)), 1.0);
    ivec2 atlasSizeI = textureSize(usam_blocksNormal, 0);
    vec2 atlasSize = vec2(atlasSizeI);
    vec2 spriteMin = clamp(spriteBounds.xy * atlasSize, vec2(0.0), atlasSize);
    vec2 spriteMax = clamp(spriteBounds.zw * atlasSize, spriteMin + texelEpsilon, atlasSize);
    vec2 spriteExtent = spriteMax - spriteMin;
    #if SETTING_PARALLAX_MODE != 0
    ivec2 spriteTexelMin = ivec2(round(spriteMin));
    ivec2 spriteTexelMax = ivec2(round(spriteMax));
    #endif
    vec2 rayStart = clamp(atlasTexCoord * atlasSize, spriteMin + texelEpsilon, spriteMax - texelEpsilon);
    vec2 rayStep = sign(rayDeltaTexels);
    vec2 rayDeltaRcp = rayStep / max(abs(rayDeltaTexels), vec2(1e-20));

    float rayMaxT = 1.0;

    int maxExtent = max(1, int(ceil(max(spriteMax.x - spriteMin.x, spriteMax.y - spriteMin.y))));
    int startLevel = min(14, findMSB(maxExtent - 1) + 1);
    int level = startLevel;
    float t = 0.0;
    #if SETTING_PARALLAX_MODE == 0
    float entryT = 0.0;
    vec2 entryNormal = vec2(0.0);
    #endif

    for (int iteration = 0; iteration < SETTING_STEEP_PARALLAX_MAX_ITERATIONS && t <= rayMaxT; iteration++) {
        vec2 rawPosition = rayStart + rayDeltaTexels * t;
        bvec2 activeAxis = notEqual(rayDeltaTexels, vec2(0.0));
        vec2 samplePosition = spriteMin + mod(rawPosition + mix(vec2(0.0), rayStep * texelEpsilon, activeAxis) - spriteMin, spriteExtent);
        int cellScaleI = 1 << level;
        float cellScale = float(cellScaleI);
        ivec2 cell = ivec2(samplePosition / cellScale);
        vec2 cellMin = max(vec2(cell * cellScaleI), spriteMin);
        vec2 cellMax = min(vec2((cell + 1) * cellScaleI), spriteMax);

        vec2 cellExit = mix(cellMin, cellMax, greaterThan(rayStep, vec2(0.0)));
        vec2 tExitXY = mix(vec2(rayMaxT), t + (cellExit - samplePosition + rayStep * texelEpsilon) * rayDeltaRcp, activeAxis);
        float tExit = min(rayMaxT, min(tExitXY.x, tExitXY.y));
        if (level == 0) {
            bool leafHit;
            #if SETTING_PARALLAX_MODE == 0
            float surfaceDepth = 1.0 - materialDepthMaxAlpha(cell, 0, atlasSizeI);
            bool sideHit = any(notEqual(entryNormal, vec2(0.0))) && surfaceDepth + tEpsilon < entryT;
            float candidateT = max(t, surfaceDepth);
            leafHit = candidateT <= tExit + tEpsilon;
            if (leafHit) {
                hitT = sideHit ? entryT : candidateT;
                hitSurfaceNormal = sideHit ? vec3(entryNormal, 0.0) : vec3(0.0, 0.0, 1.0);
            }
            #else
            ivec2 texel00 = wrapParallaxCell(cell, spriteTexelMin, spriteTexelMax);
            ivec2 texel10 = wrapParallaxCell(cell + ivec2(1, 0), spriteTexelMin, spriteTexelMax);
            ivec2 texel01 = wrapParallaxCell(cell + ivec2(0, 1), spriteTexelMin, spriteTexelMax);
            ivec2 texel11 = wrapParallaxCell(cell + ivec2(1), spriteTexelMin, spriteTexelMax);
            float depth00 = 1.0 - materialDepthMaxAlpha(texel00, 0, atlasSizeI);
            float depth10 = 1.0 - materialDepthMaxAlpha(texel10, 0, atlasSizeI);
            float depth01 = 1.0 - materialDepthMaxAlpha(texel01, 0, atlasSizeI);
            float depth11 = 1.0 - materialDepthMaxAlpha(texel11, 0, atlasSizeI);
            vec2 localPosition = samplePosition - mix(vec2(0.0), rayStep * texelEpsilon, activeAxis) - vec2(cell);
            float segmentLength = max(tExit - t, 0.0);
            vec2 segmentDelta = rayDeltaTexels * segmentLength;
            #if SETTING_PARALLAX_MODE == 1
            float depthX = depth10 - depth00;
            float depthY = depth01 - depth00;
            float depthXY = depth11 - depth10 - depth01 + depth00;

            float depthStart = depth00 + depthX * localPosition.x + depthY * localPosition.y
                + depthXY * localPosition.x * localPosition.y;
            float constantTerm = t - depthStart;
            float linearTerm = segmentLength - depthX * segmentDelta.x - depthY * segmentDelta.y
                - depthXY * (localPosition.x * segmentDelta.y + localPosition.y * segmentDelta.x);
            float quadraticTerm = -depthXY * segmentDelta.x * segmentDelta.y;

            float hitSegment = constantTerm >= -tEpsilon ? 0.0 : 2.0;
            if (hitSegment > 1.0 && abs(quadraticTerm) < 1e-7) {
                if (linearTerm > 0.0) {
                    hitSegment = -constantTerm / linearTerm;
                }
            } else if (hitSegment > 1.0) {
                float discriminant = linearTerm * linearTerm - 4.0 * quadraticTerm * constantTerm;
                if (discriminant >= 0.0) {
                    float rootScale = 0.5 / quadraticTerm;
                    float rootOffset = -linearTerm * rootScale;
                    float rootDelta = sqrt(discriminant) * abs(rootScale);
                    float root0 = rootOffset - rootDelta;
                    float root1 = rootOffset + rootDelta;
                    hitSegment = root0 >= -1e-4 ? root0 : root1;
                }
            }
            leafHit = hitSegment >= -1e-4 && hitSegment <= 1.0001;
            if (leafHit) {
                hitSegment = clamp(hitSegment, 0.0, 1.0);
                hitT = t + segmentLength * hitSegment;
                vec2 hitPosition = localPosition + segmentDelta * hitSegment;
                vec2 depthGradient = vec2(depthX + depthXY * hitPosition.y, depthY + depthXY * hitPosition.x);
                hitSurfaceNormal = vec3(depthGradient * SETTING_STEEP_PARALLAX_DEPTH * spriteExtent, 1.0);
            }
            #else
            vec4 depths = vec4(depth00, depth10, depth01, depth11);
            float hitSegment = 2.0;
            float previousSegment = 0.0;
            float startDifference = t - smoothstepBilinearDepth(depths, localPosition);
            float previousDerivative = segmentLength
                - dot(smoothstepBilinearGradient(depths, localPosition), segmentDelta);
            if (startDifference >= -tEpsilon) {
                hitSegment = 0.0;
            } else {
                for (int step = 1; step <= 8; step++) {
                    float candidateSegment = float(step) * 0.125;
                    vec2 candidatePosition = localPosition + segmentDelta * candidateSegment;
                    float candidateDifference = t + segmentLength * candidateSegment
                        - smoothstepBilinearDepth(depths, candidatePosition);
                    float candidateDerivative = segmentLength
                        - dot(smoothstepBilinearGradient(depths, candidatePosition), segmentDelta);
                    float upperSegment = candidateSegment;
                    bool bracketed = candidateDifference >= -tEpsilon;
                    if (!bracketed && previousDerivative > 0.0 && candidateDerivative < 0.0) {
                        float derivativeLower = previousSegment;
                        float derivativeUpper = candidateSegment;
                        for (int refinement = 0; refinement < 8; refinement++) {
                            float middleSegment = (derivativeLower + derivativeUpper) * 0.5;
                            vec2 middlePosition = localPosition + segmentDelta * middleSegment;
                            float middleDerivative = segmentLength
                                - dot(smoothstepBilinearGradient(depths, middlePosition), segmentDelta);
                            if (middleDerivative > 0.0) {
                                derivativeLower = middleSegment;
                            } else {
                                derivativeUpper = middleSegment;
                            }
                        }
                        upperSegment = (derivativeLower + derivativeUpper) * 0.5;
                        vec2 peakPosition = localPosition + segmentDelta * upperSegment;
                        float peakDifference = t + segmentLength * upperSegment
                            - smoothstepBilinearDepth(depths, peakPosition);
                        bracketed = peakDifference >= -tEpsilon;
                    }
                    if (bracketed) {
                        float lowerSegment = previousSegment;
                        for (int refinement = 0; refinement < 6; refinement++) {
                            float middleSegment = (lowerSegment + upperSegment) * 0.5;
                            vec2 middlePosition = localPosition + segmentDelta * middleSegment;
                            float middleDifference = t + segmentLength * middleSegment
                                - smoothstepBilinearDepth(depths, middlePosition);
                            if (middleDifference >= -tEpsilon) {
                                upperSegment = middleSegment;
                            } else {
                                lowerSegment = middleSegment;
                            }
                        }
                        hitSegment = upperSegment;
                        break;
                    }
                    previousSegment = candidateSegment;
                    previousDerivative = candidateDerivative;
                }
            }
            leafHit = hitSegment <= 1.0;
            if (leafHit) {
                hitT = t + segmentLength * hitSegment;
                vec2 hitPosition = localPosition + segmentDelta * hitSegment;
                vec2 depthGradient = smoothstepBilinearGradient(depths, hitPosition);
                hitSurfaceNormal = vec3(depthGradient * SETTING_STEEP_PARALLAX_DEPTH * spriteExtent, 1.0);
            }
            #endif
            #endif
            if (leafHit) {
                vec2 hitTexel = spriteMin + mod(rayStart + rayDeltaTexels * hitT + rayStep * texelEpsilon - spriteMin, spriteExtent);
                hitTexCoord = hitTexel / atlasSize;
                return true;
            }
        } else {
            #if SETTING_PARALLAX_MODE == 0
            float surfaceDepth = 1.0 - materialDepthMaxAlpha(cell, level, atlasSizeI);
            #else
            ivec2 mipCellMin = ivec2(floor(spriteMin / cellScale));
            ivec2 mipCellMax = ivec2(ceil(spriteMax / cellScale));
            ivec2 cellX = wrapParallaxCell(cell + ivec2(1, 0), mipCellMin, mipCellMax);
            ivec2 cellY = wrapParallaxCell(cell + ivec2(0, 1), mipCellMin, mipCellMax);
            ivec2 cellXY = wrapParallaxCell(cell + ivec2(1), mipCellMin, mipCellMax);
            float maxSurfaceAlpha = materialDepthMaxAlpha(cell, level, atlasSizeI);
            maxSurfaceAlpha = max(maxSurfaceAlpha, materialDepthMaxAlpha(cellX, level, atlasSizeI));
            maxSurfaceAlpha = max(maxSurfaceAlpha, materialDepthMaxAlpha(cellY, level, atlasSizeI));
            maxSurfaceAlpha = max(maxSurfaceAlpha, materialDepthMaxAlpha(cellXY, level, atlasSizeI));
            float surfaceDepth = 1.0 - maxSurfaceAlpha;
            #endif
            if (tExit + tEpsilon >= surfaceDepth) {
                level -= 1;
                continue;
            }
        }

        if (tExit >= rayMaxT - tEpsilon) {
            break;
        }
        float nextT = max(tExit + tEpsilon, t + tEpsilon);
        if (nextT > rayMaxT) {
            break;
        }
        #if SETTING_PARALLAX_MODE == 0
        entryT = tExit;
        entryNormal = vec2(0.0);
        if (activeAxis.x && tExitXY.x <= tExitXY.y) {
            entryNormal.x = -rayStep.x;
        } else if (activeAxis.y) {
            entryNormal.y = -rayStep.y;
        }
        #endif
        t = nextT;
        level = min(startLevel, level + 1);
    }

    hitTexCoord = atlasTexCoord;
    hitT = 0.0;
    hitSurfaceNormal = vec3(0.0, 0.0, 1.0);
    return false;
}

#endif

#ifndef INCLUDE_techniques_parallax_Trace_glsl
#define INCLUDE_techniques_parallax_Trace_glsl a

#include "/techniques/parallax/Common.glsl"

uniform sampler2D usam_blocksNormal;
uniform sampler2D usam_materialDepthMip;

float _parallax_materialDepthMaxAlpha(ivec2 atlasTexel, ivec4 mipData) {
    return texelFetch(usam_materialDepthMip, mipData.zw + atlasTexel, 0).r;
}

#if SETTING_PARALLAX_MODE != 0
ivec2 _parallax_wrapParallaxCell(ivec2 cell, ivec2 cellMin, ivec2 cellMax) {
    ivec2 cellExtent = max(cellMax - cellMin, ivec2(1));
    cell = mix(cell, cell + cellExtent, lessThan(cell, cellMin));
    cell = mix(cell, cell - cellExtent, greaterThanEqual(cell, cellMax));
    return mix(cell, cell - cellExtent, greaterThanEqual(cell, cellMax));
}

float _parallax_wrappedMaterialDepth(ivec2 cell, ivec2 cellMin, ivec2 cellMax, ivec4 mipData) {
    return 1.0 - _parallax_materialDepthMaxAlpha(_parallax_wrapParallaxCell(cell, cellMin, cellMax), mipData);
}

#if SETTING_PARALLAX_MODE > 1
vec4 _parallax_gatherMaterialDepthAlpha(ivec2 cell, ivec4 mipData, vec2 packedTexelRcp) {
    vec2 texCoord = (vec2(mipData.zw + cell) + 1.0) * packedTexelRcp;
    return textureGather(usam_materialDepthMip, texCoord).wzxy;
}
#endif
#endif

#if SETTING_PARALLAX_MODE == 3
vec3 _parallax_continuousParallaxSurface(vec4 coefficients, vec2 position) {
    vec2 weight = position * position * (3.0 - 2.0 * position);
    vec2 weightGradient = 6.0 * position * (1.0 - position);
    return vec3(
        coefficients.x + coefficients.y * weight.x + coefficients.z * weight.y
            + coefficients.w * weight.x * weight.y,
        weightGradient * vec2(
            coefficients.y + coefficients.w * weight.y,
            coefficients.z + coefficients.w * weight.x
        )
    );
}
#elif SETTING_PARALLAX_MODE == 4
vec4 _parallax_bSplineAxisCoefficients(vec4 samples) {
    return vec4(
        (samples.x + 4.0 * samples.y + samples.z) * (1.0 / 6.0),
        (samples.z - samples.x) * 0.5,
        (samples.x - 2.0 * samples.y + samples.z) * 0.5,
        (-samples.x + 3.0 * samples.y - 3.0 * samples.z + samples.w) * (1.0 / 6.0)
    );
}

mat4 _parallax_bSplineCoefficients(mat4 depths) {
    mat4 coefficientsY = mat4(
        (depths[0] + 4.0 * depths[1] + depths[2]) * (1.0 / 6.0),
        (depths[2] - depths[0]) * 0.5,
        (depths[0] - 2.0 * depths[1] + depths[2]) * 0.5,
        (-depths[0] + 3.0 * depths[1] - 3.0 * depths[2] + depths[3]) * (1.0 / 6.0)
    );
    return mat4(
        _parallax_bSplineAxisCoefficients(coefficientsY[0]),
        _parallax_bSplineAxisCoefficients(coefficientsY[1]),
        _parallax_bSplineAxisCoefficients(coefficientsY[2]),
        _parallax_bSplineAxisCoefficients(coefficientsY[3])
    );
}

vec3 _parallax_continuousParallaxSurface(mat4 coefficients, vec2 position) {
    vec4 coefficientsX = ((coefficients[3] * position.y + coefficients[2]) * position.y
        + coefficients[1]) * position.y + coefficients[0];
    vec4 coefficientsDerivativeY = (3.0 * coefficients[3] * position.y
        + 2.0 * coefficients[2]) * position.y + coefficients[1];
    return vec3(
        ((coefficientsX.w * position.x + coefficientsX.z) * position.x + coefficientsX.y) * position.x + coefficientsX.x,
        (3.0 * coefficientsX.w * position.x + 2.0 * coefficientsX.z) * position.x + coefficientsX.y,
        ((coefficientsDerivativeY.w * position.x + coefficientsDerivativeY.z) * position.x
            + coefficientsDerivativeY.y) * position.x + coefficientsDerivativeY.x
    );
}
#endif

bool parallax_traceParallax(
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
    #if SETTING_PARALLAX_MODE > 1
    vec2 packedTexelRcp = 1.0 / vec2(textureSize(usam_materialDepthMip, 0));
    #endif
    #endif
    vec2 rayStart = clamp(atlasTexCoord * atlasSize, spriteMin + texelEpsilon, spriteMax - texelEpsilon);
    vec2 rayStep = sign(rayDeltaTexels);
    bvec2 activeAxis = notEqual(rayDeltaTexels, vec2(0.0));
    vec2 rayBias = mix(vec2(0.0), rayStep * texelEpsilon, activeAxis);
    vec2 rayDeltaRcp = rayStep / max(abs(rayDeltaTexels), vec2(1e-20));

    float rayMaxT = 1.0;

    int maxExtent = max(1, int(ceil(max(spriteMax.x - spriteMin.x, spriteMax.y - spriteMin.y))));
    int startLevel = min(14, findMSB(maxExtent - 1) + 1);
    int level = startLevel;
    float t = 0.0;
    #if SETTING_PARALLAX_MODE == 1
    float entryT = 0.0;
    vec2 entryNormal = vec2(0.0);
    #endif

    for (int iteration = 0; iteration < SETTING_STEEP_PARALLAX_MAX_ITERATIONS && t <= rayMaxT; iteration++) {
        vec2 rawPosition = rayStart + rayDeltaTexels * t;
        vec2 samplePosition = spriteMin + mod(rawPosition + rayBias - spriteMin, spriteExtent);
        int cellScaleI = 1 << level;
        float cellScale = float(cellScaleI);
        ivec2 cell = ivec2(samplePosition / cellScale);
        ivec4 mipData = global_parallaxMipPackedData[level];
        vec2 cellMin = max(vec2(cell * cellScaleI), spriteMin);
        vec2 cellMax = min(vec2((cell + 1) * cellScaleI), spriteMax);

        vec2 cellExit = mix(cellMin, cellMax, greaterThan(rayStep, vec2(0.0)));
        vec2 tExitXY = mix(vec2(rayMaxT), t + (cellExit - samplePosition + rayStep * texelEpsilon) * rayDeltaRcp, activeAxis);
        float tExit = min(rayMaxT, min(tExitXY.x, tExitXY.y));
        if (level == 0) {
            bool leafHit;
            #if SETTING_PARALLAX_MODE == 1
            float surfaceDepth = 1.0 - _parallax_materialDepthMaxAlpha(cell, mipData);
            bool sideHit = any(notEqual(entryNormal, vec2(0.0))) && surfaceDepth + tEpsilon < entryT;
            float candidateT = max(t, surfaceDepth);
            leafHit = candidateT <= tExit + tEpsilon;
            if (leafHit) {
                hitT = sideHit ? entryT : candidateT;
                hitSurfaceNormal = sideHit ? vec3(entryNormal, 0.0) : vec3(0.0, 0.0, 1.0);
            }
            #else
            vec2 localPosition = samplePosition - rayBias - vec2(cell);
            float segmentLength = max(tExit - t, 0.0);
            vec2 segmentDelta = rayDeltaTexels * segmentLength;
            #if SETTING_PARALLAX_MODE == 4
            mat4 depthSamples;
            if (all(greaterThan(cell, spriteTexelMin)) && all(lessThan(cell + ivec2(2), spriteTexelMax))) {
                vec4 gather00 = 1.0 - _parallax_gatherMaterialDepthAlpha(cell + ivec2(-1), mipData, packedTexelRcp);
                vec4 gather10 = 1.0 - _parallax_gatherMaterialDepthAlpha(cell + ivec2(1, -1), mipData, packedTexelRcp);
                vec4 gather01 = 1.0 - _parallax_gatherMaterialDepthAlpha(cell + ivec2(-1, 1), mipData, packedTexelRcp);
                vec4 gather11 = 1.0 - _parallax_gatherMaterialDepthAlpha(cell + ivec2(1), mipData, packedTexelRcp);
                depthSamples = mat4(
                    vec4(gather00.xy, gather10.xy),
                    vec4(gather00.zw, gather10.zw),
                    vec4(gather01.xy, gather11.xy),
                    vec4(gather01.zw, gather11.zw)
                );
            } else {
                depthSamples = mat4(
                    vec4(
                        _parallax_wrappedMaterialDepth(cell + ivec2(-1, -1), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(0, -1), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(1, -1), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(2, -1), spriteTexelMin, spriteTexelMax, mipData)
                    ),
                    vec4(
                        _parallax_wrappedMaterialDepth(cell + ivec2(-1, 0), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(0, 0), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(1, 0), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(2, 0), spriteTexelMin, spriteTexelMax, mipData)
                    ),
                    vec4(
                        _parallax_wrappedMaterialDepth(cell + ivec2(-1, 1), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(0, 1), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(1, 1), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(2, 1), spriteTexelMin, spriteTexelMax, mipData)
                    ),
                    vec4(
                        _parallax_wrappedMaterialDepth(cell + ivec2(-1, 2), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(0, 2), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(1, 2), spriteTexelMin, spriteTexelMax, mipData),
                        _parallax_wrappedMaterialDepth(cell + ivec2(2, 2), spriteTexelMin, spriteTexelMax, mipData)
                    )
                );
            }
            #else
            vec4 depths;
            if (all(lessThan(cell + ivec2(1), spriteTexelMax))) {
                depths = 1.0 - _parallax_gatherMaterialDepthAlpha(cell, mipData, packedTexelRcp);
            } else {
                depths = vec4(
                    _parallax_wrappedMaterialDepth(cell, spriteTexelMin, spriteTexelMax, mipData),
                    _parallax_wrappedMaterialDepth(cell + ivec2(1, 0), spriteTexelMin, spriteTexelMax, mipData),
                    _parallax_wrappedMaterialDepth(cell + ivec2(0, 1), spriteTexelMin, spriteTexelMax, mipData),
                    _parallax_wrappedMaterialDepth(cell + ivec2(1), spriteTexelMin, spriteTexelMax, mipData)
                );
            }
            #endif
            #if SETTING_PARALLAX_MODE == 2
            float depth00 = depths.x;
            float depth10 = depths.y;
            float depth01 = depths.z;
            float depth11 = depths.w;
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
            #if SETTING_PARALLAX_MODE == 3
            depths = vec4(
                depths.x,
                depths.y - depths.x,
                depths.z - depths.x,
                depths.w - depths.y - depths.z + depths.x
            );
            #elif SETTING_PARALLAX_MODE == 4
            mat4 depths = _parallax_bSplineCoefficients(depthSamples);
            #endif
            float hitSegment = 2.0;
            float previousSegment = 0.0;
            vec3 startSurface = _parallax_continuousParallaxSurface(depths, localPosition);
            float startDifference = t - startSurface.x;
            #if SETTING_PARALLAX_MODE == 4
            float previousDifference = startDifference;
            #endif
            float previousDerivative = segmentLength - dot(startSurface.yz, segmentDelta);
            if (startDifference >= -tEpsilon) {
                hitSegment = 0.0;
            } else {
                for (int step = 1; step <= 8; step++) {
                    float candidateSegment = float(step) * 0.125;
                    vec2 candidatePosition = localPosition + segmentDelta * candidateSegment;
                    vec3 candidateSurface = _parallax_continuousParallaxSurface(depths, candidatePosition);
                    float candidateDifference = t + segmentLength * candidateSegment - candidateSurface.x;
                    float candidateDerivative = segmentLength - dot(candidateSurface.yz, segmentDelta);
                    float upperSegment = candidateSegment;
                    #if SETTING_PARALLAX_MODE == 4
                    float upperDifference = candidateDifference;
                    #endif
                    bool bracketed = candidateDifference >= -tEpsilon;
                    if (!bracketed && previousDerivative > 0.0 && candidateDerivative < 0.0) {
                        float derivativeLower = previousSegment;
                        float derivativeUpper = candidateSegment;
                        for (int refinement = 0; refinement < 8; refinement++) {
                            float middleSegment = (derivativeLower + derivativeUpper) * 0.5;
                            vec2 middlePosition = localPosition + segmentDelta * middleSegment;
                            vec2 middleGradient = _parallax_continuousParallaxSurface(depths, middlePosition).yz;
                            float middleDerivative = segmentLength - dot(middleGradient, segmentDelta);
                            if (middleDerivative > 0.0) {
                                derivativeLower = middleSegment;
                            } else {
                                derivativeUpper = middleSegment;
                            }
                        }
                        upperSegment = (derivativeLower + derivativeUpper) * 0.5;
                        vec2 peakPosition = localPosition + segmentDelta * upperSegment;
                        float peakDepth = _parallax_continuousParallaxSurface(depths, peakPosition).x;
                        float peakDifference = t + segmentLength * upperSegment - peakDepth;
                        #if SETTING_PARALLAX_MODE == 4
                        upperDifference = peakDifference;
                        #endif
                        bracketed = peakDifference >= -tEpsilon;
                    }
                    if (bracketed) {
                        float lowerSegment = previousSegment;
                        #if SETTING_PARALLAX_MODE == 4
                        float lowerDifference = previousDifference;
                        for (int refinement = 0; refinement < 5; refinement++) {
                            float middleSegment = (lowerSegment + upperSegment) * 0.5;
                            vec2 middlePosition = localPosition + segmentDelta * middleSegment;
                            float middleDepth = _parallax_continuousParallaxSurface(depths, middlePosition).x;
                            float middleDifference = t + segmentLength * middleSegment - middleDepth;
                            if (middleDifference >= -tEpsilon) {
                                upperSegment = middleSegment;
                                upperDifference = middleDifference;
                            } else {
                                lowerSegment = middleSegment;
                                lowerDifference = middleDifference;
                            }
                        }
                        float rootWeight = (-tEpsilon - lowerDifference) / (upperDifference - lowerDifference);
                        hitSegment = mix(lowerSegment, upperSegment, rootWeight);
                        #else
                        for (int refinement = 0; refinement < 6; refinement++) {
                            float middleSegment = (lowerSegment + upperSegment) * 0.5;
                            vec2 middlePosition = localPosition + segmentDelta * middleSegment;
                            float middleDepth = _parallax_continuousParallaxSurface(depths, middlePosition).x;
                            float middleDifference = t + segmentLength * middleSegment - middleDepth;
                            if (middleDifference >= -tEpsilon) {
                                upperSegment = middleSegment;
                            } else {
                                lowerSegment = middleSegment;
                            }
                        }
                        hitSegment = upperSegment;
                        #endif
                        break;
                    }
                    previousSegment = candidateSegment;
                    #if SETTING_PARALLAX_MODE == 4
                    previousDifference = candidateDifference;
                    #endif
                    previousDerivative = candidateDerivative;
                }
            }
            leafHit = hitSegment <= 1.0;
            if (leafHit) {
                hitT = t + segmentLength * hitSegment;
                vec2 hitPosition = localPosition + segmentDelta * hitSegment;
                vec2 depthGradient = _parallax_continuousParallaxSurface(depths, hitPosition).yz;
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
            #if SETTING_PARALLAX_MODE == 1
            float surfaceDepth = 1.0 - _parallax_materialDepthMaxAlpha(cell, mipData);
            #else
            ivec2 mipCellMin = spriteTexelMin >> level;
            ivec2 mipCellMax = (spriteTexelMax + cellScaleI - 1) >> level;
            #if SETTING_PARALLAX_MODE == 4
            float maxSurfaceAlpha = 0.0;
            for (int cellY = -1; cellY <= 1; cellY++) {
                for (int cellX = -1; cellX <= 1; cellX++) {
                    ivec2 wrappedCell = _parallax_wrapParallaxCell(cell + ivec2(cellX, cellY), mipCellMin, mipCellMax);
                    maxSurfaceAlpha = max(maxSurfaceAlpha, _parallax_materialDepthMaxAlpha(wrappedCell, mipData));
                }
            }
            #else
            float maxSurfaceAlpha;
            if (all(lessThan(cell + ivec2(1), mipCellMax))) {
                vec4 gatheredAlpha = _parallax_gatherMaterialDepthAlpha(cell, mipData, packedTexelRcp);
                maxSurfaceAlpha = max(max(gatheredAlpha.x, gatheredAlpha.y), max(gatheredAlpha.z, gatheredAlpha.w));
            } else {
                ivec2 cellX = _parallax_wrapParallaxCell(cell + ivec2(1, 0), mipCellMin, mipCellMax);
                ivec2 cellY = _parallax_wrapParallaxCell(cell + ivec2(0, 1), mipCellMin, mipCellMax);
                ivec2 cellXY = _parallax_wrapParallaxCell(cell + ivec2(1), mipCellMin, mipCellMax);
                maxSurfaceAlpha = _parallax_materialDepthMaxAlpha(cell, mipData);
                maxSurfaceAlpha = max(maxSurfaceAlpha, _parallax_materialDepthMaxAlpha(cellX, mipData));
                maxSurfaceAlpha = max(maxSurfaceAlpha, _parallax_materialDepthMaxAlpha(cellY, mipData));
                maxSurfaceAlpha = max(maxSurfaceAlpha, _parallax_materialDepthMaxAlpha(cellXY, mipData));
            }
            #endif
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
        #if SETTING_PARALLAX_MODE == 1
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

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

vec4 _parallax_smoothstepLineCoefficients(float origin, float delta) {
    float deltaSquared = delta * delta;
    return vec4(
        origin * origin * (3.0 - 2.0 * origin),
        6.0 * origin * (1.0 - origin) * delta,
        3.0 * (1.0 - 2.0 * origin) * deltaSquared,
        -2.0 * deltaSquared * delta
    );
}

mat2x4 _parallax_continuousLineCoefficients(vec4 coefficients, vec2 origin, vec2 delta) {
    vec4 x = _parallax_smoothstepLineCoefficients(origin.x, delta.x);
    vec4 y = _parallax_smoothstepLineCoefficients(origin.y, delta.y);
    return mat2x4(
        vec4(
            coefficients.x + coefficients.y * x.x + coefficients.z * y.x + coefficients.w * x.x * y.x,
            coefficients.y * x.y + coefficients.z * y.y + coefficients.w * (x.x * y.y + x.y * y.x),
            coefficients.y * x.z + coefficients.z * y.z
                + coefficients.w * (x.x * y.z + x.y * y.y + x.z * y.x),
            coefficients.y * x.w + coefficients.z * y.w
                + coefficients.w * (x.x * y.w + x.y * y.z + x.z * y.y + x.w * y.x)
        ),
        vec4(
            coefficients.w * (x.y * y.w + x.z * y.z + x.w * y.y),
            coefficients.w * (x.z * y.w + x.w * y.z),
            coefficients.w * x.w * y.w,
            0.0
        )
    );
}

vec2 _parallax_continuousLineSurface(mat2x4 coefficients, float position) {
    vec4 low = coefficients[0];
    vec4 high = coefficients[1];
    float depth = (((((high.z * position + high.y) * position + high.x) * position
        + low.w) * position + low.z) * position + low.y) * position + low.x;
    float derivative = ((((6.0 * high.z * position + 5.0 * high.y) * position
        + 4.0 * high.x) * position + 3.0 * low.w) * position + 2.0 * low.z) * position + low.y;
    return vec2(depth, derivative);
}

float _parallax_continuousLineDepth(mat2x4 coefficients, float position) {
    vec4 low = coefficients[0];
    vec4 high = coefficients[1];
    return (((((high.z * position + high.y) * position + high.x) * position
        + low.w) * position + low.z) * position + low.y) * position + low.x;
}

float _parallax_continuousLineDerivative(mat2x4 coefficients, float position) {
    vec4 low = coefficients[0];
    vec4 high = coefficients[1];
    return ((((6.0 * high.z * position + 5.0 * high.y) * position
        + 4.0 * high.x) * position + 3.0 * low.w) * position + 2.0 * low.z) * position + low.y;
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

vec4 _parallax_cubicLineCoefficients(vec4 coefficients, float origin, float delta) {
    float originSquared = origin * origin;
    float deltaSquared = delta * delta;
    return vec4(
        ((coefficients.w * origin + coefficients.z) * origin + coefficients.y) * origin + coefficients.x,
        delta * ((3.0 * coefficients.w * origin + 2.0 * coefficients.z) * origin + coefficients.y),
        deltaSquared * (3.0 * coefficients.w * origin + coefficients.z),
        deltaSquared * delta * coefficients.w
    );
}

mat2x4 _parallax_bSplineLineCoefficients(mat4 coefficients, vec2 origin, vec2 delta) {
    vec4 polynomial3 = _parallax_cubicLineCoefficients(coefficients[3], origin.x, delta.x);
    vec4 polynomial2 = _parallax_cubicLineCoefficients(coefficients[2], origin.x, delta.x);
    vec4 polynomial1 = _parallax_cubicLineCoefficients(coefficients[1], origin.x, delta.x);
    vec4 polynomial0 = _parallax_cubicLineCoefficients(coefficients[0], origin.x, delta.x);

    vec4 degree4 = vec4(
        origin.y * polynomial3.x,
        origin.y * polynomial3.y + delta.y * polynomial3.x,
        origin.y * polynomial3.z + delta.y * polynomial3.y,
        origin.y * polynomial3.w + delta.y * polynomial3.z
    ) + polynomial2;
    float degree4High = delta.y * polynomial3.w;

    vec4 degree5 = vec4(
        origin.y * degree4.x,
        origin.y * degree4.y + delta.y * degree4.x,
        origin.y * degree4.z + delta.y * degree4.y,
        origin.y * degree4.w + delta.y * degree4.z
    ) + polynomial1;
    vec2 degree5High = vec2(
        origin.y * degree4High + delta.y * degree4.w,
        delta.y * degree4High
    );

    return mat2x4(
        vec4(
            origin.y * degree5.x,
            origin.y * degree5.y + delta.y * degree5.x,
            origin.y * degree5.z + delta.y * degree5.y,
            origin.y * degree5.w + delta.y * degree5.z
        ) + polynomial0,
        vec4(
            origin.y * degree5High.x + delta.y * degree5.w,
            origin.y * degree5High.y + delta.y * degree5High.x,
            delta.y * degree5High.y,
            0.0
        )
    );
}

vec2 _parallax_bSplineLineSurface(mat2x4 coefficients, float position) {
    vec4 low = coefficients[0];
    vec4 high = coefficients[1];
    float depth = (((((high.z * position + high.y) * position + high.x) * position
        + low.w) * position + low.z) * position + low.y) * position + low.x;
    float derivative = ((((6.0 * high.z * position + 5.0 * high.y) * position
        + 4.0 * high.x) * position + 3.0 * low.w) * position + 2.0 * low.z) * position + low.y;
    return vec2(depth, derivative);
}

float _parallax_bSplineLineDepth(mat2x4 coefficients, float position) {
    vec4 low = coefficients[0];
    vec4 high = coefficients[1];
    return (((((high.z * position + high.y) * position + high.x) * position
        + low.w) * position + low.z) * position + low.y) * position + low.x;
}

float _parallax_bSplineLineDerivative(mat2x4 coefficients, float position) {
    vec4 low = coefficients[0];
    vec4 high = coefficients[1];
    return ((((6.0 * high.z * position + 5.0 * high.y) * position
        + 4.0 * high.x) * position + 3.0 * low.w) * position + 2.0 * low.z) * position + low.y;
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
            mat2x4 lineDepths = _parallax_continuousLineCoefficients(depths, localPosition, segmentDelta);
            #elif SETTING_PARALLAX_MODE == 4
            mat4 depths = _parallax_bSplineCoefficients(depthSamples);
            mat2x4 lineDepths = _parallax_bSplineLineCoefficients(depths, localPosition, segmentDelta);
            #endif
            float hitSegment = 2.0;
            float previousSegment = 0.0;
            #if SETTING_PARALLAX_MODE == 4
            vec2 startLineSurface = _parallax_bSplineLineSurface(lineDepths, 0.0);
            float startDifference = t - startLineSurface.x;
            float previousDifference = startDifference;
            float previousDerivative = segmentLength - startLineSurface.y;
            #elif SETTING_PARALLAX_MODE == 3
            vec2 startLineSurface = _parallax_continuousLineSurface(lineDepths, 0.0);
            float startDifference = t - startLineSurface.x;
            float previousDerivative = segmentLength - startLineSurface.y;
            #else
            vec3 startSurface = _parallax_continuousParallaxSurface(depths, localPosition);
            float startDifference = t - startSurface.x;
            float previousDerivative = segmentLength - dot(startSurface.yz, segmentDelta);
            #endif
            if (startDifference >= -tEpsilon) {
                hitSegment = 0.0;
            } else {
                for (int step = 1; step <= 8; step++) {
                    float candidateSegment = float(step) * 0.125;
                    #if SETTING_PARALLAX_MODE == 4
                    vec2 candidateSurface = _parallax_bSplineLineSurface(lineDepths, candidateSegment);
                    float candidateDifference = t + segmentLength * candidateSegment - candidateSurface.x;
                    float candidateDerivative = segmentLength - candidateSurface.y;
                    #elif SETTING_PARALLAX_MODE == 3
                    vec2 candidateSurface = _parallax_continuousLineSurface(lineDepths, candidateSegment);
                    float candidateDifference = t + segmentLength * candidateSegment - candidateSurface.x;
                    float candidateDerivative = segmentLength - candidateSurface.y;
                    #else
                    vec2 candidatePosition = localPosition + segmentDelta * candidateSegment;
                    vec3 candidateSurface = _parallax_continuousParallaxSurface(depths, candidatePosition);
                    float candidateDifference = t + segmentLength * candidateSegment - candidateSurface.x;
                    float candidateDerivative = segmentLength - dot(candidateSurface.yz, segmentDelta);
                    #endif
                    float upperSegment = candidateSegment;
                    #if SETTING_PARALLAX_MODE == 4
                    float upperDifference = candidateDifference;
                    #endif
                    bool bracketed = candidateDifference >= -tEpsilon;
                    if (!bracketed && previousDerivative > 0.0 && candidateDerivative < 0.0) {
                        float derivativeLower = previousSegment;
                        float derivativeUpper = candidateSegment;
                        float derivativeLowerValue = previousDerivative;
                        float derivativeUpperValue = candidateDerivative;
                        for (int refinement = 0; refinement < 4; refinement++) {
                            float middleSegment = (derivativeLower + derivativeUpper) * 0.5;
                            #if SETTING_PARALLAX_MODE == 4
                            float middleDerivative = segmentLength
                                - _parallax_bSplineLineDerivative(lineDepths, middleSegment);
                            #elif SETTING_PARALLAX_MODE == 3
                            float middleDerivative = segmentLength
                                - _parallax_continuousLineDerivative(lineDepths, middleSegment);
                            #else
                            vec2 middlePosition = localPosition + segmentDelta * middleSegment;
                            vec2 middleGradient = _parallax_continuousParallaxSurface(depths, middlePosition).yz;
                            float middleDerivative = segmentLength - dot(middleGradient, segmentDelta);
                            #endif
                            if (middleDerivative > 0.0) {
                                derivativeLower = middleSegment;
                                derivativeLowerValue = middleDerivative;
                            } else {
                                derivativeUpper = middleSegment;
                                derivativeUpperValue = middleDerivative;
                            }
                        }
                        float peakWeight = derivativeLowerValue / (derivativeLowerValue - derivativeUpperValue);
                        upperSegment = mix(derivativeLower, derivativeUpper, peakWeight);
                        #if SETTING_PARALLAX_MODE == 4
                        float peakDepth = _parallax_bSplineLineDepth(lineDepths, upperSegment);
                        #elif SETTING_PARALLAX_MODE == 3
                        float peakDepth = _parallax_continuousLineDepth(lineDepths, upperSegment);
                        #else
                        vec2 peakPosition = localPosition + segmentDelta * upperSegment;
                        float peakDepth = _parallax_continuousParallaxSurface(depths, peakPosition).x;
                        #endif
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
                        for (int refinement = 0; refinement < 3; refinement++) {
                            float middleSegment = (lowerSegment + upperSegment) * 0.5;
                            float middleDepth = _parallax_bSplineLineDepth(lineDepths, middleSegment);
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
                        #elif SETTING_PARALLAX_MODE == 3
                        for (int refinement = 0; refinement < 6; refinement++) {
                            float middleSegment = (lowerSegment + upperSegment) * 0.5;
                            float middleDepth = _parallax_continuousLineDepth(lineDepths, middleSegment);
                            float middleDifference = t + segmentLength * middleSegment - middleDepth;
                            if (middleDifference >= -tEpsilon) {
                                upperSegment = middleSegment;
                            } else {
                                lowerSegment = middleSegment;
                            }
                        }
                        hitSegment = upperSegment;
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
            #if SETTING_PARALLAX_MODE == 4
            ivec2 mipCellMin = spriteTexelMin >> level;
            ivec2 mipCellMax = (spriteTexelMax + cellScaleI - 1) >> level;
            bool interiorCell = all(greaterThan(cell, mipCellMin))
                && all(lessThan(cell + ivec2(1), mipCellMax));
            float maxSurfaceAlpha = interiorCell ? _parallax_materialDepthMaxAlpha(cell, mipData) : 1.0;
            #else
            ivec2 mipCellMax = (spriteTexelMax + cellScaleI - 1) >> level;
            bool interiorCell = all(lessThan(cell + ivec2(1), mipCellMax));
            float maxSurfaceAlpha = interiorCell ? _parallax_materialDepthMaxAlpha(cell, mipData) : 1.0;
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

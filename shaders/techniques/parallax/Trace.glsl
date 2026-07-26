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

float materialDepthAxisExit(float origin, float delta, float lower, float upper) {
    if (delta > 0.0) {
        return (upper - origin) / delta;
    }
    if (delta < 0.0) {
        return (lower - origin) / delta;
    }
    return 1.0;
}


bool traceSteepParallax(
    vec2 atlasTexCoord,
    vec4 spriteBounds,
    vec2 rayDeltaTexels,
    out vec2 hitTexCoord,
    out float hitT,
    out vec3 hitTangentNormal
) {
    const float texelEpsilon = 1e-3;
    float tEpsilon = texelEpsilon / max(max(abs(rayDeltaTexels.x), abs(rayDeltaTexels.y)), 1.0);
    ivec2 atlasSizeI = textureSize(usam_blocksNormal, 0);
    vec2 atlasSize = vec2(atlasSizeI);
    vec2 spriteMin = clamp(spriteBounds.xy * atlasSize, vec2(0.0), atlasSize);
    vec2 spriteMax = clamp(spriteBounds.zw * atlasSize, spriteMin + texelEpsilon, atlasSize);
    vec2 rayStart = clamp(atlasTexCoord * atlasSize, spriteMin + texelEpsilon, spriteMax - texelEpsilon);

    float rayMaxT = 1.0;

    int maxExtent = max(1, int(ceil(max(spriteMax.x - spriteMin.x, spriteMax.y - spriteMin.y))));
    int startLevel = min(14, findMSB(maxExtent - 1) + 1);
    int level = startLevel;
    float t = 0.0;
    float entryT = 0.0;
    vec2 entryNormal = vec2(0.0);

    while (t <= rayMaxT) {
        vec2 rawPosition = rayStart + rayDeltaTexels * t;
        vec2 position = clamp(rawPosition, spriteMin + texelEpsilon, spriteMax - texelEpsilon);
        vec2 activeDelta = rayDeltaTexels;
        if ((activeDelta.x < 0.0 && rawPosition.x <= spriteMin.x) || (activeDelta.x > 0.0 && rawPosition.x >= spriteMax.x)) {
            activeDelta.x = 0.0;
        }
        if ((activeDelta.y < 0.0 && rawPosition.y <= spriteMin.y) || (activeDelta.y > 0.0 && rawPosition.y >= spriteMax.y)) {
            activeDelta.y = 0.0;
        }
        vec2 samplePosition = clamp(position + sign(activeDelta) * texelEpsilon, spriteMin + texelEpsilon, spriteMax - texelEpsilon);
        int cellScaleI = 1 << level;
        float cellScale = float(cellScaleI);
        ivec2 cell = ivec2(floor(samplePosition / cellScale));
        vec2 cellMin = max(vec2(cell * cellScaleI), spriteMin);
        vec2 cellMax = min(vec2((cell + 1) * cellScaleI), spriteMax);

        float tExitX = materialDepthAxisExit(rayStart.x, activeDelta.x, cellMin.x, cellMax.x);
        float tExitY = materialDepthAxisExit(rayStart.y, activeDelta.y, cellMin.y, cellMax.y);
        float tExit = min(rayMaxT, min(tExitX, tExitY));
        float maxAlpha = materialDepthMaxAlpha(cell, level, atlasSizeI);
        float minSurfaceDepth = 1.0 - maxAlpha;

        if (level == 0) {
            float alpha = maxAlpha;
            float surfaceDepth = 1.0 - alpha;
            bool sideHit = dot(entryNormal, entryNormal) > 0.0 && surfaceDepth + tEpsilon < entryT;
            float candidateT = max(t, surfaceDepth);
            if (candidateT <= tExit + tEpsilon) {
                hitT = sideHit ? entryT : candidateT;
                vec2 hitTexel = clamp(rayStart + rayDeltaTexels * hitT, spriteMin + texelEpsilon, spriteMax - texelEpsilon);
                hitTexCoord = hitTexel / atlasSize;
                hitTangentNormal = sideHit ? normalize(vec3(entryNormal, 0.0)) : vec3(0.0, 0.0, 1.0);
                return true;
            }
        } else if (tExit + tEpsilon >= minSurfaceDepth) {
            level -= 1;
            continue;
        }

        if (tExit >= rayMaxT - tEpsilon) {
            break;
        }
        float nextT = max(tExit + tEpsilon, t + tEpsilon);
        if (nextT > rayMaxT) {
            break;
        }
        entryT = tExit;
        entryNormal = vec2(0.0);
        if (activeDelta.x != 0.0 && tExitX <= tExitY) {
            entryNormal.x = -sign(rayDeltaTexels.x);
        } else if (activeDelta.y != 0.0) {
            entryNormal.y = -sign(rayDeltaTexels.y);
        }
        t = nextT;
        level += 1;
        level = min(startLevel, level);
    }

    hitTexCoord = atlasTexCoord;
    hitT = 0.0;
    hitTangentNormal = vec3(0.0, 0.0, 1.0);
    return false;
}

#endif

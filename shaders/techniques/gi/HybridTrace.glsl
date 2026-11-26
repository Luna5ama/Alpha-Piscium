#ifndef INCLUDE_techniques_gi_HybridTrace_glsl
#define INCLUDE_techniques_gi_HybridTrace_glsl a

#include "/util/GBufferData.glsl"
#include "/techniques/voxel/VoxelTraversal.glsl"

#if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
#include "/util/VoxyFaceNormal.glsl"
vec3 gi_wideNormal(uint code) {
    if (code >= 65536u) return voxy_faceWorldNormal(code & 7u);
    vec3 n, tangent;
    nzpacking_unpackNormalOct16(code, n, tangent);
    return n;
}
vec3 gi_retainedContactNormal;
uint gi_localEmission;
#endif

const float GI_SCREEN_NEAR_DISTANCE = 4.0;
const float GI_RAY_END = 1e20;
const int GI_SCREEN_CLEAR = 0;
const int GI_SCREEN_HIT = 1;
const int GI_SCREEN_EXHAUSTED = 2;
const int GI_SCREEN_UNCOVERED = 3;

// Clip t in original ray units against a half-space with signed distance a + bt.
bool gi_clipPlane(float a, float b, inout vec2 interval) {
    if (b == 0.0) return a >= 0.0;
    float t = -a / b;
    if (b > 0.0) interval.x = max(interval.x, t);
    else interval.y = min(interval.y, t);
    return interval.x < interval.y;
}

bool gi_gridInterval(vec3 worldPos, vec3 dir, inout vec2 interval) {
    vec3 p = worldPos - _voxel_gridOriginF;
    vec3 q = vec3(VOXEL_GRID_SIZE * VOXEL_BRICK_SIZE) - p;
    if ((dir.x == 0.0 && q.x <= 0.0) || (dir.y == 0.0 && q.y <= 0.0) || (dir.z == 0.0 && q.z <= 0.0)) return false;
    return gi_clipPlane(p.x, dir.x, interval) && gi_clipPlane(q.x, -dir.x, interval)
        && gi_clipPlane(p.y, dir.y, interval) && gi_clipPlane(q.y, -dir.y, interval)
        && gi_clipPlane(p.z, dir.z, interval) && gi_clipPlane(q.z, -dir.z, interval);
}

bool gi_projectionInterval(vec3 p, vec3 d, inout vec2 interval, mat4 proj, vec2 jitter) {
    vec4 a = proj * vec4(p, 1.0);
    vec4 b = proj * vec4(d, 0.0);
    a.xy += 2.0 * jitter * a.w;
    b.xy += 2.0 * jitter * b.w;
    return gi_clipPlane(-nearPlane - p.z, -d.z, interval)
        && gi_clipPlane(65536.0 + p.z, d.z, interval)
        && gi_clipPlane(a.w + a.x, b.w + b.x, interval)
        && gi_clipPlane(a.w - a.x, b.w - b.x, interval)
        && gi_clipPlane(a.w + a.y, b.w + b.y, interval)
        && gi_clipPlane(a.w - a.y, b.w - b.y, interval);
}

vec3 gi_projectRaster(vec3 p, mat4 proj, vec2 jitter) {
    vec2 uv = coords_viewToScreen(p, proj).xy + jitter;
    return vec3(uv * uval_mainImageSize, near / -p.z);
}

vec3 gi_unprojectSurface(ivec2 pixel, float z, mat4 inverseProj, vec2 jitter) {
    vec2 uv = coords_texelToUV(pixel, uval_mainImageSizeRcp) - jitter;
    return coords_toViewCoord(uv, z, inverseProj);
}

bool gi_screenInterval(vec3 p, vec3 d, inout vec2 span) {
    return gi_projectionInterval(p, d, span, global_camProj, uval_taaJitterUV);
}

vec3 gi_rasterPos(vec3 p) { return gi_projectRaster(p, global_camProj, uval_taaJitterUV); }

vec3 gi_screenSurface(ivec2 pixel, float z) {
    return gi_unprojectSurface(pixel, z, global_camProjInverse, uval_taaJitterUV);
}

// Hi-Z bounds select cells. The pixel's depth and geometric plane decide the hit.
int gi_traceProjection(vec3 p, vec3 d, vec2 segment, bool requireCoverage,
                   mat4 proj, mat4 inverseProj, vec2 jitter, bool wide,
                   out float hitT, out ivec2 hitPixel) {
    hitT = GI_RAY_END;
    hitPixel = ivec2(-1);
    if (segment.y <= segment.x) return GI_SCREEN_CLEAR;
    vec2 span = segment;
    bool covered = gi_projectionInterval(p, d, span, proj, jitter);
    if (requireCoverage && (!covered || any(notEqual(span, segment)))) return GI_SCREEN_UNCOVERED;
    if (!covered) return GI_SCREEN_CLEAR;

    vec3 a = gi_projectRaster(fma(d, vec3(span.x), p), proj, jitter);
    vec3 delta = gi_projectRaster(fma(d, vec3(span.y), p), proj, jitter) - a;
    vec2 stepDir = sign(delta.xy);
    int level = 0;
    int maxLevel = min(findMSB(min(uval_mainImageSizeI.x, uval_mainImageSizeI.y)), 12);
    float u = 0.0;
    for (int stepIndex = 0; stepIndex < 128; ++stepIndex) {
        vec2 xy = fma(delta.xy, vec2(u), a.xy);
        float size = float(1 << level);
        // A boundary belongs to the cell ahead in the direction of traversal.
        vec2 cellXY = mix(floor(xy / size), ceil(xy / size) - 1.0, lessThan(stepDir, vec2(0.0)));
        ivec4 tile = global_hizTiles[level];
        ivec2 cell = clamp(ivec2(cellXY), ivec2(0), tile.zw - 1);
        vec2 edge = (vec2(cell) + vec2(greaterThan(stepDir, vec2(0.0)))) * size;
        vec2 crossU = mix((edge - a.xy) / delta.xy, vec2(2.0), equal(stepDir, vec2(0.0)));
        float endU = min(1.0, min(crossU.x, crossU.y));
        vec2 depth;
        #if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
        if (wide) {
            ivec2 q = tile.xy + cell + ivec2(global_hizTiles[1].z, 0);
            // At leaves intersect the actual plane. Odd render sizes may also have
            // a final mip column outside the half-width texture; descend there.
            depth = vec2(FLT_MAX, 0.0);
            if (level > 0 && q.x < uval_mainImageSizeI.x) {
                depth = texelFetch(usam_hiz, q, 0).rg;
            }
        } else
        #endif
        depth = texelFetch(usam_hiz, tile.xy + cell, 0).ba;
        vec2 z = fma(vec2(u, endU), vec2(delta.z), vec2(a.z));
        bool overlap = max(z.x, z.y) >= depth.y && min(z.x, z.y) <= depth.x;
        if (overlap && level > 0) {
            --level;
            continue;
        }
        if (overlap) {
            float viewZ;
            #if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
            vec2 wideSurface;
            if (wide) {
                wideSurface = texelFetch(usam_hiz, cell + ivec2(0, 2 * uval_mainImageSizeI.y), 0).xy;
                viewZ = wideSurface.x;
                if (!(viewZ < -nearPlane)) return requireCoverage ? GI_SCREEN_UNCOVERED : GI_SCREEN_CLEAR;
            } else
            #endif
            viewZ = texelFetch(usam_gbufferSolidViewZ, cell, 0).r;
            bool originSurface = viewZ == p.z && p.z <= -nearPlane
                && all(equal(ivec2(floor(gi_projectRaster(p, proj, jitter).xy)), cell));
            if (viewZ > -65536.0 && !originSurface) {
                GBufferData data = gbufferData_init();
                #if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
                if (wide) {
                    data.geomNormal = gi_wideNormal(uint(wideSurface.y));
                    data.geomNormal = mat3(global_giWideView) * data.geomNormal;
                } else
                #endif
                gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, cell, 0), data);
                float cosTheta = dot(data.geomNormal, d);
                if (abs(cosTheta) > 1e-5) {
                    float t = dot(gi_unprojectSurface(cell, viewZ, inverseProj, jitter) - p, data.geomNormal) / cosTheta;
                    if (t > 0.0 && t >= span.x && t <= span.y) {
                        vec2 raster = gi_projectRaster(fma(d, vec3(t), p), proj, jitter).xy;
                        if (all(equal(ivec2(floor(raster)), cell))) {
                            hitT = t;
                            hitPixel = cell;
                            return GI_SCREEN_HIT;
                        }
                    }
                }
            }
        }
        if (endU >= 1.0) return GI_SCREEN_CLEAR;
        // Step one projected-coordinate ULP past the boundary shared by adjacent cells.
        vec2 pixelUlp = max(abs(edge), vec2(1.0)) * 1.2e-7;
        vec2 du = pixelUlp / max(abs(delta.xy), vec2(1e-30));
        u = max(uintBitsToFloat(floatBitsToUint(u) + 1u), endU + (crossU.x <= crossU.y ? du.x : du.y));
        if (u >= 1.0) return GI_SCREEN_CLEAR;
        level = min(level + 1, maxLevel);
    }
    return GI_SCREEN_EXHAUSTED;
}

int gi_traceScreen(vec3 p, vec3 d, vec2 segment, bool requireCoverage, out float t, out ivec2 pixel) {
    return gi_traceProjection(p, d, segment, requireCoverage, global_camProj, global_camProjInverse,
                              uval_taaJitterUV, false, t, pixel);
}

#if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
#include "/techniques/gi/ObservedBlockVisibility.glsl"
int gi_traceWide(vec3 p, vec3 d, vec2 segment, bool requireCoverage, out float t, out ivec2 pixel) {
    vec3 eyeDelta = vec3(cameraPositionInt - global_giWideEyeInt) + cameraPositionFract - global_giWideEyeFract;
    vec3 sceneP = (gbufferModelViewInverse * vec4(p, 1.0)).xyz + eyeDelta;
    vec3 wideP = (global_giWideView * vec4(sceneP, 1.0)).xyz;
    vec3 wideD = mat3(global_giWideView) * coords_dir_viewToWorld(d);
    int result = gi_traceProjection(wideP, wideD, segment, requireCoverage, global_giWideProj,
                              global_giWideInvProj, global_giWideJitter, true, t, pixel);
    if (result == GI_SCREEN_HIT || (requireCoverage && result == GI_SCREEN_CLEAR)) {
        vec3 local = vec3(cameraPositionInt - global_giModelOrigin) + cameraPositionFract
                   + (gbufferModelViewInverse * vec4(p, 1.0)).xyz;
        vec3 dir = coords_dir_viewToWorld(d);
        ivec3 endpoint = ivec3(-1);
        if (result == GI_SCREEN_HIT) {
            vec4 sampleData = texelFetch(usam_hiz, pixel + ivec2(0, 2 * uval_mainImageSizeI.y), 0);
            uint code = uint(sampleData.y);
            vec3 n = gi_wideNormal(code);
            gi_retainedContactNormal = n;
            endpoint = ivec3(floor(fma(dir, vec3(t), local) - n * 0.0625));
            // A magnified wide pixel can straddle a lamp and the empty cell next
            // to it. Intersect the sampled block model before admitting emission.
            if (code >= 65536u && floatBitsToUint(sampleData.w) != 0x00800000u) {
                vec3 sampledView = gi_unprojectSurface(pixel, sampleData.x, global_giWideInvProj, vec2(0.0));
                vec3 sampled = vec3(global_giWideEyeInt - global_giModelOrigin) + global_giWideEyeFract
                    + transpose(mat3(global_giWideView)) * (sampledView - global_giWideView[3].xyz);
                uint source = (code - 65536u) >> 3u;
                if (!gi_sourceContact(local, dir, segment, sampled, source, n, endpoint, t, gi_retainedContactNormal)) return GI_SCREEN_UNCOVERED;
            }
        }
        if (gi_observedBlockOccludes(local, dir, result == GI_SCREEN_HIT ? t : segment.y, endpoint)) {
            return GI_SCREEN_UNCOVERED;
        }
    }
    return result;
}
#endif

// Current pixels have precedence over retained pixels on every overlapping interval.
int gi_traceTerrainProjections(vec3 p, vec3 d, vec2 segment, bool requireCoverage,
                    out float t, out ivec2 pixel, out bool retained) {
    retained = false;
    #if defined(VOXY) && defined(GI_RETAINED_PROJECTION)
    if (global_giWideState == 3 && segment.y > segment.x) {
        vec2 live = segment;
        bool overlaps = gi_screenInterval(p, d, live);
        if (!overlaps) {
            retained = true;
            return gi_traceWide(p, d, segment, requireCoverage, t, pixel);
        }
        if (live.x > segment.x) {
            int status = gi_traceWide(p, d, vec2(segment.x, live.x), requireCoverage, t, pixel);
            if (status != GI_SCREEN_CLEAR) {
                retained = true;
                return status;
            }
        }
        int status = gi_traceScreen(p, d, live, false, t, pixel);
        #ifdef RESTIR_INITIAL_CANDIDATE_WRITE
        if (status == GI_SCREEN_HIT) {
            uvec4 raw = texelFetch(usam_gbufferVoxySolidData, pixel, 0);
            float z = texelFetch(usam_gbufferSolidViewZ, pixel, 0).r;
            if (raw.w == floatBitsToUint(z) && gi_currentHitEmits(pixel)) {
                vec3 eye = vec3(cameraPositionInt - global_giModelOrigin) + cameraPositionFract;
                vec3 local = eye + (gbufferModelViewInverse * vec4(p, 1.0)).xyz;
                vec3 dir = coords_dir_viewToWorld(d);
                vec3 sampledView = gi_unprojectSurface(pixel, z, global_camProjInverse, vec2(0.0));
                vec3 sampled = eye + (gbufferModelViewInverse * vec4(sampledView, 1.0)).xyz;
                vec3 n = voxy_faceWorldNormal(bitfieldExtract(raw.y, 24, 3));
                ivec3 endpoint = ivec3(floor(fma(dir, vec3(t), local) - n * 0.0625));
                vec3 contactNormal;
                if (!gi_sourceContact(local, dir, live, sampled, raw.z >> 16u, n, endpoint, t, contactNormal)
                    || gi_observedBlockOccludes(local, dir, t, endpoint)) return GI_SCREEN_UNCOVERED;
            }
        }
        #endif
        if (status != GI_SCREEN_CLEAR || live.y >= segment.y) return status;
        retained = true;
        return gi_traceWide(p, d, vec2(live.y, segment.y), requireCoverage, t, pixel);
    }
    #endif
    return gi_traceScreen(p, d, segment, requireCoverage, t, pixel);
}

int gi_traceTerrain(vec3 p, vec3 d, vec2 segment, bool requireCoverage,
                    out float t, out ivec2 pixel, out bool retained) {
    #if defined(VOXY) && defined(GI_RETAINED_PROJECTION) && defined(RESTIR_INITIAL_CANDIDATE_WRITE) && SETTING_PBR_MATERIAL == 0
    bool hasSource = false;
    uint emission;
    vec3 normal;
    float sourceT;
    if (!requireCoverage && global_giLightEpoch == global_giModelEpoch) {
        vec3 eye = vec3(cameraPositionInt - global_giModelOrigin) + cameraPositionFract;
        vec3 local = eye + (gbufferModelViewInverse * vec4(p, 1.0)).xyz;
        vec3 dir = coords_dir_viewToWorld(d);
        ivec3 source;
        hasSource = gi_observedSource(local, dir, segment, source, sourceT, normal, emission)
                 && !gi_observedBlockOccludes(local, dir, sourceT, source);
        if (hasSource) {
            segment.y = max(segment.x, sourceT - max(0.0001, 8e-6 * sourceT));
            requireCoverage = true;
        }
    }
    #endif
    int status = gi_traceTerrainProjections(p, d, segment, requireCoverage, t, pixel, retained);
    #if defined(VOXY) && defined(GI_RETAINED_PROJECTION) && defined(RESTIR_INITIAL_CANDIDATE_WRITE) && SETTING_PBR_MATERIAL == 0
    if (hasSource && status == GI_SCREEN_CLEAR) {
        t = sourceT;
        pixel = ivec2(-2);
        retained = true;
        gi_retainedContactNormal = normal;
        gi_localEmission = emission;
        return GI_SCREEN_HIT;
    }
    #endif
    return status;
}

#endif

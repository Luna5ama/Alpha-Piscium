#if defined(MC_GL_VENDOR_NVIDIA)
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_NV_shader_subgroup_partitioned : enable
#endif

#include "/util/Coords.glsl"
#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"
#include "/util/Fresnel.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(128, 128, 1);

layout(r32i) uniform iimage2D uimg_causticsPhoton;

shared mat4 shared_camProj;
shared mat4 shared_shadowNDCToShadowView;
shared mat4 shared_shadowViewToScene;
shared uint shared_frustumPlaneCount;
shared vec4 shared_frustrumPlanes[6];

void initSharedData() {
    if (gl_LocalInvocationIndex < 6){
        shared_frustrumPlanes[gl_LocalInvocationIndex] = global_cameraData.frustumPlanes[gl_LocalInvocationIndex];
        if (gl_LocalInvocationIndex == 0) {
            shared_camProj = global_camProj;
            shared_frustumPlaneCount = global_cameraData.frustumPlaneCount;

            shared_shadowNDCToShadowView = global_shadowRotationMatrixInverse * global_shadowProjInversePrev;
            shared_shadowViewToScene = global_shadowViewInverse;
        }
    }
}

bool intersectViewFrustumFromProj(vec3 ro, vec3 rd, out vec3 hitPos) {
    float tHit = 0.0;
    hitPos = ro;


    const float EPS = 1e-6;
    // Quick inside test: if origin already inside all planes, start at origin
    bool inside = true;
    for (int i = 0; i < shared_frustumPlaneCount; ++i) {
        vec4 plane = shared_frustrumPlanes[i];
        float val = dot(plane.xyz, ro) + plane.w;
        if (val < -EPS) {
            inside = false;
            break;
        }
    }

    if (inside) {
        return true;
    }

    // Clip ray against the convex polyhedron: find t interval where plane(p(t)) >= 0
    float t0 = -1e20;// lower bound
    float t1 =  1e20;// upper bound

    for (int i = 0; i < shared_frustumPlaneCount; ++i) {
        vec4 plane = shared_frustrumPlanes[i];
        vec3 n = plane.xyz;
        float d = plane.w;

        float num   = dot(n, ro) + d;// plane(ro)
        float denom = dot(n, rd);// plane(rd)

        if (abs(denom) < EPS) {
            // Ray parallel to plane
            if (num < 0.0) {
                // Outside and parallel -> miss
                return false;
            } else {
                // Inside this plane for all t -> no constraint
                continue;
            }
        }

        float t = -num / denom;
        if (denom > 0.0) {
            // s(t) = num + denom*t; denom>0 => s >= 0 for t >= t -> lower bound (entering)
            t0 = max(t0, t);
        } else {
            // denom < 0 => s >=0 for t <= t -> upper bound (exiting)
            t1 = min(t1, t);
        }

        // early reject
        if (t0 > t1) return false;
    }

    // At this point we have an interval [t0, t1] where ray is inside. Need t1 >= 0.
    if (t1 < 0.0) return false;

    tHit = max(t0, 0.0);
    hitPos = ro + rd * tHit;
    return true;
}

void main() {
    initSharedData();

    ivec2 texelPosWarped = ivec2(gl_GlobalInvocationID.xy);
    vec2 jitter = rand_stbnVec2(texelPosWarped, frameCounter);
    vec2 screenPosWarped = (vec2(texelPosWarped) + jitter) / 1.0 / SHADOW_MAP_SIZE.x;
    vec4 waterNormalAndMast = texture(usam_shadow_waterNormal, screenPosWarped);
    barrier();

    if (waterNormalAndMast.a >= 0.9999) {
        vec2 screenPos = texture(usam_shadow_unwarpedUV, screenPosWarped).rg;
        float shadowDepth = texture(shadowtex0, screenPosWarped).r;
        vec3 shadowScreenPos = vec3(screenPos, shadowDepth);

        vec3 shadowNDCPos = shadowScreenPos * 2.0 - 1.0;
        vec4 shadowViewPos = shared_shadowNDCToShadowView * vec4(shadowNDCPos, 1.0);
        vec4 scenePos = shared_shadowViewToScene * shadowViewPos;

        vec3 waterNormal = normalize(waterNormalAndMast.xyz * 2.0 - 1.0);

        const float RIOR = AIR_IOR / WATER_IOR;
        vec3 L = uval_shadowLightDirWorld;
        L = rand_sampleInCone(L, SUN_ANGULAR_RADIUS, jitter);
        vec3 refractDir = refract(-L, waterNormal, RIOR);

        {
            vec4 rayTempOriginScene = scenePos;
            vec4 rayTempOriginView = gbufferModelView * rayTempOriginScene;
            rayTempOriginView.w /= rayTempOriginView.w;

            vec3 rayTempDirWorld = refractDir;
            vec3 rayTempDirView = normalize(mat3(gbufferModelView) * rayTempDirWorld);

            vec3 newOriginView;
            bool intersects = intersectViewFrustumFromProj(
                rayTempOriginView.xyz, rayTempDirView,
                newOriginView
            );

            rayTempOriginView.xyz = newOriginView;

            vec4 rayTempOriginClip = shared_camProj * rayTempOriginView;
            vec3 rayTempOriginNDC = rayTempOriginClip.xyz / rayTempOriginClip.w;
            vec3 rayTempOriginScreen = rayTempOriginNDC;
            rayTempOriginScreen.xy = rayTempOriginScreen.xy * 0.5 + 0.5;

            vec4 rayTempEndClip = shared_camProj * vec4(rayTempOriginView.xyz + rayTempDirView, 1.0);
            vec3 rayTempEndNDC = rayTempEndClip.xyz / rayTempEndClip.w;
            vec3 rayTempEndScreen = rayTempEndNDC;
            rayTempEndScreen.xy = rayTempEndScreen.xy * 0.5 + 0.5;

            vec3 pRayStart = rayTempOriginScreen;
            vec3 rayDirScreen = normalize(rayTempEndScreen - rayTempOriginScreen);
            vec3 rcpRayDirScreen = rcp(rayDirScreen);

            float maxT = 3.0;
            maxT = rayDirScreen.z != 0.0f ? min((float(rayDirScreen.z > 0.0f) - pRayStart.z) * rcpRayDirScreen.z, maxT) : maxT;
            maxT = rayDirScreen.x != 0.0f ? min((float(rayDirScreen.x > 0.0f) - pRayStart.x) * rcpRayDirScreen.x, maxT) : maxT;
//            maxT = rayDirScreen.y != 0.0f ? min((float(rayDirScreen.y > 0.0f) - pRayStart.y) * rcpRayDirScreen.y, maxT) : maxT;

            vec3 pRayVector = rayDirScreen * maxT;


            if (intersects) {
                vec2 startTexelF = pRayStart.xy * uval_mainImageSize;
                float startDepth = hiz_closest_sample(startTexelF, 0);
                if (pRayStart.z > startDepth) {
                    const uint SST_STEP = 8u;
                    float dt = 1.0 / float(SST_STEP - 2u);
                    float currT = 0.0;

                    float jitterStep = rand_stbnVec1(texelPosWarped, frameCounter);
                    bool isBackwardRay = pRayVector.z < 0.0;
                    const float maxThickness = 0.5;
                    float maxThicknessFactor = rcp(1.0 - maxThickness); // 1.0 / (1.0 - maxThickness)

                    for (uint step = 0u; step < SST_STEP; ++step) {
                        float stepF = float(step) - jitterStep;
                        currT = saturate(stepF * dt);
                        vec3 currScreen = pRayStart + pRayVector * currT;
                        vec2 currTexelF = currScreen.xy * uval_mainImageSize;
                        float depth = hiz_closest_sample(currTexelF / 16, 4);
                        depth = isBackwardRay ? min(depth, pRayStart.z) : max(depth, pRayStart.z);

                        float thicknessFactor = currScreen.z * maxThicknessFactor;
                        if (currScreen.z < depth && thicknessFactor > depth) {
                            break;
                        }
                    }

                    if (currT > 0.0 && currT < 1.0) {
                        float minT = saturate(currT - dt);
                        float maxT = saturate(currT + dt);

                        for (uint step = 0u; step < 8; step++) {
                            float midT = (minT + maxT) * 0.5;
                            vec3 midScreen = pRayStart + pRayVector * midT;
                            vec2 midTexelF = midScreen.xy * uval_mainImageSize;
                            float depth = hiz_closest_sample(midTexelF, 0);

                            if (midScreen.z > depth) {
                                minT = midT;
                            } else {
                                maxT = midT;
                            }
                        }

                        currT = (minT + maxT) * 0.5;

                        vec3 camScreenPos = pRayStart + pRayVector * currT;
                        vec2 camTexelPosF = camScreenPos.xy * uval_mainImageSize;
                        ivec2 camTexelPos = ivec2(camTexelPosF);

                        if (all(greaterThanEqual(camTexelPos, ivec2(0))) && all(lessThan(camTexelPos, uval_mainImageSizeI))) {
                            ivec2 readPos = camTexelPos;
                            readPos.y += uval_mainImageSizeI.y;
                            float camAreaSize = texelFetch(usam_causticsPhoton, readPos, 0).r;

                            float area = texture(usam_shadow_pixelArea, screenPosWarped).r / camAreaSize;
                            int areaQuantitized = int(area * (256.0 * 16.0));

                            #if defined(MC_GL_VENDOR_NVIDIA)
                            uint p = bitfieldInsert(camTexelPos.x, camTexelPos.y, 16, 16);
                            uvec4 pballot = subgroupPartitionNV(p);
                            int sumAreaQ = subgroupPartitionedAddNV(areaQuantitized, pballot);
                            if (subgroupBallotFindLSB(pballot) == gl_SubgroupInvocationID) {
                                imageAtomicAdd(uimg_causticsPhoton, camTexelPos, sumAreaQ);
                            }
                            #else
                            imageAtomicAdd(uimg_causticsPhoton, camTexelPos, areaQuantitized);
                            #endif
                        }
                    }
                }
            }
        }
    }
}
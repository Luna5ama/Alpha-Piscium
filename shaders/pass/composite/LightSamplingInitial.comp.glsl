layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/techniques/debug/LightSampling.glsl"
#ifdef SETTING_DEBUG_LIGHT_CUBE
#include "/techniques/debug/LightSamplingCube.glsl"
#endif
#include "/techniques/voxel/SurfaceData.glsl"
#define VOXEL_TRACE_TRUST_MATERIAL_ID
#define VOXEL_EMISSIVE_CUBES a
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"

layout(rgba32f) uniform writeonly image2D uimg_lightSampling;
#if SETTING_DEBUG_LIGHT_SAMPLING == 3
uniform isampler2D usam_lightSamplingReference;
#endif

vec3 ls_evaluate(vec3 origin, vec3 normal, vec3 albedo, inout uint rng, uint auditSlot, out uint reason
    #if SETTING_DEBUG_LIGHT_SAMPLING == 3
    , LightSamplingNode fixedEmitter
    #endif
) {
    reason = 0u;
    vec3 direction;
    float weight = 1.0;
    vec3 target = vec3(0.0);
    uint targetMaterial = 0u;
    #if SETTING_DEBUG_LIGHT_SAMPLING == 1
    float radius = sqrt(ls_random(rng));
    float phi = 6.28318530718 * ls_random(rng);
    vec3 tangent = normalize(cross(normal, abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0)));
    direction = tangent * (radius * cos(phi)) + cross(normal, tangent) * (radius * sin(phi)) + normal * sqrt(1.0 - radius * radius);
    #else
    #if SETTING_DEBUG_LIGHT_SAMPLING == 3
    float lightPdf = 1.0;
    LightSamplingNode emitter = fixedEmitter;
    #else
    if (ls_nodes[1].boundsMinMass.w == 0.0) return vec3(0.0);
    float lightPdf;
    LightSamplingNode emitter = ls_select(origin, normal, rng, auditSlot, lightPdf);
    if (lightPdf == 0.0) return vec3(0.0);
    #endif
    float conditionalPdf;
    #ifdef SETTING_DEBUG_LIGHT_CUBE
    if (!ls_sampleCube(emitter, origin, normal, vec2(ls_random(rng), ls_random(rng)), direction, conditionalPdf)) { reason = 1u; return vec3(0.0); }
    #else
    vec3 delta = emitter.boundsMinMass.xyz + 0.5 - origin;
    float distanceSquared = dot(delta, delta);
    vec3 axis = distanceSquared > 0.0 ? delta * inversesqrt(distanceSquared) : normal;
    // One uniform solid-angle draw from the cube's bounding sphere. A cone miss
    // is a zero sample, never retried: the PDF remains 1 / solidAngle.
    float coneWidth = distanceSquared <= 0.75 ? 2.0 : (0.75 / distanceSquared) / (1.0 + sqrt(1.0 - 0.75 / distanceSquared));
    float cosineTheta = 1.0 - ls_random(rng) * coneWidth;
    float sineTheta = sqrt(max(0.0, (1.0 - cosineTheta) * (1.0 + cosineTheta)));
    float phi = 6.28318530718 * ls_random(rng);
    vec3 tangent = normalize(cross(axis, abs(axis.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0)));
    direction = axis * cosineTheta + (tangent * cos(phi) + cross(axis, tangent) * sin(phi)) * sineTheta;
    conditionalPdf = 1.0 / (6.28318530718 * coneWidth);
    #endif
    reason = 2u;
    float cosine = max(dot(normal, direction), 0.0);
    if (cosine == 0.0) return vec3(0.0);
    vec3 t0 = (emitter.boundsMinMass.xyz - origin) / direction;
    vec3 t1 = (emitter.boundsMaxLink.xyz - origin) / direction;
    vec3 entry = min(t0, t1);
    vec3 exitT = max(t0, t1);
    float nearT = max(entry.x, max(entry.y, entry.z));
    float farT = min(exitT.x, min(exitT.y, exitT.z));
    if (nearT < 0.0 || nearT > farT) return vec3(0.0);
    target = origin + direction * nearT;
    weight = cosine / (3.14159265359 * conditionalPdf * lightPdf);
    targetMaterial = floatBitsToUint(emitter.boundsMaxLink.w) & ~LS_LEAF_BIT;
    #ifdef SETTING_DEBUG_LIGHT_AUDIT
    if (auditSlot != LS_INVALID) ls_audit[auditSlot * 4u + 2u] = vec4(direction, conditionalPdf);
    #endif
    #endif

    #ifdef SETTING_DEBUG_LIGHT_AUDIT
    atomicAdd(ls_traceCount, 1u);
    #endif
    VoxelRay ray = voxelray_setup(origin, direction, 0u);
    VoxelHit hit = voxel_traceRay(ray, 2048, true);
    if (voxel_traceExhausted(hit, ray)) {
        reason = 3u;
        #ifdef SETTING_DEBUG_LIGHT_AUDIT
        atomicAdd(ls_exhaustedCount, 1u);
        #endif
        return vec3(uintBitsToFloat(0x7FC00000u));
    }
    if (!hit.hit) { reason = 4u; return vec3(0.0); }
    if (!ls_isEmitter(hit.materialID)) { reason = 5u; return vec3(0.0); }
    // A primary point inside an emissive cube proxy has no surface entry.
    // Light sampling also assigns zero to this case (negative nearT).
    if (dot(hit.normal, hit.normal) == 0.0) { reason = 6u; return vec3(0.0); }
    #if SETTING_DEBUG_LIGHT_SAMPLING == 2 || SETTING_DEBUG_LIGHT_SAMPLING == 3
    if (hit.materialID != targetMaterial || distance(hit.hitPos, target) > 0.01) { reason = 7u; return vec3(0.0); }
    #endif
    #ifdef SETTING_DEBUG_LIGHT_AUDIT
    atomicAdd(ls_visibleCount, 1u);
    #endif
    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    vec3 contribution = albedo * surface.material.emissive * weight;
    reason = any(notEqual(contribution, vec3(0.0))) ? 9u : 8u;
    return contribution;
}

#if SETTING_DEBUG_LIGHT_SAMPLING == 3
void main() {
    voxel_initShared();
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;
    uint index = uint(texelPos.x + texelPos.y * uval_mainImageSizeI.x);
    uint lights = uint(textureSize(usam_lightSamplingReference, 0).x);
    uint groups = lights * 2u;
    uint samples = uint(uval_mainImageSizeI.x * uval_mainImageSizeI.y) / groups;
    vec4 result = vec4(0.0);
    if (index < groups * samples) {
        uint light = index % lights;
        uint point = (index / lights) % 2u;
        uint slot = point == 0u ? 54u : 2u;
        ivec2 pixel = (ivec2(int(slot % 8u), int(slot / 8u)) * 2 + 1) * uval_mainImageSizeI / 16;
        vec3 origin = vec3(0.0), normal = vec3(0.0), albedo = vec3(0.0);
        bool receiver = ls_loadReceiver(pixel, origin, normal, albedo);
        if (light == 0u && index < groups) {
            ls_audit[point * 4u] = vec4(origin, float(slot));
            ls_audit[point * 4u + 1u] = vec4(normal, float(samples));
            ls_audit[point * 4u + 2u] = vec4(albedo, receiver ? 1.0 : 0.0);
            ls_audit[point * 4u + 3u] = vec4(vec2(pixel), float(lights), 3.0);
        }
        if (receiver) {
            ivec4 record = texelFetch(usam_lightSamplingReference, ivec2(int(light), 0), 0);
            LightSamplingNode emitter = ls_emptyNode();
            emitter.boundsMinMass = vec4(vec3(record.xyz), 1.0);
            emitter.boundsMaxLink = vec4(vec3(record.xyz) + 1.0, uintBitsToFloat(uint(record.w) | LS_LEAF_BIT));
            uint rng = hash_pcg_11(uint(texelPos.x) + uint(texelPos.y) * 65536u) ^ hash_pcg_11(uint(frameCounter));
            uint reason;
            vec3 contribution = ls_evaluate(origin, normal, albedo, rng, LS_INVALID, reason, emitter);
            result.rgb = colors_colorspaces_convert(COLORS_WORKING_COLORSPACE, COLORS_COLORSPACES_SRGB, contribution);
            float luminance = dot(result.rgb, vec3(0.2126, 0.7152, 0.0722));
            result.a = luminance * luminance;
        } else {
            result = vec4(uintBitsToFloat(0x7FC00000u));
        }
    }
    if (ls_overflowCount != 0u) result = vec4(uintBitsToFloat(0x7FC00000u));
    imageStore(uimg_lightSampling, texelPos, result);
}
#else
void main() {
    voxel_initShared();
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;
    vec4 result = vec4(0.0);
    uint auditSlot = LS_INVALID;
    #ifdef SETTING_DEBUG_LIGHT_AUDIT
    ivec2 bin = texelPos * 8 / uval_mainImageSizeI;
    ivec2 samplePixel = (bin * 2 + 1) * uval_mainImageSizeI / 16;
    if (all(equal(texelPos, samplePixel))) {
        auditSlot = uint(bin.y * 8 + bin.x);
        ls_audit[auditSlot * 4u] = vec4(0.0);
        ls_audit[auditSlot * 4u + 1u] = vec4(0.0);
        ls_audit[auditSlot * 4u + 2u] = vec4(0.0);
        ls_audit[auditSlot * 4u + 3u] = vec4(0.0);
    }
    #endif
    vec3 origin, normal, albedo;
    if (ls_loadReceiver(texelPos, origin, normal, albedo)) {
        uint rng = hash_pcg_11(uint(texelPos.x) + uint(texelPos.y) * 65536u) ^ hash_pcg_11(uint(frameCounter));
        #ifdef SETTING_DEBUG_LIGHT_AUDIT
        atomicAdd(ls_receiverCount, 1u);
        #endif
        #if defined(SETTING_DEBUG_LIGHT_AUDIT) && SETTING_DEBUG_LIGHT_SAMPLING == 2
        if (auditSlot != LS_INVALID) {
            ls_audit[auditSlot * 4u] = vec4(origin, 0.0);
            ls_audit[auditSlot * 4u + 1u] = vec4(normal, 0.0);
            ls_audit[auditSlot * 4u + 3u].w = -1.0;
        }
        #endif
        uint reason;
        result.rgb = ls_evaluate(origin, normal, albedo, rng, auditSlot, reason);
        #ifdef SETTING_DEBUG_LIGHT_AUDIT
        atomicAdd(ls_reasons[reason], 1u);
        if (any(isnan(result.rgb)) || any(isinf(result.rgb))) atomicAdd(ls_reasons[10], 1u);
        #endif
        result.rgb = colors_colorspaces_convert(COLORS_WORKING_COLORSPACE, COLORS_COLORSPACES_SRGB, result.rgb);
        result.a = 1.0;
    }
    if (ls_overflowCount != 0u) result.rgb = vec3(uintBitsToFloat(0x7FC00000u));
    imageStore(uimg_lightSampling, texelPos, result);
}
#endif

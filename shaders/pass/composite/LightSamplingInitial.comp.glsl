layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/techniques/debug/LightSampling.glsl"
#include "/techniques/voxel/SurfaceData.glsl"
#define VOXEL_TRACE_TRUST_MATERIAL_ID
#define VOXEL_EMISSIVE_CUBES a
#include "/techniques/voxel/VoxelTrace.glsl"
#include "/util/FullScreenComp.glsl"
#include "/util/Coords.glsl"

layout(rgba32f) uniform writeonly image2D uimg_lightSampling;

vec3 ls_evaluate(vec3 origin, vec3 normal, vec3 albedo, inout uint rng, uint auditSlot) {
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
    if (ls_nodes[1].boundsMinMass.w == 0.0) return vec3(0.0);
    float lightPdf;
    LightSamplingNode emitter = ls_select(origin, normal, rng, auditSlot, lightPdf);
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
    weight = cosine * (2.0 * coneWidth) / lightPdf;
    targetMaterial = floatBitsToUint(emitter.boundsMaxLink.w) & ~LS_LEAF_BIT;
    #endif

    atomicAdd(ls_traceCount, 1u);
    VoxelRay ray = voxelray_setup(origin, direction, 0u);
    VoxelHit hit = voxel_traceRay(ray, 2048, true);
    if (voxel_traceExhausted(hit, ray)) {
        atomicAdd(ls_exhaustedCount, 1u);
        return vec3(uintBitsToFloat(0x7FC00000u));
    }
    if (!hit.hit || !ls_isEmitter(hit.materialID)) return vec3(0.0);
    // A primary point inside an emissive cube proxy has no surface entry.
    // Light sampling also assigns zero to this case (negative nearT).
    if (dot(hit.normal, hit.normal) == 0.0) return vec3(0.0);
    #if SETTING_DEBUG_LIGHT_SAMPLING == 2
    if (hit.materialID != targetMaterial || distance(hit.hitPos, target) > 0.01) return vec3(0.0);
    #endif
    atomicAdd(ls_visibleCount, 1u);
    voxel_SurfaceData surface = voxel_sampleVoxelSurface(hit, 0.0);
    return albedo * surface.material.emissive * weight;
}

void main() {
    voxel_initShared();
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;
    vec4 result = vec4(0.0);
    vec3 origin, normal, albedo;
    if (ls_loadReceiver(texelPos, origin, normal, albedo)) {
        uint rng = hash_pcg_11(uint(texelPos.x) + uint(texelPos.y) * 65536u) ^ hash_pcg_11(uint(frameCounter));
        uint auditSlot = LSV_INVALID;
        if (texelPos.x >= 80 && texelPos.y >= 45 && (texelPos.x - 80) % 160 == 0 && (texelPos.y - 45) % 90 == 0) {
            auditSlot = uint((texelPos.y - 45) / 90 * 8 + (texelPos.x - 80) / 160);
        }
        atomicAdd(ls_receiverCount, 1u);
        result.rgb = ls_evaluate(origin, normal, albedo, rng, auditSlot);
        result.rgb = colors_colorspaces_convert(COLORS_WORKING_COLORSPACE, COLORS_COLORSPACES_SRGB, result.rgb);
        result.a = 1.0;
    }
    if (ls_overflowCount != 0u || lsv_exhausted != 0u) result.rgb = vec3(uintBitsToFloat(0x7FC00000u));
    imageStore(uimg_lightSampling, texelPos, result);
}

#extension GL_KHR_shader_subgroup_basic : enable

#include "/util/Morton.glsl"
#include "/util/Coords.glsl"
#include "ss/Common.glsl"
#include "CumulusRay.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(RENDER_MULTIPLIER, RENDER_MULTIPLIER);

layout(rgba32ui) uniform writeonly uimage2D uimg_rgba32ui;

void main() {
    uint lane = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    ivec2 pixel = ivec2((gl_WorkGroupID.xy << 3u) + morton_8bDecode(lane));
    if (any(greaterThanEqual(pixel, renderSize))) return;

    vec2 center = clouds_ss_upscaledTexelCenter(pixel);
    ivec2 fullPixel = min(ivec2(center), uval_mainImageSizeI - 1);
    float viewZ = texelFetch(usam_gbufferSolidViewZ, fullPixel, 0).r;
    vec3 view = coords_toViewCoord(center * uval_mainImageSizeRcp, viewZ, global_camProjInverse);
    vec3 direction = normalize(mat3(gbufferModelViewInverse) * view);
    float limit = viewZ <= -65536.0 ? CU_RANGE : length(view) * 0.001;
    float nearDistance = clouds_ss_segmentStart(clouds_ss_translucentZ(fullPixel), view);
    vec3 origin = clouds_cu_origin(gbufferModelViewInverse[3].xyz);
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 atmOrigin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    vec4 jitter = vec4(rand_stbnVec2(rand_newStbnPos(pixel, 1), frameCounter),
        rand_stbnVec2(rand_newStbnPos(pixel, 2), frameCounter));
    CloudSSHistoryData history = clouds_ss_historyData_init();
    ScatteringResult cloud = clouds_cu_render(atmosphere, origin, atmOrigin, direction,
        vec2(nearDistance, limit), jitter, false, history.distanceKm);
    history.inScattering = cloud.inScattering;
    history.transmittance = cloud.transmittance;
    history.hLen = 1.0;
    uvec4 encoded;
    clouds_ss_historyData_pack(encoded, history);
    transient_lowCloudRender_store(pixel, encoded);
}

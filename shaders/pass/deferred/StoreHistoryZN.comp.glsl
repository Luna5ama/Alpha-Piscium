#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec3 worldNormal = vec3(0.0, 1.0, 0.0);
        vec3 geomWorldNomral = vec3(0.0, 1.0, 0.0);
        float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            worldNormal = coords_dir_viewToWorld(gData.normal);
            geomWorldNomral = coords_dir_viewToWorld(gData.geomNormal);
        }

        history_viewZ_store(texelPos, vec4(viewZ));
        history_geomWorldNormal_store(texelPos, vec4(geomWorldNomral * 0.5 + 0.5, 0.0));
        history_worldNormal_store(texelPos, vec4(worldNormal * 0.5 + 0.5, 0.0));
    }
}

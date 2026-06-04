#ifndef INCLUDE_techniques_voxel_SurfaceData_glsl
#define INCLUDE_techniques_voxel_SurfaceData_glsl a

#include "Voxelization.glsl"
#include "VoxelFaceTexcoords.glsl"
#include "VoxelTrace.glsl"
#include "/util/Material.glsl"

struct voxel_SurfaceData {
    Material material;
    bool valid;
};

voxel_SurfaceData voxel_sampleVoxelSurface(VoxelHit hit, float lod) {
    voxel_SurfaceData surface;
    surface.valid = false;
    surface.material = material_init();

    if (!hit.hit || hit.materialID == 0u || hit.materialID == MATERIAL_ID_WATER) {
        return surface;
    }

    uint faceId = voxel_faceIndexFromNormal(hit.normal);
    uvec2 tcData = voxel_faceTexcoords[voxel_faceTexcoordIndex(hit.materialID, faceId)];
    vec4 tc = unpackUnorm4x16(tcData);
    if (all(equal(tc, vec4(0.0)))) {
        return surface;
    }

    vec2 localUV = voxel_faceLocalUV(faceId, hit.hitPos);
    vec2 atlasUV = mix(tc.xw, tc.zy, localUV);
    vec4 albedoData = textureLod(usam_blockAtlasColor, atlasUV, lod);
    vec4 speuclarData = textureLod(usam_blockAtlasSpecular, atlasUV, lod);

    float emissiveS = speuclarData.a;
    emissiveS *= float(speuclarData.a < 1.0);
    speuclarData.a = emissiveS;

    GBufferData gData = gbufferData_init();
    gData.albedo = albedoData.rgb;
    gData.materialID = hit.materialID;
    gData.pbrSpecular = speuclarData;
    gData.geomNormal = voxel_faceNormal(faceId);
    gData.normal = gData.geomNormal;
    gData.geomTangent = voxel_faceTangent(faceId);

    Material material = material_decode(gData);

    surface.material = material;
    surface.valid = true;
    return surface;
}

#endif
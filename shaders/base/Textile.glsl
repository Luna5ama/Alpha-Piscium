#define saturate(x) clamp(x, 0.0, 1.0)
ivec2 _textile_texelToTexel(ivec2 texelPos, ivec2 tileOffset, ivec2 tileSize) {
    return clamp(texelPos, ivec2(0), tileSize - 1) + tileOffset;
}

vec2 _textile_uvToUV(vec2 uv, vec2 tileOffsetF, vec2 tileSizeF, vec2 textureSizeRcp) {
    vec2 textureTexelPos = clamp(uv * tileSizeF, vec2(0.5), tileSizeF - 0.5) + tileOffsetF;
    return saturate(textureTexelPos * textureSizeRcp);
}

vec2 _textile_uvToGatherUV(vec2 uv, vec2 tileOffsetF, vec2 tileSizeF, vec2 textureSizeRcp) {
    vec2 textureTexelPos = clamp(uv * tileSizeF, vec2(1.0), tileSizeF - 1.0) + tileOffsetF;
    return saturate(textureTexelPos * textureSizeRcp);
}
#undef saturate

#define _RGBA32UI_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _RGBA32UI_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _RGBA32UI_0_SIZE uval_mainImageSizeI
#define _RGBA32UI_0_SIZE_F uval_mainImageSize
#define _RGBA32UI_0_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA32UI_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA32UI_0_OFFSET, _RGBA32UI_0_SIZE)
#define _RGBA32UI_0_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA32UI_0_OFFSET_F, _RGBA32UI_0_SIZE_F, _RGBA32UI_0_SIZE_RCP)
#define _RGBA32UI_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA32UI_0_OFFSET_F, _RGBA32UI_0_SIZE_F, _RGBA32UI_0_SIZE_RCP)
#define _RGBA32UI_1_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 1)
#define _RGBA32UI_1_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 1)
#define _RGBA32UI_1_SIZE uval_mainImageSizeI
#define _RGBA32UI_1_SIZE_F uval_mainImageSize
#define _RGBA32UI_1_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA32UI_1_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA32UI_1_OFFSET, _RGBA32UI_1_SIZE)
#define _RGBA32UI_1_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA32UI_1_OFFSET_F, _RGBA32UI_1_SIZE_F, _RGBA32UI_1_SIZE_RCP)
#define _RGBA32UI_1_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA32UI_1_OFFSET_F, _RGBA32UI_1_SIZE_F, _RGBA32UI_1_SIZE_RCP)
#define _RGBA32UI_2_OFFSET ivec2(uval_mainImageSizeI.x * 1, uval_mainImageSizeI.y * 0)
#define _RGBA32UI_2_OFFSET_F vec2(uval_mainImageSizeI.x * 1, uval_mainImageSize.y * 0)
#define _RGBA32UI_2_SIZE uval_mainImageSizeI
#define _RGBA32UI_2_SIZE_F uval_mainImageSize
#define _RGBA32UI_2_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA32UI_2_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA32UI_2_OFFSET, _RGBA32UI_2_SIZE)
#define _RGBA32UI_2_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA32UI_2_OFFSET_F, _RGBA32UI_2_SIZE_F, _RGBA32UI_2_SIZE_RCP)
#define _RGBA32UI_2_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA32UI_2_OFFSET_F, _RGBA32UI_2_SIZE_F, _RGBA32UI_2_SIZE_RCP)
#define _RGBA32UI_3_OFFSET ivec2(uval_mainImageSizeI.x * 1, uval_mainImageSizeI.y * 1)
#define _RGBA32UI_3_OFFSET_F vec2(uval_mainImageSizeI.x * 1, uval_mainImageSize.y * 1)
#define _RGBA32UI_3_SIZE uval_mainImageSizeI
#define _RGBA32UI_3_SIZE_F uval_mainImageSize
#define _RGBA32UI_3_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA32UI_3_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA32UI_3_OFFSET, _RGBA32UI_3_SIZE)
#define _RGBA32UI_3_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA32UI_3_OFFSET_F, _RGBA32UI_3_SIZE_F, _RGBA32UI_3_SIZE_RCP)
#define _RGBA32UI_3_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA32UI_3_OFFSET_F, _RGBA32UI_3_SIZE_F, _RGBA32UI_3_SIZE_RCP)
#define history_gi_sample(x) texture(usam_rgba32ui, _RGBA32UI_0_UV_TO_UV(x))
#define history_gi_gather(x, c) textureGather(usam_rgba32ui, _RGBA32UI_0_UV_TO_GATHER_UV(x), c)
#define history_gi_fetch(x) texelFetch(usam_rgba32ui, _RGBA32UI_0_TEXEL_TO_TEXEL(x), 0)
#define history_gi_load(x) imageLoad(uimg_rgba32ui, _RGBA32UI_0_TEXEL_TO_TEXEL(x))
#define history_gi_store(x, v) imageStore(uimg_rgba32ui, _RGBA32UI_0_TEXEL_TO_TEXEL(x), v)
#define history_lowCloud_sample(x) texture(usam_rgba32ui, _RGBA32UI_1_UV_TO_UV(x))
#define history_lowCloud_gather(x, c) textureGather(usam_rgba32ui, _RGBA32UI_1_UV_TO_GATHER_UV(x), c)
#define history_lowCloud_fetch(x) texelFetch(usam_rgba32ui, _RGBA32UI_1_TEXEL_TO_TEXEL(x), 0)
#define history_lowCloud_load(x) imageLoad(uimg_rgba32ui, _RGBA32UI_1_TEXEL_TO_TEXEL(x))
#define history_lowCloud_store(x, v) imageStore(uimg_rgba32ui, _RGBA32UI_1_TEXEL_TO_TEXEL(x), v)
#define transient_lowCloudRender_sample(x) texture(usam_rgba32ui, _RGBA32UI_2_UV_TO_UV(x))
#define transient_lowCloudRender_gather(x, c) textureGather(usam_rgba32ui, _RGBA32UI_2_UV_TO_GATHER_UV(x), c)
#define transient_lowCloudRender_fetch(x) texelFetch(usam_rgba32ui, _RGBA32UI_2_TEXEL_TO_TEXEL(x), 0)
#define transient_lowCloudRender_load(x) imageLoad(uimg_rgba32ui, _RGBA32UI_2_TEXEL_TO_TEXEL(x))
#define transient_lowCloudRender_store(x, v) imageStore(uimg_rgba32ui, _RGBA32UI_2_TEXEL_TO_TEXEL(x), v)
#define transient_giReprojected_sample(x) texture(usam_rgba32ui, _RGBA32UI_3_UV_TO_UV(x))
#define transient_giReprojected_gather(x, c) textureGather(usam_rgba32ui, _RGBA32UI_3_UV_TO_GATHER_UV(x), c)
#define transient_giReprojected_fetch(x) texelFetch(usam_rgba32ui, _RGBA32UI_3_TEXEL_TO_TEXEL(x), 0)
#define transient_giReprojected_load(x) imageLoad(uimg_rgba32ui, _RGBA32UI_3_TEXEL_TO_TEXEL(x))
#define transient_giReprojected_store(x, v) imageStore(uimg_rgba32ui, _RGBA32UI_3_TEXEL_TO_TEXEL(x), v)
#define transient_lowCloudAccumulated_sample(x) texture(usam_rgba32ui, _RGBA32UI_0_UV_TO_UV(x))
#define transient_lowCloudAccumulated_gather(x, c) textureGather(usam_rgba32ui, _RGBA32UI_0_UV_TO_GATHER_UV(x), c)
#define transient_lowCloudAccumulated_fetch(x) texelFetch(usam_rgba32ui, _RGBA32UI_0_TEXEL_TO_TEXEL(x), 0)
#define transient_lowCloudAccumulated_load(x) imageLoad(uimg_rgba32ui, _RGBA32UI_0_TEXEL_TO_TEXEL(x))
#define transient_lowCloudAccumulated_store(x, v) imageStore(uimg_rgba32ui, _RGBA32UI_0_TEXEL_TO_TEXEL(x), v)
#define _RGBA16F_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _RGBA16F_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _RGBA16F_0_SIZE uval_mainImageSizeI
#define _RGBA16F_0_SIZE_F uval_mainImageSize
#define _RGBA16F_0_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA16F_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA16F_0_OFFSET, _RGBA16F_0_SIZE)
#define _RGBA16F_0_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA16F_0_OFFSET_F, _RGBA16F_0_SIZE_F, _RGBA16F_0_SIZE_RCP)
#define _RGBA16F_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA16F_0_OFFSET_F, _RGBA16F_0_SIZE_F, _RGBA16F_0_SIZE_RCP)
#define _RGBA16F_1_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 1)
#define _RGBA16F_1_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 1)
#define _RGBA16F_1_SIZE uval_mainImageSizeI
#define _RGBA16F_1_SIZE_F uval_mainImageSize
#define _RGBA16F_1_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA16F_1_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA16F_1_OFFSET, _RGBA16F_1_SIZE)
#define _RGBA16F_1_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA16F_1_OFFSET_F, _RGBA16F_1_SIZE_F, _RGBA16F_1_SIZE_RCP)
#define _RGBA16F_1_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA16F_1_OFFSET_F, _RGBA16F_1_SIZE_F, _RGBA16F_1_SIZE_RCP)
#define transient_shadow_sample(x) texture(usam_rgba16f, _RGBA16F_0_UV_TO_UV(x))
#define transient_shadow_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_0_UV_TO_GATHER_UV(x), c)
#define transient_shadow_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), 0)
#define transient_shadow_load(x) imageLoad(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x))
#define transient_shadow_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), v)
#define transient_directDiffusePassThrough_sample(x) texture(usam_rgba16f, _RGBA16F_1_UV_TO_UV(x))
#define transient_directDiffusePassThrough_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_1_UV_TO_GATHER_UV(x), c)
#define transient_directDiffusePassThrough_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x), 0)
#define transient_directDiffusePassThrough_load(x) imageLoad(uimg_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x))
#define transient_directDiffusePassThrough_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x), v)
#define transient_atrous1_sample(x) texture(usam_rgba16f, _RGBA16F_0_UV_TO_UV(x))
#define transient_atrous1_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_0_UV_TO_GATHER_UV(x), c)
#define transient_atrous1_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), 0)
#define transient_atrous1_load(x) imageLoad(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x))
#define transient_atrous1_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), v)
#define transient_atrous2_sample(x) texture(usam_rgba16f, _RGBA16F_1_UV_TO_UV(x))
#define transient_atrous2_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_1_UV_TO_GATHER_UV(x), c)
#define transient_atrous2_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x), 0)
#define transient_atrous2_load(x) imageLoad(uimg_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x))
#define transient_atrous2_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x), v)
#define transient_translucentReflection_sample(x) texture(usam_rgba16f, _RGBA16F_0_UV_TO_UV(x))
#define transient_translucentReflection_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_0_UV_TO_GATHER_UV(x), c)
#define transient_translucentReflection_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), 0)
#define transient_translucentReflection_load(x) imageLoad(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x))
#define transient_translucentReflection_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), v)
#define transient_translucentRefraction_sample(x) texture(usam_rgba16f, _RGBA16F_1_UV_TO_UV(x))
#define transient_translucentRefraction_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_1_UV_TO_GATHER_UV(x), c)
#define transient_translucentRefraction_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x), 0)
#define transient_translucentRefraction_load(x) imageLoad(uimg_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x))
#define transient_translucentRefraction_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_1_TEXEL_TO_TEXEL(x), v)
#define transient_dofInput_sample(x) texture(usam_rgba16f, _RGBA16F_0_UV_TO_UV(x))
#define transient_dofInput_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_0_UV_TO_GATHER_UV(x), c)
#define transient_dofInput_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), 0)
#define transient_dofInput_load(x) imageLoad(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x))
#define transient_dofInput_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), v)
#define transient_bloom_sample(x) texture(usam_rgba16f, _RGBA16F_0_UV_TO_UV(x))
#define transient_bloom_gather(x, c) textureGather(usam_rgba16f, _RGBA16F_0_UV_TO_GATHER_UV(x), c)
#define transient_bloom_fetch(x) texelFetch(usam_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), 0)
#define transient_bloom_load(x) imageLoad(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x))
#define transient_bloom_store(x, v) imageStore(uimg_rgba16f, _RGBA16F_0_TEXEL_TO_TEXEL(x), v)
#define _RGB10_A2_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _RGB10_A2_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _RGB10_A2_0_SIZE uval_mainImageSizeI
#define _RGB10_A2_0_SIZE_F uval_mainImageSize
#define _RGB10_A2_0_SIZE_RCP uval_mainImageSizeRcp
#define _RGB10_A2_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGB10_A2_0_OFFSET, _RGB10_A2_0_SIZE)
#define _RGB10_A2_0_UV_TO_UV(uv) _textile_uvToUV(uv, _RGB10_A2_0_OFFSET_F, _RGB10_A2_0_SIZE_F, _RGB10_A2_0_SIZE_RCP)
#define _RGB10_A2_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGB10_A2_0_OFFSET_F, _RGB10_A2_0_SIZE_F, _RGB10_A2_0_SIZE_RCP)
#define _RGB10_A2_1_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 1)
#define _RGB10_A2_1_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 1)
#define _RGB10_A2_1_SIZE uval_mainImageSizeI
#define _RGB10_A2_1_SIZE_F uval_mainImageSize
#define _RGB10_A2_1_SIZE_RCP uval_mainImageSizeRcp
#define _RGB10_A2_1_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGB10_A2_1_OFFSET, _RGB10_A2_1_SIZE)
#define _RGB10_A2_1_UV_TO_UV(uv) _textile_uvToUV(uv, _RGB10_A2_1_OFFSET_F, _RGB10_A2_1_SIZE_F, _RGB10_A2_1_SIZE_RCP)
#define _RGB10_A2_1_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGB10_A2_1_OFFSET_F, _RGB10_A2_1_SIZE_F, _RGB10_A2_1_SIZE_RCP)
#define history_geomWorldNormal_sample(x) texture(usam_rgb10_a2, _RGB10_A2_0_UV_TO_UV(x))
#define history_geomWorldNormal_gather(x, c) textureGather(usam_rgb10_a2, _RGB10_A2_0_UV_TO_GATHER_UV(x), c)
#define history_geomWorldNormal_fetch(x) texelFetch(usam_rgb10_a2, _RGB10_A2_0_TEXEL_TO_TEXEL(x), 0)
#define history_geomWorldNormal_load(x) imageLoad(uimg_rgb10_a2, _RGB10_A2_0_TEXEL_TO_TEXEL(x))
#define history_geomWorldNormal_store(x, v) imageStore(uimg_rgb10_a2, _RGB10_A2_0_TEXEL_TO_TEXEL(x), v)
#define history_worldNormal_sample(x) texture(usam_rgb10_a2, _RGB10_A2_1_UV_TO_UV(x))
#define history_worldNormal_gather(x, c) textureGather(usam_rgb10_a2, _RGB10_A2_1_UV_TO_GATHER_UV(x), c)
#define history_worldNormal_fetch(x) texelFetch(usam_rgb10_a2, _RGB10_A2_1_TEXEL_TO_TEXEL(x), 0)
#define history_worldNormal_load(x) imageLoad(uimg_rgb10_a2, _RGB10_A2_1_TEXEL_TO_TEXEL(x))
#define history_worldNormal_store(x, v) imageStore(uimg_rgb10_a2, _RGB10_A2_1_TEXEL_TO_TEXEL(x), v)
#define _RGBA8_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _RGBA8_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _RGBA8_0_SIZE uval_mainImageSizeI
#define _RGBA8_0_SIZE_F uval_mainImageSize
#define _RGBA8_0_SIZE_RCP uval_mainImageSizeRcp
#define _RGBA8_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RGBA8_0_OFFSET, _RGBA8_0_SIZE)
#define _RGBA8_0_UV_TO_UV(uv) _textile_uvToUV(uv, _RGBA8_0_OFFSET_F, _RGBA8_0_SIZE_F, _RGBA8_0_SIZE_RCP)
#define _RGBA8_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RGBA8_0_OFFSET_F, _RGBA8_0_SIZE_F, _RGBA8_0_SIZE_RCP)
#define transient_edgeMask_sample(x) texture(usam_rgba8, _RGBA8_0_UV_TO_UV(x))
#define transient_edgeMask_gather(x, c) textureGather(usam_rgba8, _RGBA8_0_UV_TO_GATHER_UV(x), c)
#define transient_edgeMask_fetch(x) texelFetch(usam_rgba8, _RGBA8_0_TEXEL_TO_TEXEL(x), 0)
#define transient_edgeMask_load(x) imageLoad(uimg_rgba8, _RGBA8_0_TEXEL_TO_TEXEL(x))
#define transient_edgeMask_store(x, v) imageStore(uimg_rgba8, _RGBA8_0_TEXEL_TO_TEXEL(x), v)
#define transient_solidAlbedo_sample(x) texture(usam_rgba8, _RGBA8_0_UV_TO_UV(x))
#define transient_solidAlbedo_gather(x, c) textureGather(usam_rgba8, _RGBA8_0_UV_TO_GATHER_UV(x), c)
#define transient_solidAlbedo_fetch(x) texelFetch(usam_rgba8, _RGBA8_0_TEXEL_TO_TEXEL(x), 0)
#define transient_solidAlbedo_load(x) imageLoad(uimg_rgba8, _RGBA8_0_TEXEL_TO_TEXEL(x))
#define transient_solidAlbedo_store(x, v) imageStore(uimg_rgba8, _RGBA8_0_TEXEL_TO_TEXEL(x), v)
#define _RG32UI_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _RG32UI_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _RG32UI_0_SIZE uval_mainImageSizeI
#define _RG32UI_0_SIZE_F uval_mainImageSize
#define _RG32UI_0_SIZE_RCP uval_mainImageSizeRcp
#define _RG32UI_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RG32UI_0_OFFSET, _RG32UI_0_SIZE)
#define _RG32UI_0_UV_TO_UV(uv) _textile_uvToUV(uv, _RG32UI_0_OFFSET_F, _RG32UI_0_SIZE_F, _RG32UI_0_SIZE_RCP)
#define _RG32UI_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RG32UI_0_OFFSET_F, _RG32UI_0_SIZE_F, _RG32UI_0_SIZE_RCP)
#define _RG32UI_1_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 1)
#define _RG32UI_1_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 1)
#define _RG32UI_1_SIZE uval_mainImageSizeI
#define _RG32UI_1_SIZE_F uval_mainImageSize
#define _RG32UI_1_SIZE_RCP uval_mainImageSizeRcp
#define _RG32UI_1_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _RG32UI_1_OFFSET, _RG32UI_1_SIZE)
#define _RG32UI_1_UV_TO_UV(uv) _textile_uvToUV(uv, _RG32UI_1_OFFSET_F, _RG32UI_1_SIZE_F, _RG32UI_1_SIZE_RCP)
#define _RG32UI_1_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _RG32UI_1_OFFSET_F, _RG32UI_1_SIZE_F, _RG32UI_1_SIZE_RCP)
#define transient_packedZN_sample(x) texture(usam_rg32ui, _RG32UI_0_UV_TO_UV(x))
#define transient_packedZN_gather(x, c) textureGather(usam_rg32ui, _RG32UI_0_UV_TO_GATHER_UV(x), c)
#define transient_packedZN_fetch(x) texelFetch(usam_rg32ui, _RG32UI_0_TEXEL_TO_TEXEL(x), 0)
#define transient_packedZN_load(x) imageLoad(uimg_rg32ui, _RG32UI_0_TEXEL_TO_TEXEL(x))
#define transient_packedZN_store(x, v) imageStore(uimg_rg32ui, _RG32UI_0_TEXEL_TO_TEXEL(x), v)
#define transient_ssgiOut_sample(x) texture(usam_rg32ui, _RG32UI_1_UV_TO_UV(x))
#define transient_ssgiOut_gather(x, c) textureGather(usam_rg32ui, _RG32UI_1_UV_TO_GATHER_UV(x), c)
#define transient_ssgiOut_fetch(x) texelFetch(usam_rg32ui, _RG32UI_1_TEXEL_TO_TEXEL(x), 0)
#define transient_ssgiOut_load(x) imageLoad(uimg_rg32ui, _RG32UI_1_TEXEL_TO_TEXEL(x))
#define transient_ssgiOut_store(x, v) imageStore(uimg_rg32ui, _RG32UI_1_TEXEL_TO_TEXEL(x), v)
#define _R32UI_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _R32UI_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _R32UI_0_SIZE uval_mainImageSizeI
#define _R32UI_0_SIZE_F uval_mainImageSize
#define _R32UI_0_SIZE_RCP uval_mainImageSizeRcp
#define _R32UI_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _R32UI_0_OFFSET, _R32UI_0_SIZE)
#define _R32UI_0_UV_TO_UV(uv) _textile_uvToUV(uv, _R32UI_0_OFFSET_F, _R32UI_0_SIZE_F, _R32UI_0_SIZE_RCP)
#define _R32UI_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _R32UI_0_OFFSET_F, _R32UI_0_SIZE_F, _R32UI_0_SIZE_RCP)
#define transient_geometryNormal_sample(x) texture(usam_r32ui, _R32UI_0_UV_TO_UV(x))
#define transient_geometryNormal_gather(x, c) textureGather(usam_r32ui, _R32UI_0_UV_TO_GATHER_UV(x), c)
#define transient_geometryNormal_fetch(x) texelFetch(usam_r32ui, _R32UI_0_TEXEL_TO_TEXEL(x), 0)
#define transient_geometryNormal_load(x) imageLoad(uimg_r32ui, _R32UI_0_TEXEL_TO_TEXEL(x))
#define transient_geometryNormal_store(x, v) imageStore(uimg_r32ui, _R32UI_0_TEXEL_TO_TEXEL(x), v)
#define _R32F_0_OFFSET ivec2(uval_mainImageSizeI.x * 0, uval_mainImageSizeI.y * 0)
#define _R32F_0_OFFSET_F vec2(uval_mainImageSizeI.x * 0, uval_mainImageSize.y * 0)
#define _R32F_0_SIZE uval_mainImageSizeI
#define _R32F_0_SIZE_F uval_mainImageSize
#define _R32F_0_SIZE_RCP uval_mainImageSizeRcp
#define _R32F_0_TEXEL_TO_TEXEL(texelPos) _textile_texelToTexel(texelPos, _R32F_0_OFFSET, _R32F_0_SIZE)
#define _R32F_0_UV_TO_UV(uv) _textile_uvToUV(uv, _R32F_0_OFFSET_F, _R32F_0_SIZE_F, _R32F_0_SIZE_RCP)
#define _R32F_0_UV_TO_GATHER_UV(uv) _textile_uvToGatherUV(uv, _R32F_0_OFFSET_F, _R32F_0_SIZE_F, _R32F_0_SIZE_RCP)
#define history_viewZ_sample(x) texture(usam_r32f, _R32F_0_UV_TO_UV(x))
#define history_viewZ_gather(x, c) textureGather(usam_r32f, _R32F_0_UV_TO_GATHER_UV(x), c)
#define history_viewZ_fetch(x) texelFetch(usam_r32f, _R32F_0_TEXEL_TO_TEXEL(x), 0)
#define history_viewZ_load(x) imageLoad(uimg_r32f, _R32F_0_TEXEL_TO_TEXEL(x))
#define history_viewZ_store(x, v) imageStore(uimg_r32f, _R32F_0_TEXEL_TO_TEXEL(x), v)

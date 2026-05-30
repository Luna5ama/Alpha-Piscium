#include "/util/Colors2.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;

// 18x18 shared memory: 16x16 workgroup + 1 px border each side
shared vec4 s_data[18][18];

float fxaa_luma(vec3 rgb) {
    return mmax3(rgb);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uvec2 groupOrigin = gl_WorkGroupID.xy << 4u;

    // Cooperatively load 18*18 = 324 texels using 256 threads (2 rounds)
    uint idx = gl_LocalInvocationIndex;
    for (uint i = idx; i < 324u; i += 256u) {
        uvec2 sxy = uvec2(i % 18u, i / 18u);
        ivec2 src = ivec2(groupOrigin) + ivec2(sxy) - 1;
        src = clamp(src, ivec2(0), uval_mainImageSizeI - 1);
        s_data[sxy.y][sxy.x] = transient_taaOutput_fetch(src);
    }

    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ivec2 lp = ivec2(gl_LocalInvocationID.xy) + 1; // local pos offset by border

        vec4 dataM = s_data[lp.y][lp.x];

        // Write pre-FXAA TAA output to history (before any FXAA smoothing)
        history_taa_store(texelPos, dataM);

        vec3 colM  = dataM.rgb;
        vec3 colN  = s_data[lp.y - 1][lp.x    ].rgb;
        vec3 colS  = s_data[lp.y + 1][lp.x    ].rgb;
        vec3 colE  = s_data[lp.y    ][lp.x + 1].rgb;
        vec3 colW  = s_data[lp.y    ][lp.x - 1].rgb;

        float lumaM = fxaa_luma(colM);
        float lumaN = fxaa_luma(colN);
        float lumaS = fxaa_luma(colS);
        float lumaE = fxaa_luma(colE);
        float lumaW = fxaa_luma(colW);

        float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
        float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
        float lumaRange = lumaMax - lumaMin;

        const float FXAA_EDGE_THRESHOLD     = 0.125;
        const float FXAA_EDGE_THRESHOLD_MIN = 0.0312;
        const float FXAA_SUBPIX_QUALITY     = 1.0;

        vec4 outputData = dataM;

        if (lumaRange >= max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
            vec3 colNW = s_data[lp.y - 1][lp.x - 1].rgb;
            vec3 colNE = s_data[lp.y - 1][lp.x + 1].rgb;
            vec3 colSW = s_data[lp.y + 1][lp.x - 1].rgb;
            vec3 colSE = s_data[lp.y + 1][lp.x + 1].rgb;

            float lumaNW = fxaa_luma(colNW);
            float lumaNE = fxaa_luma(colNE);
            float lumaSW = fxaa_luma(colSW);
            float lumaSE = fxaa_luma(colSE);

            // Subpixel blend factor (weighted neighbourhood average vs centre)
            float lumaLow = (lumaN + lumaS + lumaE + lumaW) * 2.0
                          + lumaNW + lumaNE + lumaSW + lumaSE;
            float lumaAvg = lumaLow * (1.0 / 12.0);
            float subpixBlend = abs(lumaAvg - lumaM) / lumaRange;
            subpixBlend = smoothstep(0.0, 1.0, subpixBlend);
            subpixBlend = subpixBlend * subpixBlend * FXAA_SUBPIX_QUALITY;

            // Edge orientation: horizontal vs vertical
            float edgeH = abs(lumaNW - 2.0 * lumaW + lumaSW)
                        + abs(lumaN  - 2.0 * lumaM + lumaS ) * 2.0
                        + abs(lumaNE - 2.0 * lumaE + lumaSE);
            float edgeV = abs(lumaNW - 2.0 * lumaN + lumaNE)
                        + abs(lumaW  - 2.0 * lumaM + lumaE ) * 2.0
                        + abs(lumaSW - 2.0 * lumaS + lumaSE);
            bool isHorizEdge = edgeH >= edgeV;

            // Neighbours perpendicular to the detected edge
            float luma1 = isHorizEdge ? lumaN : lumaW;
            float luma2 = isHorizEdge ? lumaS : lumaE;
            vec3  col1  = isHorizEdge ? colN  : colW;
            vec3  col2  = isHorizEdge ? colS  : colE;

            float grad1 = abs(luma1 - lumaM);
            float grad2 = abs(luma2 - lumaM);

            // Blend toward the steeper-gradient neighbour
            vec3 edgeColor = (grad1 >= grad2) ? col1 : col2;

            // Edge blend capped at 0.5 (single pixel edge)
            float edgeBlend = 0.5 * lumaRange / max(lumaMax, 1e-5);
            edgeBlend = clamp(edgeBlend, 0.0, 0.5);

            float finalBlend = max(subpixBlend, edgeBlend);

            // Modulate subpixBlend by a 3x3 filtered lumaDiff heuristic from TAA.
            // Sample at 4 bilinear corners (0.5 px offset) to get a 3x3 box filter:
            {
                vec2 texelUV = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
                vec2 hp = 0.5 * uval_mainImageSizeRcp;
                float filteredLumaDiff = 0.0;
                filteredLumaDiff += transient_lumaDiff_sample(texelUV + vec2(-hp.x, -hp.y)).r;
                filteredLumaDiff += transient_lumaDiff_sample(texelUV + vec2( hp.x, -hp.y)).r;
                filteredLumaDiff += transient_lumaDiff_sample(texelUV + vec2(-hp.x,  hp.y)).r;
                filteredLumaDiff += transient_lumaDiff_sample(texelUV + vec2( hp.x,  hp.y)).r;
                filteredLumaDiff *= 0.25;
                finalBlend *= filteredLumaDiff;
            }

            outputData.rgb = mix(colM, edgeColor, finalBlend);
        }

        transient_fxaaOutput_store(texelPos, outputData);
    }
}


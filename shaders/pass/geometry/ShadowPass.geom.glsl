#include "/Base.glsl"

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
in vec2 vert_texcoord[];
#if defined(SHADOW_PASS_TRANSLUCENT)
in vec3 vert_color[];
#endif
#endif

in vec2 vert_screenPos[];

in uint vert_worldNormalMaterialID[];
in uint vert_survived[];

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
out vec2 frag_texcoord;
#if defined(SHADOW_PASS_TRANSLUCENT)
out vec3 frag_color;
#endif
#endif

out vec2 frag_screenPos;
out uint frag_worldNormalMaterialID;

#if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
out uvec2 frag_texcoordMinMax;
#endif


void main() {
    uint survived = vert_survived[0] & vert_survived[1] & vert_survived[2];
    vec4 screenBoundF = vec4(
        min(gl_in[0].gl_Position.xy, min(gl_in[1].gl_Position.xy, gl_in[2].gl_Position.xy)),
        max(gl_in[0].gl_Position.xy, max(gl_in[1].gl_Position.xy, gl_in[2].gl_Position.xy))
    );
    screenBoundF = screenBoundF * 0.5 + 0.5;
    screenBoundF *= float(SETTING_SHADOW_MAP_RESOLUTION);
    uvec4 screenBound = uvec4(floor(screenBoundF.xy), ceil(screenBoundF.zw));
    survived = survived & uint(!any(equal(round(screenBoundF.xy), round(screenBoundF.zw))));
    survived = survived & uint(all(greaterThan(screenBound.xy, uvec2(2u))));
    survived = survived & uint(all(lessThan(screenBound.zw, uvec2(SETTING_SHADOW_MAP_RESOLUTION - 2u))));

    if (bool(survived)) {
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        vec2 texcoordMin = vec2(1.0);
        vec2 texcoordMax = vec2(0.0);
        texcoordMin = min(texcoordMin, vert_texcoord[0]);
        texcoordMax = max(texcoordMax, vert_texcoord[0]);
        texcoordMin = min(texcoordMin, vert_texcoord[1]);
        texcoordMax = max(texcoordMax, vert_texcoord[1]);
        texcoordMin = min(texcoordMin, vert_texcoord[2]);
        texcoordMax = max(texcoordMax, vert_texcoord[2]);
        uvec2 texcoordMinMaxPacked;
        texcoordMinMaxPacked.x = packSnorm2x16(texcoordMin);
        texcoordMinMaxPacked.y = packSnorm2x16(texcoordMax);
        #endif

        gl_Position = gl_in[0].gl_Position;
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        frag_texcoord = vert_texcoord[0];
        #if defined(SHADOW_PASS_TRANSLUCENT)
        frag_color = vert_color[0];
        #endif
        #endif
        frag_screenPos = vert_screenPos[0];
        frag_worldNormalMaterialID = vert_worldNormalMaterialID[0];
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        frag_texcoordMinMax = texcoordMinMaxPacked;
        #endif
        EmitVertex();

        gl_Position = gl_in[1].gl_Position;
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        frag_texcoord = vert_texcoord[1];
        #if defined(SHADOW_PASS_TRANSLUCENT)
        frag_color = vert_color[1];
        #endif
        #endif
        frag_screenPos = vert_screenPos[1];
        frag_worldNormalMaterialID = vert_worldNormalMaterialID[1];
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        frag_texcoordMinMax = texcoordMinMaxPacked;
        #endif
        EmitVertex();

        gl_Position = gl_in[2].gl_Position;
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        frag_texcoord = vert_texcoord[2];
        #if defined(SHADOW_PASS_TRANSLUCENT)
        frag_color = vert_color[2];
        #endif
        #endif
        frag_screenPos = vert_screenPos[2];
        frag_worldNormalMaterialID = vert_worldNormalMaterialID[2];
        #if defined(SHADOW_PASS_ALPHA_TEST) || defined(SHADOW_PASS_TRANSLUCENT)
        frag_texcoordMinMax = texcoordMinMaxPacked;
        #endif
        EmitVertex();

        EndPrimitive();
    }
}
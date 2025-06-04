layout(triangles) in;
layout(line_strip, max_vertices = 256) out;

layout(rgba32ui) uniform readonly uimage2D uimg_epipolarData;

void main() {
    int baseOffset = (LINE_PASS << 7) + (gl_PrimitiveIDIn << 7);

    for (uint i = 0u; i < 128u; i++) {
        vec4 endPoints = uintBitsToFloat(imageLoad(uimg_epipolarData, ivec2(baseOffset + i, 0)));
        gl_Position = vec4(endPoints.xy, 0.0, 1.0);
        EmitVertex();
        gl_Position = vec4(endPoints.zw, 0.0, 1.0);
        EmitVertex();
        EndPrimitive();
    }
}
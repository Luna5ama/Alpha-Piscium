#ifndef INCLUDE_debug_LightSamplingCube_glsl
#define INCLUDE_debug_LightSamplingCube_glsl
#include "/techniques/debug/LightSamplingPolygon.glsl"

void ls_clipEdge(vec3 a, vec3 b, inout LSVertices p, inout uint count) {
    if (a.z >= 0.0) ls_setVertex(p, count++, a);
    if ((a.z < 0.0 && b.z > 0.0) || (a.z > 0.0 && b.z < 0.0)) {
        vec3 v = mix(a, b, a.z / (a.z - b.z));
        v.z = 0.0;
        ls_setVertex(p, count++, v);
    }
}

LSVertices ls_cubeFace(vec3 low, vec3 high, vec3 origin, mat3 toLocal, uint axis, out uint count) {
    vec3 n = axis == 0u ? vec3(1, 0, 0) : (axis == 1u ? vec3(0, 1, 0) : vec3(0, 0, 1));
    vec3 u = n.zxy;
    vec3 v = n.yzx;
    float o = dot(origin, n);
    float lo = dot(low, n);
    float hi = dot(high, n);
    LSVertices p;
    count = 0u;
    if (o >= lo && o <= hi) return p;
    vec3 center = (low + high) * 0.5;
    center += n * ((o < lo ? lo : hi) - dot(center, n));
    u *= 0.5;
    v *= o < lo ? 0.5 : -0.5;
    vec3 a = toLocal * (center - u - v - origin);
    vec3 b = toLocal * (center + u - v - origin);
    vec3 c = toLocal * (center + u + v - origin);
    vec3 d = toLocal * (center - u + v - origin);
    ls_clipEdge(a, b, p, count);
    ls_clipEdge(b, c, p, count);
    ls_clipEdge(c, d, p, count);
    ls_clipEdge(d, a, p, count);
    if (count < 5u) ls_setVertex(p, count, p.v0);
    return p;
}

float ls_faceIntegral(LSVertices p, uint count) {
    if (count < 3u) return 0.0;
    float integral = 0.0;
    vec3 a = normalize(p.v0);
    for (uint i = 0u; i < count; ++i) {
        vec3 b = normalize(ls_vertex(p, (i + 1u) % count));
        vec3 edge = lsp_cross_stable(a, b);
        float s = length(edge);
        if (s > 0.0) integral += edge.z * (atan(s, dot(a, b)) / s);
        a = b;
    }
    return max(0.0, 0.5 * integral);
}

float ls_faceArea(LSVertices p, uint count, out vec3 areaCenter) {
    float area = 0.0;
    areaCenter = vec3(0.0);
    for (uint i = 1u; i + 1u < count; ++i) {
        vec3 b = ls_vertex(p, i), c = ls_vertex(p, i + 1u);
        float triangleArea = 0.5 * length(lsp_cross_stable(b - p.v0, c - p.v0));
        area += triangleArea;
        areaCenter += triangleArea * ((p.v0 + b + c) / 3.0);
    }
    if (area > 0.0) areaCenter /= area;
    return area;
}

float ls_faceWeight(LSVertices p, uint count) {
    if (count < 3u) return 0.0;
    float h = ls_faceIntegral(p, count);
    if (h >= 1e-5) return h;
    // Tiny horizon slivers lose precision in ellipse-area subtraction.
    // Estimate only their face-selection weight; sample their planar area below.
    vec3 areaCenter;
    float area = ls_faceArea(p, count, areaCenter);
    if (area == 0.0) return 0.0;
    vec3 faceNormal = normalize(lsp_cross_stable(p.v1 - p.v0, p.v2 - p.v0));
    float d2 = dot(areaCenter, areaCenter);
    return area * max(areaCenter.z, 0.0) * abs(dot(faceNormal, areaCenter)) / (d2 * d2);
}

vec3 ls_sampleFaceArea(LSVertices p, uint count, vec2 randoms, out float pdf) {
    vec3 areaCenter;
    float area = ls_faceArea(p, count, areaCenter);
    float target = randoms.x * area;
    vec3 b, c;
    float triangleArea;
    for (uint i = 1u; i + 1u < count; ++i) {
        b = ls_vertex(p, i); c = ls_vertex(p, i + 1u);
        triangleArea = 0.5 * length(lsp_cross_stable(b - p.v0, c - p.v0));
        if (target < triangleArea || i + 2u == count) break;
        target -= triangleArea;
    }
    float r = sqrt(target / triangleArea);
    vec3 point = p.v0 * (1.0 - r) + r * mix(b, c, randoms.y);
    float d2 = dot(point, point);
    vec3 direction = point * inversesqrt(d2);
    vec3 faceNormal = normalize(lsp_cross_stable(b - p.v0, c - p.v0));
    pdf = d2 / (area * abs(dot(faceNormal, direction)));
    return direction;
}

bool ls_sampleCube(LightSamplingNode emitter, vec3 origin, vec3 normal, vec2 randoms, out vec3 direction, out float pdf) {
    vec3 tangent = normalize(cross(normal, abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0)));
    mat3 toWorld = mat3(tangent, cross(normal, tangent), normal);
    mat3 toLocal = transpose(toWorld);
    uint count;
    LSVertices p = ls_cubeFace(emitter.boundsMinMass.xyz, emitter.boundsMaxLink.xyz, origin, toLocal, 0u, count);
    float hx = ls_faceWeight(p, count);
    p = ls_cubeFace(emitter.boundsMinMass.xyz, emitter.boundsMaxLink.xyz, origin, toLocal, 1u, count);
    float hy = ls_faceWeight(p, count);
    p = ls_cubeFace(emitter.boundsMinMass.xyz, emitter.boundsMaxLink.xyz, origin, toLocal, 2u, count);
    float hz = ls_faceWeight(p, count);
    float total = hx + hy + hz;
    if (total <= 0.0) return false;
    float target = randoms.x * total;
    uint face = target < hx ? 0u : (target < hx + hy ? 1u : 2u);
    float faceWeight = face == 0u ? hx : (face == 1u ? hy : hz);
    target -= face == 0u ? 0.0 : (face == 1u ? hx : hx + hy);
    p = ls_cubeFace(emitter.boundsMinMass.xyz, emitter.boundsMaxLink.xyz, origin, toLocal, face, count);
    vec2 faceRandoms = vec2(target / faceWeight, randoms.y);
    vec3 localDirection;
    if (ls_faceIntegral(p, count) < 1e-5) {
        localDirection = ls_sampleFaceArea(p, count, faceRandoms, pdf);
    } else {
        LSProjectedPolygon polygon = lsp_prepare_projected_solid_angle_polygon_sampling(count, p);
        localDirection = lsp_sample_projected_solid_angle_polygon(polygon, faceRandoms);
        pdf = localDirection.z / polygon.projected_solid_angle;
    }
    direction = toWorld * localDirection;
    pdf *= faceWeight / total;
    return true;
}
#endif

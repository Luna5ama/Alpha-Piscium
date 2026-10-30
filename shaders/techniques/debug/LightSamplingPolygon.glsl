// Copyright (C) 2021, Christoph Peters, Karlsruhe Institute of Technology
//
// Adapted under the three-clause BSD license; fixed five-vertex storage.
// Original: MomentsInGraphics/vulkan_renderer, polygon_sampling.glsl (2021).
//
//
// Three-clause BSD license:
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#ifndef INCLUDE_debug_LightSamplingPolygon_glsl
#define INCLUDE_debug_LightSamplingPolygon_glsl
struct LSVertices { vec3 v0, v1, v2, v3, v4; };
vec3 ls_vertex(LSVertices p, uint i) {
    if (i == 0u) return p.v0;
    if (i == 1u) return p.v1;
    if (i == 2u) return p.v2;
    if (i == 3u) return p.v3;
    return p.v4;
}
void ls_setVertex(inout LSVertices p, uint i, vec3 v) {
    if (i == 0u) p.v0 = v;
    else if (i == 1u) p.v1 = v;
    else if (i == 2u) p.v2 = v;
    else if (i == 3u) p.v3 = v;
    else p.v4 = v;
}
float lsp_positive_atan(float t) { return atan(t) + (t < 0.0 ? 3.14159265359 : 0.0); }
float lsp_mix(float a, float b, float t) { return fma(t, b, fma(-t, a, a)); }
struct LSProjectedPolygon {

    uint vertex_count;

    vec2 v0, v1, v2, v3, v4;

    vec2 e0, e1, e2, e3, e4;

    vec2 inner_ellipse_0;

    float a0, a1, a2, a3, a4;

    float projected_solid_angle;
};
vec2 lsp_get_v(LSProjectedPolygon p, uint i) {
    if (i == 0u) return p.v0;
    if (i == 1u) return p.v1;
    if (i == 2u) return p.v2;
    if (i == 3u) return p.v3;
    return p.v4;
}
void lsp_set_v(inout LSProjectedPolygon p, uint i, vec2 value) {
    if (i == 0u) p.v0 = value;
    else if (i == 1u) p.v1 = value;
    else if (i == 2u) p.v2 = value;
    else if (i == 3u) p.v3 = value;
    else p.v4 = value;
}
vec2 lsp_get_e(LSProjectedPolygon p, uint i) {
    if (i == 0u) return p.e0;
    if (i == 1u) return p.e1;
    if (i == 2u) return p.e2;
    if (i == 3u) return p.e3;
    return p.e4;
}
void lsp_set_e(inout LSProjectedPolygon p, uint i, vec2 value) {
    if (i == 0u) p.e0 = value;
    else if (i == 1u) p.e1 = value;
    else if (i == 2u) p.e2 = value;
    else if (i == 3u) p.e3 = value;
    else p.e4 = value;
}
float lsp_get_a(LSProjectedPolygon p, uint i) {
    if (i == 0u) return p.a0;
    if (i == 1u) return p.a1;
    if (i == 2u) return p.a2;
    if (i == 3u) return p.a3;
    return p.a4;
}
void lsp_set_a(inout LSProjectedPolygon p, uint i, float value) {
    if (i == 0u) p.a0 = value;
    else if (i == 1u) p.a1 = value;
    else if (i == 2u) p.a2 = value;
    else if (i == 3u) p.a3 = value;
    else p.a4 = value;
}

float lsp_kahan(float a, float b, float c, float d) {

    float cd = c * d;
    float error = fma(c, d, -cd);
    float result = fma(a, b, -cd);
    return result - error;
}

vec3 lsp_cross_stable(vec3 lhs, vec3 rhs) {
    return vec3(
        lsp_kahan(lhs.y, rhs.z, lhs.z, rhs.y),
        lsp_kahan(lhs.z, rhs.x, lhs.x, rhs.z),
        lsp_kahan(lhs.x, rhs.y, lhs.y, rhs.x)
    );
}

vec2 lsp_rotate_90(vec2 input_vector) {
    return vec2(-input_vector.y, input_vector.x);
}

bool lsp_is_inner_ellipse(vec2 ellipse) {

    return (floatBitsToUint(ellipse.x) & 0x80000000) != 0;
}

bool lsp_is_central_case(LSProjectedPolygon polygon) {
    return polygon.inner_ellipse_0.x > 0.0f;
}

vec2 lsp_ellipse_from_edge(vec3 vertex_0, vec3 vertex_1) {
    vec3 normal = lsp_cross_stable(vertex_0, vertex_1);
    float scaling = 1.0f / normal.z;
    scaling = lsp_is_inner_ellipse(normal.xy) ? -scaling : scaling;
    vec2 ellipse = normal.xy * scaling;

    ellipse.x = (normal.z != 0.0f) ? ellipse.x : uintBitsToFloat(0x7f800000u);
    return ellipse;
}

vec2 lsp_ellipse_transform(vec2 ellipse, vec2 point) {
    return fma(vec2(dot(ellipse, point)), ellipse, point);
}

float lsp_get_ellipse_det(vec2 ellipse) {
    return fma(ellipse.x, ellipse.x, fma(ellipse.y, ellipse.y, 1.0f));
}

float lsp_get_ellipse_rsqrt_det(vec2 ellipse) {
    return inversesqrt(lsp_get_ellipse_det(ellipse));
}

float lsp_get_ellipse_direction_factor_rsq(vec2 ellipse, vec2 dir) {
    float ellipse_dot_dir = dot(ellipse, dir);
    float dir_dot_dir = dot(dir, dir);
    return fma(ellipse_dot_dir, ellipse_dot_dir, dir_dot_dir);
}

float lsp_get_ellipse_direction_factor(vec2 ellipse, vec2 dir) {
    return inversesqrt(lsp_get_ellipse_direction_factor_rsq(ellipse, dir));
}

float lsp_get_ellipse_normalized_direction_factor(vec2 ellipse, vec2 normalized_dir) {
    float ellipse_dot_dir = dot(ellipse, normalized_dir);
    return inversesqrt(fma(ellipse_dot_dir, ellipse_dot_dir, 1.0f));
}

float lsp_get_area_between_ellipses_in_sector_from_tangents(float inner_rsqrt_det, float inner_tangent, float outer_rsqrt_det, float outer_tangent) {
    float inner_area = inner_rsqrt_det * lsp_positive_atan(inner_tangent);
    float result = fma(outer_rsqrt_det, lsp_positive_atan(outer_tangent), -inner_area);

    return (result > 0.0f) ? (0.5f * result) : 0.0f;
}

float lsp_get_area_between_ellipses_in_sector(vec2 inner_ellipse, float inner_rsqrt_det, vec2 outer_ellipse, float outer_rsqrt_det, vec2 dir_0, vec2 dir_1) {
    float det_dirs = max(+0.0f, dot(dir_1, lsp_rotate_90(dir_0)));
    float inner_dot = inner_rsqrt_det * dot(dir_0, lsp_ellipse_transform(inner_ellipse, dir_1));
    float outer_dot = outer_rsqrt_det * dot(dir_0, lsp_ellipse_transform(outer_ellipse, dir_1));
    return lsp_get_area_between_ellipses_in_sector_from_tangents(
        inner_rsqrt_det, det_dirs / inner_dot,
        outer_rsqrt_det, det_dirs / outer_dot);
}

float lsp_get_ellipse_area_in_sector(vec2 ellipse, vec2 dir_0, vec2 dir_1) {
    float ellipse_rsqrt_det = lsp_get_ellipse_rsqrt_det(ellipse);
    float det_dirs = max(+0.0f, dot(dir_1, lsp_rotate_90(dir_0)));
    float ellipse_dot = ellipse_rsqrt_det * dot(dir_0, lsp_ellipse_transform(ellipse, dir_1));
    float area = 0.5f * ellipse_rsqrt_det * lsp_positive_atan(det_dirs / ellipse_dot);

    return (ellipse_rsqrt_det > 0.0f) ? area : 0.0f;
}

void lsp_compare_and_swap(inout LSProjectedPolygon polygon, uint lhs, uint rhs) {
    vec2 lhs_copy = lsp_get_v(polygon, lhs);

    float normal_z = lsp_kahan(lhs_copy.x, -lsp_get_v(polygon, rhs).y, lhs_copy.y, -lsp_get_v(polygon, rhs).x);

    bool swap = (normal_z == 0.0f) ? isinf(lsp_get_e(polygon, rhs).x) : (normal_z > 0.0f);
    lsp_set_v(polygon, lhs, swap ? lsp_get_v(polygon, rhs) : lhs_copy);
    lsp_set_v(polygon, rhs, swap ? lhs_copy : lsp_get_v(polygon, rhs));
    lhs_copy = lsp_get_e(polygon, lhs);
    lsp_set_e(polygon, lhs, swap ? lsp_get_e(polygon, rhs) : lhs_copy);
    lsp_set_e(polygon, rhs, swap ? lhs_copy : lsp_get_e(polygon, rhs));
}

void lsp_sort_convex_polygon_vertices(inout LSProjectedPolygon polygon) {
    if (polygon.vertex_count == 3) {
        lsp_compare_and_swap(polygon, 1, 2);
    }
#if 5 >= 4
    else if (polygon.vertex_count == 4) {
        lsp_compare_and_swap(polygon, 1, 3);
    }
#endif
#if 5 >= 5
    else if (polygon.vertex_count == 5) {
        lsp_compare_and_swap(polygon, 2, 4);
        lsp_compare_and_swap(polygon, 1, 3);
        lsp_compare_and_swap(polygon, 1, 2);
        lsp_compare_and_swap(polygon, 0, 3);
        lsp_compare_and_swap(polygon, 3, 4);
    }
#endif

    lsp_compare_and_swap(polygon, 0, 2);
#if 5 >= 4
    if (polygon.vertex_count >= 4) {

        lsp_compare_and_swap(polygon, 2, 3);
    }
#endif

    lsp_compare_and_swap(polygon, 0, 1);
}

LSProjectedPolygon lsp_prepare_projected_solid_angle_polygon_sampling(uint vertex_count, LSVertices vertices) {
    LSProjectedPolygon polygon;

    polygon.vertex_count = vertex_count;
    polygon.inner_ellipse_0 = vec2(1.0f, 0.0f);
    lsp_set_v(polygon, 0, ls_vertex(vertices, 0).xy);
    lsp_set_e(polygon, 0, lsp_ellipse_from_edge(ls_vertex(vertices, 0), ls_vertex(vertices, 1)));
    vec2 previous_ellipse = lsp_get_e(polygon, 0);

    for (uint i = 1; i != 5; ++i) {
        lsp_set_v(polygon, i, ls_vertex(vertices, i).xy);
        if (i > 2 && i == polygon.vertex_count) break;
        vec2 ellipse = lsp_ellipse_from_edge(ls_vertex(vertices, i), ls_vertex(vertices, (i + 1) % 5));
        bool ellipse_inner = lsp_is_inner_ellipse(ellipse);

        lsp_set_e(polygon, i, ellipse_inner ? previous_ellipse : ellipse);

        polygon.inner_ellipse_0 = (lsp_is_inner_ellipse(previous_ellipse) && !ellipse_inner) ? previous_ellipse : polygon.inner_ellipse_0;
        previous_ellipse = ellipse;
    }

    vec2 ellipse = lsp_get_e(polygon, 0);
    bool ellipse_inner = lsp_is_inner_ellipse(ellipse);
    lsp_set_e(polygon, 0, ellipse_inner ? previous_ellipse : ellipse);
    polygon.inner_ellipse_0 = (lsp_is_inner_ellipse(previous_ellipse) && !ellipse_inner) ? previous_ellipse : polygon.inner_ellipse_0;

    polygon.projected_solid_angle = 0.0f;
    if (lsp_is_central_case(polygon)) {

        for (uint i = 0; i != 5; ++i) {
            if (i > 2 && i == polygon.vertex_count) break;
            lsp_set_a(polygon, i, lsp_get_ellipse_area_in_sector(lsp_get_e(polygon, i), lsp_get_v(polygon, i), lsp_get_v(polygon, (i + 1) % 5)));
            polygon.projected_solid_angle += lsp_get_a(polygon, i);
        }
    }
    else {

        lsp_sort_convex_polygon_vertices(polygon);

        vec2 inner_ellipse = polygon.inner_ellipse_0;
        float inner_rsqrt_det = lsp_get_ellipse_rsqrt_det(inner_ellipse);
        vec2 outer_ellipse;
        float outer_rsqrt_det;

        for (uint i = 0; i != 5 - 1; ++i) {
            if (i > 1 && i + 1 == polygon.vertex_count) break;
            vec2 vertex_ellipse = lsp_get_e(polygon, i);
            bool vertex_inner = lsp_is_inner_ellipse(vertex_ellipse);
            float vertex_rsqrt_det = lsp_get_ellipse_rsqrt_det(vertex_ellipse);
            if (i == 0) {
                outer_ellipse = vertex_ellipse;
                outer_rsqrt_det = vertex_rsqrt_det;
            }
            else {
                inner_ellipse = vertex_inner ? vertex_ellipse : inner_ellipse;
                inner_rsqrt_det = vertex_inner ? vertex_rsqrt_det : inner_rsqrt_det;
                outer_ellipse = vertex_inner ? outer_ellipse : vertex_ellipse;
                outer_rsqrt_det = vertex_inner ? outer_rsqrt_det : vertex_rsqrt_det;
            }
            lsp_set_a(polygon, i, lsp_get_area_between_ellipses_in_sector(
                inner_ellipse, inner_rsqrt_det, outer_ellipse, outer_rsqrt_det, lsp_get_v(polygon, i), lsp_get_v(polygon, i + 1)));
            polygon.projected_solid_angle += lsp_get_a(polygon, i);
        }
    }
    return polygon;
}

vec2 lsp_normalize_approx_and_flip(vec2 rhs, vec2 semi_circle) {
    float scaling = abs(rhs.x) + abs(rhs.y);

    scaling = uintBitsToFloat(floatBitsToUint(scaling) ^ 0x7F800000u);

    scaling = (dot(rhs, semi_circle) >= 0.0f) ? scaling : -scaling;
    return scaling * rhs;
}

vec2 lsp_solve_homogeneous_quadratic(mat2 quadratic) {
    float coeff_xy = 0.5f * (quadratic[0][1] + quadratic[1][0]);
    float sqrt_discriminant = sqrt(max(0.0f, coeff_xy * coeff_xy - quadratic[0][0] * quadratic[1][1]));
    float scaled_root = abs(coeff_xy) + sqrt_discriminant;
    return (coeff_xy >= 0.0f) ? vec2(scaled_root, -quadratic[0][0]) : vec2(quadratic[1][1], scaled_root);
}

vec2 lsp_sample_sector_between_ellipses(vec2 random_numbers, float target_area, vec2 inner_ellipse, vec2 outer_ellipse, vec2 dir_0, vec2 dir_1, uint iteration_count) {

    vec2 quad_dirs[3];
    quad_dirs[0] = normalize(dir_0);
    quad_dirs[2] = normalize(dir_1);
    quad_dirs[1] = quad_dirs[0] + quad_dirs[2];

    float normalization_factor[2][3] = {
        {
            lsp_get_ellipse_normalized_direction_factor(inner_ellipse, quad_dirs[0]),
            lsp_get_ellipse_direction_factor(inner_ellipse, quad_dirs[1]),
            lsp_get_ellipse_normalized_direction_factor(inner_ellipse, quad_dirs[2])
        },
        {
            lsp_get_ellipse_normalized_direction_factor(outer_ellipse, quad_dirs[0]),
            lsp_get_ellipse_direction_factor(outer_ellipse, quad_dirs[1]),
            lsp_get_ellipse_normalized_direction_factor(outer_ellipse, quad_dirs[2])
        }
    };

    float sector_areas[2] = {
        normalization_factor[1][0] * normalization_factor[1][1] - normalization_factor[0][0] * normalization_factor[0][1],
        normalization_factor[1][1] * normalization_factor[1][2] - normalization_factor[0][1] * normalization_factor[0][2]
    };

    float target_quad_area = lsp_mix(-sector_areas[0], sector_areas[1], random_numbers[0]);
    quad_dirs[2] = (target_quad_area <= 0.0f) ? quad_dirs[0] : quad_dirs[2];
    normalization_factor[0][2] = (target_quad_area <= 0.0f) ? normalization_factor[0][0] : normalization_factor[0][2];
    normalization_factor[1][2] = (target_quad_area <= 0.0f) ? normalization_factor[1][0] : normalization_factor[1][2];
    target_quad_area += (target_quad_area <= 0.0f) ? sector_areas[0] : -sector_areas[1];

    target_quad_area *= abs(determinant(mat2(quad_dirs[1], quad_dirs[2])));

    vec2 quad_normals[2] = {
        quad_dirs[1] * normalization_factor[0][1] + quad_dirs[2] * normalization_factor[0][2],
        quad_dirs[1] * normalization_factor[1][1] + quad_dirs[2] * normalization_factor[1][2]
    };
    quad_normals[0] = lsp_ellipse_transform(inner_ellipse, quad_normals[0]);
    quad_normals[1] = lsp_ellipse_transform(outer_ellipse, quad_normals[1]);

    float quad_offsets[2] = {
        dot(quad_normals[0], quad_dirs[1]) * normalization_factor[0][1],
        dot(quad_normals[1], quad_dirs[1]) * normalization_factor[1][1]
    };

    mat2 quadratic = outerProduct((quad_offsets[1] * normalization_factor[1][2]) * lsp_rotate_90(quad_dirs[2]), quad_normals[0]);
    quadratic -= outerProduct((quad_offsets[0] * normalization_factor[0][2]) * lsp_rotate_90(quad_dirs[2]) + target_quad_area * quad_normals[0], quad_normals[1]);
    vec2 current_dir = lsp_solve_homogeneous_quadratic(quadratic);

#ifndef USE_BIASED_PROJECTED_SOLID_ANGLE_SAMPLING

    float acceptable_error = 1.0e-5f;
    iteration_count = (abs(random_numbers.x - 0.5f) <= 0.5f - acceptable_error) ? iteration_count : 0;

    float inner_rsqrt_det = lsp_get_ellipse_rsqrt_det(inner_ellipse);
    float outer_rsqrt_det = lsp_get_ellipse_rsqrt_det(outer_ellipse);

    for (uint i = 0; i != iteration_count; ++i) {

        current_dir = lsp_normalize_approx_and_flip(current_dir, quad_dirs[1]);

        vec2 inner_dir = lsp_ellipse_transform(inner_ellipse, current_dir);
        vec2 outer_dir = lsp_ellipse_transform(outer_ellipse, current_dir);

        float det_dirs = max(+0.0f, dot(current_dir, lsp_rotate_90(quad_dirs[0])));
        float error = target_area - lsp_get_area_between_ellipses_in_sector_from_tangents(
            inner_rsqrt_det, det_dirs / (inner_rsqrt_det * dot(quad_dirs[0], inner_dir)),
            outer_rsqrt_det, det_dirs / (outer_rsqrt_det * dot(quad_dirs[0], outer_dir)));

        quadratic = outerProduct(inner_dir - outer_dir, lsp_rotate_90(current_dir)) - outerProduct((2.0f * error) * inner_dir, outer_dir);
        current_dir = lsp_solve_homogeneous_quadratic(quadratic);
    }
#endif

    current_dir = (dot(current_dir, quad_dirs[1]) >= 0.0f) ? current_dir : -current_dir;

    float inner_factor = 1.0f / lsp_get_ellipse_direction_factor_rsq(inner_ellipse, current_dir);
    float outer_factor = 1.0f / lsp_get_ellipse_direction_factor_rsq(outer_ellipse, current_dir);
    current_dir *= sqrt(lsp_mix(inner_factor, outer_factor, random_numbers[1]));
    return current_dir;
}

vec3 lsp_sample_projected_solid_angle_polygon(LSProjectedPolygon polygon, vec2 random_numbers) {
    float target_projected_solid_angle = random_numbers[0] * polygon.projected_solid_angle;

    vec3 sampled_dir;
    vec2 outer_ellipse;
    vec2 dir_0;
    if (lsp_is_central_case(polygon)) {

        for (uint i = 0; i != 5; ++i) {
            if (i > 0) {
                target_projected_solid_angle -= lsp_get_a(polygon, i - 1);
            }
            outer_ellipse = lsp_get_e(polygon, i);
            dir_0 = lsp_get_v(polygon, i);
            if ((i >= 2 && i + 1 == polygon.vertex_count) || target_projected_solid_angle < lsp_get_a(polygon, i))
                break;
        }

        float sqrt_det = sqrt(lsp_get_ellipse_det(outer_ellipse));
        float angle = 2.0f * target_projected_solid_angle * sqrt_det;
        sampled_dir.xy = (cos(angle) * sqrt_det) * dir_0 + sin(angle) * lsp_rotate_90(lsp_ellipse_transform(outer_ellipse, dir_0));

        sampled_dir.xy *= sqrt(random_numbers[1] / lsp_get_ellipse_direction_factor_rsq(outer_ellipse, sampled_dir.xy));
    }

    else {

        float sector_projected_solid_angle;
        vec2 inner_ellipse = polygon.inner_ellipse_0;
        vec2 dir_1;

        for (uint i = 0; i != 5 - 1; ++i) {
            vec2 vertex_ellipse = lsp_get_e(polygon, i);
            if (i == 0) {
                outer_ellipse = vertex_ellipse;
            }
            else {
                target_projected_solid_angle -= lsp_get_a(polygon, i - 1);
                bool vertex_inner = lsp_is_inner_ellipse(vertex_ellipse);
                inner_ellipse = vertex_inner ? vertex_ellipse : inner_ellipse;
                outer_ellipse = vertex_inner ? outer_ellipse : vertex_ellipse;
            }
            dir_0 = lsp_get_v(polygon, i);
            dir_1 = lsp_get_v(polygon, i + 1);
            sector_projected_solid_angle = lsp_get_a(polygon, i);
            if ((i >= 1 && i + 2 == polygon.vertex_count) || target_projected_solid_angle < sector_projected_solid_angle)
                break;
        }

        random_numbers[0] = target_projected_solid_angle / sector_projected_solid_angle;
        sampled_dir.xy = lsp_sample_sector_between_ellipses(random_numbers, target_projected_solid_angle, inner_ellipse, outer_ellipse, dir_0, dir_1, 2);
    }

    sampled_dir.z = sqrt(max(0.0f, fma(-sampled_dir.x, sampled_dir.x, fma(-sampled_dir.y, sampled_dir.y, 1.0f))));
    return sampled_dir;
}


#endif

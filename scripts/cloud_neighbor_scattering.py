"""Double-scattering change when a second cloud is placed beside the first.

Both clouds are homogeneous unit spheres in vacuum. The camera is on the -Z
side. The sun lights them from +Z or +Y. Only scattered radiance is measured.
The spheres have disjoint solar and camera projections. Single scattering and
paths with both events in the first sphere are identical with the neighbor off
or on. The remaining second-order paths go from the neighbor to the first cloud.
Run with: python scripts/cloud_neighbor_scattering.py
"""

import math


SEPARATION = 2.5
EXTINCTION = 2.0
ALBEDO = 0.999
ASYMMETRY = 0.8


def phase(cosine):
    g = ASYMMETRY
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cosine) ** 1.5)


def exit_distance(projection, radius_squared):
    return math.sqrt(projection**2 + 1 - radius_squared) - projection


def single_scattering(ray_x, sun_z):
    half_chord = math.sqrt(1 - ray_x * ray_x)
    if sun_z == 1:
        optical_depth = 2 * EXTINCTION * half_chord
        return ALBEDO * phase(1) * optical_depth * math.exp(-optical_depth)
    steps = 4096
    step = 2 * half_chord / steps
    transmission = 0.0
    for i in range(steps):
        z = (i + 0.5) * step - half_chord
        sun_exit = exit_distance(sun_z * z, ray_x**2 + z**2)
        transmission += math.exp(-EXTINCTION * (half_chord + z + sun_exit))
    return EXTINCTION * ALBEDO * phase(sun_z) * step * transmission


def neighbor_second_order(ray_x, sun_y, sun_z, resolution):
    # Integrate along the camera chord, over the solid angle of the neighbor,
    # and along the chord inside the neighbor for each such direction.
    samples = [(i + 0.5) / resolution for i in range(resolution)]
    azimuths = [(math.cos(math.tau * u), math.sin(math.tau * u)) for u in samples]
    half_chord = math.sqrt(1 - ray_x * ray_x)
    total = 0.0
    for view_u in samples:
        z = (2 * view_u - 1) * half_chord
        to_camera = z + half_chord
        distance = math.hypot(SEPARATION - ray_x, z)
        axis_x, axis_z = (SEPARATION - ray_x) / distance, -z / distance
        cap_width = 1 - math.sqrt(1 - 1 / distance**2)
        for polar_u in samples:
            cosine = 1 - polar_u * cap_width
            sine = math.sqrt(1 - cosine**2)
            half_neighbor = math.sqrt(1 - (distance * sine) ** 2)
            near = distance * cosine - half_neighbor
            for ca, sa in azimuths:
                dx = axis_x * cosine - axis_z * sine * ca
                dy = sine * sa
                dz = axis_z * cosine + axis_x * sine * ca
                projection = ray_x * dx + z * dz
                in_main = exit_distance(projection, ray_x**2 + z**2)
                # The physical direction of the first scattered ray is (-dx, -dy, -dz).
                phase_pair = phase(sun_y * dy + sun_z * dz) * phase(dz)
                chord_sum = 0.0
                for light_u in samples:
                    in_neighbor = 2 * half_neighbor * light_u
                    travel = near + in_neighbor
                    bx = ray_x + travel * dx - SEPARATION
                    by = travel * dy
                    bz = z + travel * dz
                    sun_exit = exit_distance(sun_y * by + sun_z * bz, bx * bx + by * by + bz * bz)
                    chord_sum += math.exp(-EXTINCTION * (sun_exit + in_neighbor))
                transmittance = math.exp(-EXTINCTION * (in_main + to_camera))
                total += cap_width * (2 * half_neighbor) * phase_pair * transmittance * chord_sum
    measure = (2 * half_chord) * math.tau / resolution**4
    return (EXTINCTION * ALBEDO) ** 2 * measure * total


if __name__ == '__main__':
    print('sun  ray_x      tau   single/E_sun    delta_double/E_sun (16, 32, 64 samples per dimension)')
    for label, sun_y, sun_z in (('+Z', 0, 1), ('+Y', 1, 0)):
        for x in (0.0, math.sqrt(1 - 0.25**2)):
            tau = 2 * EXTINCTION * math.sqrt(1 - x * x)
            estimates = [neighbor_second_order(x, sun_y, sun_z, n) for n in (16, 32, 64)]
            print(f'{label}   {x:.6f}  {tau:5.2f}  {single_scattering(x, sun_z):.8f}    '
                  + '   '.join(f'{v:.8g}' for v in estimates))

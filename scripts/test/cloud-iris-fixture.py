"""Build a Vibris source patch from a sealed cloud-reference manifest and a production shader revision."""
import argparse
import difflib
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CLOUD = 'shaders/techniques/atmospherics/clouds/'
COLUMN = 'fb8c36487e24b20bc5f50f8b1890a3849a1f7306'
TRANSPORT = 'c09cfdc76e79cb3e4e986d16e29730777283e78d'


def glvec(values):
    return 'vec3(' + ', '.join(str(float(v)) for v in values) + ')'


def body(source, signature, replacement):
    start = source.index('{', source.index(signature)) + 1
    end = source.index('\n}', start)
    return source[:start] + '\n' + replacement.strip('\n') + source[end:]


def section(source, start, end, replacement):
    a = source.index(start)
    b = source.index(end, a)
    return source[:a] + replacement + source[b:]


def write_patch(path, before, edited):
    diff = []
    for name, source in edited.items():
        diff.extend(difflib.unified_diff(before[name].splitlines(True), source.splitlines(True),
                                         'a/' + name, 'b/' + name))
    path.write_bytes(''.join(diff).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--method', choices=('column', 'transport'), required=True)
    parser.add_argument('--revision', help='exact Git revision for a transport candidate')
    parser.add_argument('--sun', choices=('side', 'back'))
    parser.add_argument('--neighbor', action='store_true')
    parser.add_argument('--sky', action='store_true')
    parser.add_argument('--world-origin', nargs=2, type=float, metavar=('X_M', 'Z_M'),
                        help='place the fixed density in the game, retaining the normal perspective, lighting, depth and composition')
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text(encoding='utf-8'))
    if m['fixture'] != 'ellipsoid-tower-v1' or m['camera']['direction'] != [0, 0, 1]:
        raise ValueError('this calibration uses the forward orthographic tower fixture')
    legacy = args.method == 'column'
    if legacy and args.revision:
        parser.error('--revision applies to transport candidates')
    if args.world_origin and (args.sky or args.sun):
        parser.error('the native view uses the game sky and sun')
    if not args.world_origin and not args.sun:
        parser.error('choose --sun for the orthographic fixture or --world-origin for a native view')
    revision = COLUMN if legacy else args.revision or TRANSPORT
    before = {}
    edited = {}

    def read(path):
        before[path] = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=ROOT).decode('utf-8')
        return before[path]

    shapes = m['ellipsoids_center_radius']
    # Atmosphere distances are kilometres. Keep the analytic reference in metres.
    density = 1000 * m['density_inside_union']
    sun = glvec(m['light_directions_to_sun'][1 if args.sun == 'back' else 0])
    angle = math.radians(m['sun_angular_radius_degrees'])
    sky = m['upper_hemisphere_sky_radiance'] if args.sky else 0.
    width, height = m['shape'][4], m['shape'][3]
    xs, ys, cz = m['camera']['x'], m['camera']['y_top_bottom'], m['camera']['z']
    near = min(c[2]-r[2] for c, r in shapes)
    far = max(c[2]+r[2] for c, r in shapes)
    geometry = """
bool fixtureInside(vec3 p) {
    vec3 d;
"""
    column = """
float fixtureColumnTop(vec2 p) {
    float top = 0.0;
    vec2 d;
"""
    for c, r in shapes:
        geometry += f'    d = (p - {glvec(c)}) / {glvec(r)};\n    if (dot(d, d) < 1.0) return true;\n'
        column += (f'    d = (p - {glvec(c)}.xz) / {glvec(r)}.xz;\n'
                   f'    if (dot(d, d) < 1.0) top = max(top, {c[1]:.1f} + {r[1]:.1f} * sqrt(1.0 - dot(d, d)));\n')
    geometry += '    return false;\n}\n'
    column += '    return top;\n}\n'
    neighbor = glvec(m['neighbor_offset_m'])
    inside = 'fixtureInside(p)'
    base_x, base_z = args.world_origin or (0., 0.)
    xz = f'(rayPosXZ * 1000.0 - vec2({base_x}, {base_z}))'
    coverage = f'fixtureColumnTop({xz})'
    if args.neighbor:
        inside += f' || fixtureInside(p - {neighbor})'
        coverage = f'max({coverage}, fixtureColumnTop({xz} - {neighbor}.xz))'
    helper = f"""
float fixtureDensity(vec3 rayPos) {{
    vec3 p = (rayPos - vec3({base_x*.001}, getAtmosphereParameters().bottom + SETTING_CLOUDS_CU_HEIGHT, {base_z*.001})) * 1000.0;
    return {inside} ? {density} : 0.0;
}}
"""
    if not args.world_origin:
        helper += f"""
vec3 fixtureOrigin(ivec2 texel, ivec2 size) {{
    vec2 bin = floor((vec2(texel) + 0.5) * vec2({width}.0, {height}.0) / vec2(size)) + 0.5;
    vec2 xy = vec2({float(xs[0])}, {float(ys[1])}) + bin * vec2({(xs[1]-xs[0])/width}, {(ys[0]-ys[1])/height});
    return vec3(xy, {float(cz)}) * 0.001 + vec3(0.0, getAtmosphereParameters().bottom + SETTING_CLOUDS_CU_HEIGHT, 0.0);
}}
"""
    dpath = CLOUD + 'Cumulus.glsl'
    density_source = read(dpath)
    if legacy:
        density_source = density_source.replace('float clouds_cu_baseCoverage(vec2 pos)', geometry + column + helper + '\nfloat clouds_cu_baseCoverage(vec2 rayPosXZ)')
        density_source = body(density_source, 'float clouds_cu_baseCoverage(', f'    return {coverage} * 0.001 / SETTING_CLOUDS_CU_THICKNESS;')
        density_source = body(density_source, 'bool clouds_cu_density(', """
    densityOut = fixtureDensity(rayPos) / (96.0 * SETTING_CLOUDS_CU_DENSITY);
    densityLodOut = densityOut;
    coverageOut = clouds_cu_baseCoverage(rayPos.xz);
    return densityOut > 0.0;""")
        path = CLOUD + 'Mediums.glsl'
        medium = read(path)
        sigma_t = m['extinction_per_metre']/m['density_inside_union']
        for coefficient, val in [('EXTINCTION', sigma_t), ('SCATTERING', m['single_scattering_albedo']*sigma_t)]:
            medium = re.sub(rf'(const vec3 CLOUDS_CU_{coefficient} = )[^;]+;', rf'\g<1>vec3({val});', medium)
        edited[path] = medium
    else:
        density_source = density_source.replace('float clouds_cu_density(', geometry + helper + '\nfloat clouds_cu_density(')
        density_source = body(density_source, 'float clouds_cu_density(', '    return fixtureDensity(rayPos);')
        density_source = re.sub(r'(const float CU_DENSITY_(?:SCALE|MAX) = )[^;]+;', rf'\g<1>{density};', density_source)
        density_source = body(density_source, 'vec3 clouds_cu_wind(', '    return vec3(0.0);')
    edited[dpath] = density_source
    if args.world_origin:
        write_patch(args.output, before, edited)
        print(revision, args.output)
        return
    rpath = CLOUD + 'RenderVolumetric.comp.glsl'
    render = read(rpath)
    render = re.sub(r'shared_worldGroupCheck = hiz_groupSkyCheck4x4\([^;]+;', 'shared_worldGroupCheck = true;', render)
    if legacy:
        main_ray = f"""    AtmosphereParameters atmosphere = getAtmosphereParameters();
    CloudMainRayParams mainRayParams;
    mainRayParams.rayStart = fixtureOrigin(texelPosDownScale, renderSize);
    mainRayParams.rayDir = vec3(0.0, 0.0, 1.0);
    mainRayParams.rayEnd = mainRayParams.rayStart + mainRayParams.rayDir;
    mainRayParams.rayStartHeight = length(mainRayParams.rayStart);
    mainRayParams.rayEndHeight = length(mainRayParams.rayEnd);
    vec3 lightDir = {sun};
    vec3 lightIlluminance = vec3(1.0);
"""
        render = section(render, '    vec2 texelPosF =', '    CloudRenderParams renderParams =', main_ray)
        render = section(render, '        float cuMidHeight =', '            #define CLOUDS_CU_DENSITY',
                         f'        float cuOrigin2RayStart = {(near-cz)*.001};\n        float cuRayLen = {(far-near)*.001};\n        const float CLOUDS_CU_MAX_RAY_LENGTH = 50.0;\n        {{\n')
        path = CLOUD + 'amblut/Gather.comp.glsl'
        edited[path] = re.sub(r'vec3 inSctr = [^;]+;', f'vec3 inSctr = vec3(rayDir.y > 0.0 ? {sky} : 0.0);', read(path))
        render = render.replace('SUN_ANGULAR_RADIUS * 4.0', f'{angle} * 4.0')
        path = CLOUD + 'Common.glsl'
        edited[path] = re.sub(r'vec3 tLightToSample = [^;]+;', 'vec3 tLightToSample = vec3(1.0);', read(path))
    else:
        render = section(render, '    vec2 screenPos =', '    if (intersects &&',
                         f"""    vec3 origin = fixtureOrigin(texelPos, renderSize);
    vec3 direction = vec3(0.0, 0.0, 1.0);
    vec2 layerRange = atmosphere.bottom + vec2(SETTING_CLOUDS_CU_HEIGHT, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS);
    vec2 interval = vec2({(near-cz)*.001}, {(far-cz)*.001});
    bool intersects = true;
""")
        render = section(render, '    LightParameters sunParams =', '    vec3 pos = origin + direction * distance;',
                         '    ScatteringResult air = scatteringResult_init();\n')
        render = re.sub(r'vec3 atmosphereTr = [^;]+;', 'vec3 atmosphereTr = vec3(1.0);', render)
        path = CLOUD + 'Transport.glsl'
        transport = body(read(path), 'vec3 ct_boundary(', f'    return vec3(omega.y < 0.0 ? {sky} : 0.0);')
        edited[path] = body(transport, 'vec3 ct_worldCenter(', f'    return vec3(0.0, 0.0, {cz*.001});')
        path = CLOUD + 'LocalTransport.glsl'
        edited[path] = body(read(path), 'vec3 ct_beamIncident(', '    return light == 0 ? vec3(1.0) : vec3(0.0);')
        path = 'shaders/pass/begin/CloudTransportPrepare.comp.glsl'
        edited[path] = re.sub(r'vec3 airTr = [^;]+;', 'vec3 airTr = vec3(1.0);', read(path))
    edited[rpath] = render
    if not legacy:
        for path in (rpath, CLOUD + 'LocalTransport.glsl', 'shaders/pass/begin/CloudTransportPrepare.comp.glsl'):
            for key, value in (('uval_sunDirWorld', sun), ('uval_moonDirWorld', 'vec3(0.0, -1.0, 0.0)'),
                               ('SUN_ILLUMINANCE', 'vec3(1.0)'), ('MOON_ILLUMINANCE', 'vec3(0.0)'),
                               ('SUN_ANGULAR_RADIUS', str(angle))):
                edited[path] = edited[path].replace(key, value)
    # Capture the half values the normal cloud pass stored. DirectLighting
    # overwrites colortex0 later. Native-view patches omit this measurement.
    render = edited[rpath].replace('shared bool shared_worldGroupCheck;',
                                   'layout(rgba16f) uniform writeonly image2D uimg_main;\nshared bool shared_worldGroupCheck;')
    pos = 'texelPosDownScale' if legacy else 'texelPos'
    store = f'transient_lowCloudRender_store({pos}, packedOutput);'
    edited[rpath] = render.replace(store, store + f'\n    imageStore(uimg_main, {pos}, vec4(unpackHalf2x16(packedOutput.x), unpackHalf2x16(packedOutput.y)));')
    write_patch(args.output, before, edited)
    print(revision, args.output)


if __name__ == '__main__':
    main()

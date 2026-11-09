"""Fit an experimental sparse Bloom shape from a linear grayscale PNG or PSF function.

Python targets define psf(x, y), where x and y are full-resolution pixel
offset grids. PNG samples are raw scalar weights. Both use x right, y down
and an odd-sized image centered on its middle texel.
"""
import argparse
import hashlib
import json
import os
import re
import runpy
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.signal import correlate2d


def soft_strip(x, y):
    """An asymmetric soft horizontal strip, quantized for the PNG example."""
    profile = .7 * np.exp(-.5 * ((x - 10) / 8)**2) + .3 * np.exp(-.5 * ((x + 9) / 5)**2)
    return np.rint(65535 * profile * np.exp(-.5 * (y / 1.6)**2))


def read_target(args):
    if args.image:
        with Image.open(args.image) as image:
            if image.format != "PNG" or image.mode not in {"L", "I", "I;16"} or "transparency" in image.info:
                raise ValueError("Expected an opaque 8-bit or 16-bit grayscale PNG")
            target = np.asarray(image, dtype=np.float64)
    else:
        if args.size <= 0 or args.size % 2 == 0:
            raise ValueError("Expected a positive odd function canvas size")
        axis = np.arange(args.size) - args.size // 2
        x, y = np.meshgrid(axis, axis)
        psf = runpy.run_path(str(args.function))["psf"] if args.function else soft_strip
        target = np.asarray(psf(x, y), dtype=np.float64)
        if target.shape != x.shape:
            raise ValueError("psf(x, y) must return the same shape as its input grids")
    if target.ndim != 2 or any(n % 2 == 0 for n in target.shape):
        raise ValueError("Expected positive odd target dimensions")
    if not np.isfinite(target).all() or target.min() < 0 or target.max() <= 0:
        raise ValueError("Expected finite nonnegative weights with positive mass")
    target = target / target.max()
    return target / target.sum()


def bloom_taps():
    source = Path(__file__).resolve().parents[1] / "shaders/techniques/BloomTaps.glsl"
    down, up = source.read_text(encoding="utf-8").split("#else", 1)
    pattern = r"bloom_sample\(center \+ vec2\(([-.\d]+), ([-.\d]+)\), tile\) \* ([-.\d]+);"
    kernels = [np.array(re.findall(pattern, block), dtype=float) for block in (down, up)]
    for kernel in kernels:
        if not len(kernel) or abs(kernel[:, 2].sum() - 1) > 1e-7:
            raise ValueError(f"Expected normalized tap data in {source}")
        kernel[:, 2] /= kernel[:, 2].sum()
    return kernels


def phase_carriers(depth, runtime=False):
    interp = runpy.run_path(str(Path(__file__).with_name("fit-bloom.py")))["interpolation"]
    step = 2**depth
    sizes = [64 >> i for i in range(depth + 1)]
    centered = [(0, 0, 1.)]
    down = bloom_taps()[0] if runtime else [(x, y, .25) for x in (-.75, .75) for y in (-.75, .75)]
    operations = []

    def stage(src, center, taps):
        offsets = {offset for x, y, _ in taps for offset in (x, y)}
        matrices = {offset: interp(src, center + offset) for offset in offsets}
        return [(matrices[x], matrices[y], weight) for x, y, weight in taps]

    for src, dst in zip(sizes, sizes[1:]):
        operations.append(stage(src, 2 * np.arange(dst) + .5, down))
    for src, dst in zip(sizes[:0:-1], sizes[-2::-1]):
        operations.append(stage(src, .5 * np.arange(dst) - .25, centered))
    radius = 3 * step
    for y in range(step):
        for x in range(step):
            image = np.zeros((64, 64))
            image[32 + y, 32 + x] = 1
            for taps in operations:
                image = sum(w * (mx @ (my @ image).T).T for mx, my, w in taps)
            carrier = image[32+y-radius:33+y+radius, 32+x-radius:33+x+radius]
            assert abs(carrier.sum() - image.sum()) < 1e-12
            assert abs(carrier.sum() - 1) < 1e-7
            yield (y, x), carrier


def shape_response(carrier, shape, step, gather, weights):
    center = np.array(shape) // 2
    radius = carrier.shape[0] // 2
    image = np.zeros(shape)
    for (gx, gy), weight in zip(gather, weights):
        oy, ox = center - step * np.array([gy, gx])
        image[oy-radius:oy+radius+1, ox-radius:ox+radius+1] += weight * carrier
    return image


def fit(target, depth, budget, runtime=False):
    step = 2**depth
    carriers = list(phase_carriers(depth, runtime))
    training = [h for (y, x), h in carriers if (y + x) % 2 == 0]
    radius = training[0].shape[0] // 2
    target = np.pad(target, 2 * radius)
    center = np.array(target.shape) // 2
    corr = np.mean([correlate2d(target, h, mode="same") for h in training], axis=0)
    points = np.argwhere(corr > corr.max() * 1e-7)
    points = points[np.all((points - center) % step == 0, axis=1)]
    offsets = (points - center) // step
    corr = corr[tuple(points.T)]
    ac = np.mean([correlate2d(h, h, mode="full") for h in training], axis=0)
    energy = np.sum(target**2)

    def gram(left, right):
        y, x = ((offsets[left, None] - offsets[None, right]) * step + 2 * radius).transpose(2, 0, 1)
        valid = (y >= 0) & (x >= 0) & (y < ac.shape[0]) & (x < ac.shape[1])
        result = np.zeros(valid.shape)
        result[valid] = ac[y[valid], x[valid]]
        return result

    active = [int(np.argmax(corr))]
    weights = np.ones(1)
    all_indices = np.arange(len(offsets))
    while len(active) < min(budget, len(offsets)):
        gradient = gram(all_indices, active) @ weights - corr
        reduced = gradient[active] @ weights
        gradient[active] = np.inf
        i = int(np.argmin(gradient))
        if gradient[i] >= reduced - 1e-10:
            break
        active.append(i)
        a = gram(active, active) / energy
        b = corr[active] / energy

        def objective(w):
            g = a @ w
            return .5 * w @ g - b @ w, g - b

        result = minimize(objective, np.r_[weights, 0.], jac=True, method="SLSQP",
                          bounds=[(0., 1.)] * len(active),
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                        "jac": lambda w: np.ones_like(w)}],
                          options={"ftol": 1e-8, "maxiter": 180})
        if not result.success:
            raise RuntimeError(result.message)
        weights = result.x

    keep = weights > 1e-8
    weights = weights[keep]
    weights /= weights.sum()
    gather = -offsets[np.asarray(active)[keep]][:, ::-1]
    return score(target, carriers, depth, budget, gather, weights)


def score(target, carriers, depth, budget, gather, weights):
    step = 2**depth
    center = np.array(target.shape) // 2
    energy = np.sum(target**2)
    norms = {"fit": [], "held_out": [], "held_out_rotated_target": []}
    centroid = []
    yy, xx = np.indices(target.shape) - center[:, None, None]
    for (y, x), h in carriers:
        k = shape_response(h, target.shape, step, gather, weights)
        assert abs(k.sum() - 1) < 1e-10
        label = "held_out" if (y + x) % 2 else "fit"
        norms[label].append(np.linalg.norm(k - target) / np.sqrt(energy))
        if label == "held_out":
            norms["held_out_rotated_target"].append(np.linalg.norm(k - target[::-1, ::-1]) / np.sqrt(energy))
            centroid.append([np.sum(xx * k), np.sum(yy * k)])
    return {
        "carrier_mip": depth,
        "tap_budget": budget,
        "active_taps": len(weights),
        "gather_offsets_xy_mip_texels": gather.tolist(),
        "weights": weights.tolist(),
        "target_centroid_xy_full_res_px": [float(np.sum(xx * target)), float(np.sum(yy * target))],
        "held_out_centroid_xy_full_res_px": np.mean(centroid, axis=0).tolist(),
        "relative_l2": {key: {"phases": len(v), "mean": float(np.mean(v)), "max": float(max(v))}
                        for key, v in norms.items()},
    }


def fit_builtin(carriers, wide, narrow, rays, offsets):
    span = 2 * max(offsets) + 8
    axis = np.arange(-span, span + 1)
    x, y = np.meshgrid(axis, axis if len(rays) == 4 else np.arange(-8, 9))
    distances = 2 * np.hypot(*rays[0]) * np.array([0, *offsets])
    end = distances[-1]

    def envelope(distance):
        t = np.clip((end - np.abs(distance)) / (end * .28), 0, 1)
        taper = t * t * (3 - 2 * t)
        tail = np.exp(-.5 * (distance / wide)**2)
        if len(rays) == 4:
            return tail * taper
        core = np.exp(-.5 * (distance / 22.)**2)
        return (.3 * core / 22. + .7 * tail / wide) * taper

    if len(rays) == 4:
        u, v = (x + y) / np.sqrt(2), (x - y) / np.sqrt(2)
        target = envelope(u) * np.exp(-.5 * (v / narrow)**2)
        target += envelope(v) * np.exp(-.5 * (u / narrow)**2)
    else:
        target = envelope(x) * np.exp(-.5 * (y / narrow)**2)
    target /= target.sum()

    # Fit one weight per distance to keep the rays symmetric.
    groups = [[(0, 0)]] + [[(d * x, d * y) for x, y in rays] for d in offsets]
    train = [h for (y, x), h in carriers if (y + x) % 2 == 0]
    design = np.stack([
        np.concatenate([shape_response(h, target.shape, 2, group, np.full(len(group), 1 / len(group))).ravel()
                        for h in train]) for group in groups
    ], axis=1)
    reference = np.tile(target.ravel(), len(train))
    energy = reference @ reference
    a = (design.T @ design) / energy
    b = (design.T @ reference) / energy

    def objective(w):
        g = a @ w
        return .5 * w @ g - b @ w, g - b

    prior = envelope(distances) * np.array([len(group) for group in groups])
    prior /= prior.sum()
    # Bound tail weights by the taper even where the carrier cannot match the target exactly.
    result = minimize(objective, prior, jac=True, method="SLSQP",
                      bounds=[(0., min(1., 2 * w)) for w in prior],
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                    "jac": lambda w: np.ones_like(w)}],
                      options={"ftol": 1e-12, "maxiter": 180})
    if not result.success:
        raise RuntimeError(result.message)
    taps = [(pos, weight / len(group)) for group, weight in zip(groups, result.x)
            if weight > 1e-8 for pos in group]
    gather, weights = zip(*taps)
    weights = np.asarray(weights)
    weights /= weights.sum()
    return score(target, carriers, 1, sum(map(len, groups)), np.asarray(gather), weights)


def bilinear_taps(fit):
    remaining = {tuple(pos): weight for pos, weight in zip(fit["gather_offsets_xy_mip_texels"], fit["weights"])}
    for x, y in sorted(remaining, key=lambda pos: pos[::-1]):
        if (x, y) not in remaining:
            continue
        weight = remaining.pop((x, y))
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in remaining:
                neighbor = remaining.pop((x + dx, y + dy))
                frac = neighbor / (weight + neighbor)
                yield x + dx * frac, y + dy * frac, weight + neighbor
                break
        else:
            yield x, y, weight


def glsl_shape(fit, gain):
    lines = [f"const float bloom_shapeGain = {gain};",
             "vec3 bloom_shape(vec2 center, vec4 tile) {"]
    for j, (x, y, w) in enumerate(bilinear_taps(fit)):
        prefix = "vec3 result =" if j == 0 else "result +="
        # PNG/PSF Y points down. Atlas Y points up.
        lines.append(f"    {prefix} bloom_sample(center + vec2({x:.9f}, {-y:.9f}), tile) * {w:.9f};")
    return [*lines, "    return result;", "}"]


def emit_builtins(output, report):
    carriers = list(phase_carriers(1, runtime=True))
    horizontal = [(1, 0), (-1, 0)]
    presets = [
        ("star", .375, 30., 1.23, [(x, y) for x in (-1, 1) for y in (-1, 1)], range(1, 29)),
        ("strip", .5, 61., 1.225, horizontal, range(1, 73)),
    ]
    lines = ["// Generated by scripts/fit-bloom-shape.py --builtins."]
    details = {}
    for i, (name, gain, wide, narrow, rays, offsets) in enumerate(presets, start=1):
        fit = fit_builtin(carriers, wide, narrow, rays, offsets)
        details[name] = {"setting": i, "gain": gain, "target_sigma_px": [wide, narrow],
                         "shader_fetches": len(list(bilinear_taps(fit))), **fit}
        condition = "if" if i == 1 else "elif"
        lines += [f"#{condition} SETTING_BLOOM_SHAPE == {i}", *glsl_shape(fit, gain)]
    output.write_text("\n".join([*lines, "#endif", ""]), encoding="utf-8", newline="\n")
    if report:
        report.write_text(json.dumps({"carrier": "Raw D1, half-resolution shape, centered bilinear reconstruction",
                                      "gather_axis_convention": "x right, y down",
                                      "presets": details}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="linear grayscale PNG weights")
    inputs.add_argument("--function", type=Path, help="trusted Python file exporting psf(x, y)")
    inputs.add_argument("--example", action="store_true", help="use the asymmetric soft-strip example")
    inputs.add_argument("--builtins", action="store_true", help="fit both in-game presets and emit GLSL")
    parser.add_argument("--size", type=int, default=81, help="odd square canvas for function input")
    parser.add_argument("--mip", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tap-budget", type=int, default=32, help="maximum number of sparse taps")
    parser.add_argument("--carrier", choices=("centered", "bloom"), default="centered",
                        help="use the centered reference or the in-game Bloom sampling carrier")
    parser.add_argument("--report", type=Path, help="optional JSON report for --builtins")
    parser.add_argument("--glsl", type=Path,
                        help="also export a raw mip-1 Bloom fragment for the custom target")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.tap_budget < 1:
        parser.error("--tap-budget must be positive")
    if args.builtins:
        if args.glsl:
            parser.error("--builtins already emits GLSL to --output")
        if args.output.suffix.lower() != ".glsl":
            parser.error("--builtins requires a .glsl output")
        emit_builtins(args.output, args.report)
        return
    if args.report:
        parser.error("--report requires --builtins")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must have a .json extension")
    if args.carrier == "bloom" and args.mip != 1:
        parser.error("The in-game Bloom carrier requires --mip 1")
    if args.glsl and (args.glsl.suffix.lower() != ".glsl" or args.carrier != "bloom"):
        parser.error("--glsl requires a .glsl path and --carrier bloom")
    target = read_target(args)
    height, width = target.shape
    report = {
        "schema_version": 1,
        "representation": "experimental_sparse_mip_kernel",
        "target_size_xy_px": [width, height],
        "target_center_xy_texel": [width // 2, height // 2],
        "target_unit": "linear scalar mass normalized to one",
        "axis_convention": "x right, y down",
        "target_sha256": hashlib.sha256(target.astype("<f8").tobytes()).hexdigest(),
        "carrier": ({"down": "current BloomTaps D1", "upsample": "centered bilinear"}
                    if args.carrier == "bloom" else
                    {"down_diagonal_offset_input_texels": .75, "down_weight_per_tap": .25,
                     "upsample": "centered bilinear"}),
        **fit(target, args.mip, args.tap_budget, runtime=args.carrier == "bloom"),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.glsl:
        lines = ["// Generated by scripts/fit-bloom-shape.py for the raw mip-1 Bloom branch.",
                 *glsl_shape(report, .375), ""]
        args.glsl.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

"""Bounded residual-footprint experiment on an existing spatial DPF export."""
import copy
from functools import lru_cache
import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix, kron

from bloom_dpf_geometry import ImageDetailLoss, composed_score, encode, quantize
from bloom_dpf_joint import JointDPF, objective_model
from bloom_dpf_model import Filter
from bloom_dpf_policy import arrays, cost, decode_fit, output
from bloom_dpf_target import resize_mass


class FixedPyramidFilter(Filter):
    """Cache the locked analysis/reconstruction operators, preserving their values."""
    @staticmethod
    @lru_cache(maxsize=192)
    def matrices(geometry, destination, offsets, weight_shape, weight_bytes):
        source, down, start, pool, source_start, full_source = geometry
        p = np.frombuffer(offsets, dtype=np.float64).reshape(-1, 2)
        weights = np.frombuffer(weight_bytes, dtype=np.float64).reshape(weight_shape)
        op = Filter(source, destination, p, down=down, start=start, pool=pool,
                    source_start=source_start, full_source=full_source)
        combined = [csr_matrix((math.prod(destination), math.prod(source)), dtype=np.float32)
                    for _ in range(3)]
        for i, row in enumerate(weights):
            tap = kron(op.y[i][0], op.x[i][0], format="csr")
            for c, weight in enumerate(row):
                if weight:
                    combined[c] += tap * weight
        return [(m, m.T.tocsr()) for m in combined]

    def operators(self, weights):
        source, offsets, down, start, pool, source_start, full_source = self.geometry
        return self.matrices((source, down, start, pool, source_start, full_source),
                             self.destination, np.asarray(offsets, np.float64).tobytes(),
                             weights.shape, np.asarray(weights, np.float64).tobytes())

    def apply(self, image, weights):
        batch, channels = image.shape[:2]
        result = np.empty((batch, 3, *self.destination), np.float32)
        for c, (matrix, _) in enumerate(self.operators(weights)):
            result[:, c] = (matrix @ image[:, c % channels].reshape(batch, -1).T).T.reshape(
                batch, *self.destination)
        return result

    def backward_positions(self, image, derivative, weights, *, input_gradient=True):
        source, offsets, *_ = self.geometry
        dx = np.empty((len(image), 3, *source), np.float32) if input_gradient else None
        if input_gradient:
            for c, (_, transpose) in enumerate(self.operators(weights)):
                dx[:, c] = (transpose @ derivative[:, c].reshape(len(image), -1).T).T.reshape(
                    len(image), *source)
        # All pyramid positions and weights are locked by refit's parameter bounds.
        return dx, np.zeros_like(weights), np.zeros_like(offsets)


def sample_matrix(xy, shape):
    x, y = xy.T
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-ix, y-iy
    rows, columns, values = [], [], []
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        valid = (ix+dx >= 0) & (iy+dy >= 0) & (ix+dx < shape[1]) & (iy+dy < shape[0])
        rows.extend(np.flatnonzero(valid))
        columns.extend(((iy[valid]+dy)*shape[1]+ix[valid]+dx))
        values.extend(weight[valid])
    return csr_matrix((values, (rows, columns)), shape=(len(xy), math.prod(shape)))


class LocalContrastLoss(ImageDetailLoss):
    """Target-derived transverse peaks/valleys and longitudinal continuity probes."""
    def __init__(self, target, count=384):
        super().__init__(target)
        luma = np.einsum("c,cyx->yx", [.2126, .7152, .0722], target)
        xx = gaussian_filter(luma, .85, order=(0, 2))
        yy = gaussian_filter(luma, .85, order=(2, 0))
        xy = gaussian_filter(luma, .85, order=(1, 1))
        angle = .5*np.arctan2(2*xy, xx-yy) + math.pi/2
        across = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        y, x = np.indices(luma.shape)
        center = np.stack((x, y), axis=-1)
        neighbors = [map_coordinates(luma, (y+sign*2*across[..., 1], x+sign*2*across[..., 0]),
                                     order=1, prefilter=False, mode="constant") for sign in (-1, 1)]
        contrast = np.maximum(luma-np.maximum(*neighbors), 0.)
        score = contrast/np.maximum(luma, luma.max()*1e-4)
        score[(contrast < luma.max()*2e-5) | (x < 5) | (y < 5) |
              (x >= luma.shape[1]-5) | (y >= luma.shape[0]-5)] = 0
        selected = []
        while len(selected) < count:
            index = np.unravel_index(np.argmax(score), score.shape)
            if score[index] <= 0:
                break
            selected.append(index)
            cy, cx = index
            score[max(0, cy-2):cy+3, max(0, cx-2):cx+3] = 0
        selected = np.array(selected, int).reshape(-1, 2)
        directions = across[selected[:, 0], selected[:, 1]]
        tangent = np.column_stack((-directions[:, 1], directions[:, 0]))
        origins = selected[:, ::-1].astype(float)
        # center, left/right valley, and two overlapping points along the stroke.
        points = np.stack((origins, origins-2*directions, origins+2*directions,
                           origins-2*tangent, origins+2*tangent), axis=1)
        self.points = points
        self.operator = sample_matrix(points.reshape(-1, 2), luma.shape)
        self.samples = (self.operator @ target.reshape(3, -1).T).reshape(-1, 5, 3)
        scale = np.maximum(self.samples[:, :1], target.max(axis=(1, 2))[None, None]*1e-4)
        self.local_weight = .5/np.maximum(scale, 1e-12)**2/max(1, len(origins))/3

    def local(self, fine, gradient=None):
        sampled = (self.operator @ fine.reshape(3, -1).T).reshape(-1, 5, 3)
        error = sampled-self.samples
        factor = np.ones_like(error)
        factor[:, 0] += 2*(error[:, 0] < 0)  # Missing peak, separately from excess valleys.
        factor[:, 1:3] += 2*(error[:, 1:3] > 0)
        value = .5*np.sum(error**2*factor*self.local_weight)
        derivative = error*factor*self.local_weight
        continuity = error[:, 4]-error[:, 3]
        value += .25*np.sum(continuity**2*self.local_weight[:, 0])
        derivative[:, 4] += .5*continuity*self.local_weight[:, 0]
        derivative[:, 3] -= .5*continuity*self.local_weight[:, 0]
        if gradient is not None:
            gradient += (self.operator.T @ derivative.reshape(-1, 3)).T.reshape(fine.shape)
        return float(value)

    def apply(self, image, regions, delta=None):
        value = super().apply(image, regions, delta)
        for region in regions:
            gradient = np.zeros_like(image[region]) if delta is not None else None
            value += self.local(image[region], gradient)/len(regions)
            if delta is not None:
                delta[region] += gradient/len(regions)
        return value


def accounting(fit):
    """Tap proxy, including the deepest constant term; carrier writes are separate."""
    components = {}
    for name, first in (("downsample", 1), ("upsample", 0), ("injection", 1)):
        components[name] = sum(len(s["offsets"])*4.**-(i+first) for i, s in enumerate(fit[name]))
    for name in ("short", "long"):
        components[name] = sum(len(c[name]["offsets"])*4.**-c.get("level", 1)
                               for c in fit.get("features", []))
    components["reads"] = sum(components.values())
    components["carrier_write_pixels"] = sum(4.**-c.get("level", 1) for c in fit.get("features", []))
    components["carrier_write_bytes_rgba16f"] = 8*components["carrier_write_pixels"]
    components["carrier_levels"] = [c.get("level", 1) for c in fit.get("features", [])]
    return components


def reduced_seed(original, ratio=.7):
    fit = copy.deepcopy(original)
    stages = [fit["injection"][0], *[c["long"] for c in fit.get("features", [])]]
    for stage in stages:
        p, w = arrays(stage)
        p, w = quantize(p, w, max(1, math.ceil(len(p)*ratio)))
        stage.update(output(p, w))
    return fit


def existing_proposals(seed):
    proposals = [copy.deepcopy(seed)]
    for pitch in (2., 3., 4.):
        candidate = copy.deepcopy(seed)
        for branch in candidate["features"]:
            angle = math.radians(branch["angle"])
            direction = np.array([math.cos(angle), math.sin(angle)])
            p, w = arrays(branch["short"])
            along = np.linspace(-pitch+1, pitch-1, len(p))
            branch["short"] = output(along[:, None]*direction, w)
        proposals.append(candidate)
    return proposals


def predicted(model, fit):
    p, w, ip, iw, *extra = decode_fit(fit, model.mass)
    return model.predict(p, w, (ip, iw), features=extra[0] if extra else ())


def score(models, fit, local=False):
    if not local:
        return composed_score(models, fit)[0]
    values = []
    for m in models:
        image, outer = predicted(m, fit)
        local_errors = [m.fine_aux.local(image[r]) for r in m.regions]
        regional = max(sum(row[k] for k in ("relative_l2", "shape_scale_relative_l2",
                                            "core_relative_l2", "pooled_outer_relative_l2"))
                       for row in m.metrics(image, outer))
        values.append(max(local_errors)+.15*regional)
    return max(values)


def matching_anchors(wanted, short, short_weights, count):
    """Reuse a local shape over the image with overlapping translated placements."""
    half_radius = math.ceil((wanted.shape[-1]//2)/2)
    remaining = resize_mass(np.maximum(wanted, 0.), half_radius, .5).astype(float)
    radius = math.ceil(abs(short).max())+3
    axis = np.arange(-radius, radius+1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.zeros((3, len(axis), len(axis)))
    for p, w in zip(short, short_weights, strict=True):
        # Sampling offsets have the opposite sign to impulse-response displacement.
        footprint = np.maximum(1-abs(xx+p[0]), 0)*np.maximum(1-abs(yy+p[1]), 0)
        kernel += w[:, None, None]*footprint
    kernel = gaussian_filter(kernel, (0, .65, .65))
    norm = np.maximum(np.sum(kernel**2, axis=(1, 2)), 1e-30)
    points, weights = [], []
    h, w = remaining.shape[-2:]
    for _ in range(count):
        correlation = np.stack([fftconvolve(c, k[::-1, ::-1], mode="same")
                                for c, k in zip(remaining, kernel, strict=True)])
        strength = np.sum(np.maximum(correlation, 0)**2/norm[:, None, None], axis=0)
        cy, cx = np.unravel_index(np.argmax(strength), strength.shape)
        weight = np.maximum(correlation[:, cy, cx], 0)/norm
        points.append([half_radius-cx, half_radius-cy])
        weights.append(weight)
        y0, y1, x0, x1 = max(0, cy-radius), min(h, cy+radius+1), max(0, cx-radius), min(w, cx+radius+1)
        remaining[:, y0:y1, x0:x1] -= weight[:, None, None]*kernel[
            :, y0-cy+radius:y1-cy+radius, x0-cx+radius:x1-cx+radius]
    weights = np.maximum(weights, 1e-16)
    return np.array(points, float), weights


def residual_proposals(seed, model, limit=4):
    image, _ = predicted(model, seed)
    fine = np.mean([image[r] for r in model.regions], axis=0)
    missing = np.maximum(model.target-fine, 0.)
    strength = (missing/np.maximum(model.target, model.target.max()*1e-4)).max(axis=0)
    strength *= np.sqrt(missing.max(axis=0))
    strength[:5] = strength[-5:] = 0
    strength[:, :5] = strength[:, -5:] = 0
    trace, proposals = [], []
    combined = copy.deepcopy(seed)
    center = model.radius
    for number in range(limit):
        cy, cx = np.unravel_index(np.argmax(strength), strength.shape)
        if strength[cy, cx] <= 0:
            break
        strength[max(0, cy-6):cy+7, max(0, cx-6):cx+7] = 0
        index = number % len(seed["features"])
        original = seed["features"][index]
        patch = missing[:, cy-4:cy+5, cx-4:cx+5]
        y, x = np.indices(patch.shape[-2:])
        p = -np.column_stack((x.ravel()-4, y.ravel()-4))/2
        w = patch.reshape(3, -1).T
        keep = w.max(axis=1) > w.max()*1e-4
        p, w = quantize(p[keep], w[keep], len(original["short"]["offsets"]))
        # One RGB-shared center preserves chromatic displacement within the patch.
        origin = np.average(p, axis=0, weights=w.sum(axis=1))
        p -= origin
        w /= np.maximum(w.sum(axis=0), 1e-30)
        # Keep identical tap counts even if a degenerate patch has fewer active cells.
        while len(p) < len(original["short"]["offsets"]):
            i = np.argmax(w.sum(axis=1))
            w[i] *= .5
            p, w = np.vstack((p, p[i])), np.vstack((w, w[i]))
        without = copy.deepcopy(seed)
        without["features"][index]["long"]["rgb_weights"] = np.zeros_like(
            original["long"]["rgb_weights"]).tolist()
        base, _ = predicted(model, without)
        wanted = model.target-np.mean([base[r] for r in model.regions], axis=0)
        lp, lw = matching_anchors(wanted, p, w, len(original["long"]["offsets"]))
        mass = np.asarray(original["long"]["rgb_weights"]).sum(axis=0)
        lw *= mass/lw.sum(axis=0)
        candidate = copy.deepcopy(seed)
        candidate["features"][index].update(short=output(p, w), long=output(lp, lw))
        proposals.append(candidate)
        combined["features"][index] = copy.deepcopy(candidate["features"][index])
        trace.append(dict(patch_center=[int(cx-center), int(cy-center)], branch=index,
                          shared_short_center=origin.tolist(), short_taps=len(p), anchors=len(lp)))
    if len(proposals) > 1:
        proposals.append(combined)
    return proposals, trace


def refit(seed, models, monitors, evaluations=48, progress=None):
    state = decode_fit(seed, models[0].mass)
    x0, bounds, decode, objective = objective_model(models, state, position_span=1.2)
    # The first 4*depth parameter groups contain only the locked pyramid.
    from bloom_dpf_joint import group_params
    p, w = group_params(state[0], state[1], models[0].depth)
    locked = sum(a.size for a in (*p, *w))
    bounds[:locked] = [(v, v) for v in x0[:locked]]
    history, checks = [], []
    best = [score(monitors, seed, True), x0.copy(), 0]
    last = [x0.copy()]

    class BudgetComplete(Exception):
        pass

    def evaluate(point):
        if len(history) == evaluations:
            raise BudgetComplete()
        value, gradient = objective(point)
        history.append(float(value))
        last[0] = point.copy()
        if progress:
            progress(len(history), float(value))
        if len(history) % 8 == 0:
            candidate = encode(seed, decode(point)[0], models[0].mass)
            value = score(monitors, candidate, True)
            checks.append([len(history), value])
            if value < best[0]:
                best[:] = value, point.copy(), len(history)
        return history[-1], gradient

    try:
        result = minimize(evaluate, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=evaluations, maxls=10, ftol=0, gtol=0))
        status = str(result.message)
    except BudgetComplete:
        status = "fixed objective evaluation budget reached"
    final_score = score(monitors, encode(seed, decode(last[0])[0], models[0].mass), True)
    if final_score < best[0]:
        best[:] = final_score, last[0], len(history)
    fit = encode(seed, decode(best[1])[0], models[0].mass)
    fit.update(residual_refit=dict(evaluations=len(history), limit=evaluations, selected_call=best[2],
                                  monitor_score=best[0], monitor_checks=checks, loss_trace=history, status=status),
               weighted_bilinear_fetches_per_full_res_pixel=cost(decode_fit(fit, models[0].mass), models[0].depth))
    return clear_seed_metadata(fit)


def clear_seed_metadata(fit):
    for field in ("geometric_counts", "geometry_selection", "level_count_selection", "holdout", "holdout_phases",
                  "trace", "loss", "iterations", "iteration_limit", "converged", "selected_iteration",
                  "solver", "simulation_side", "outer_pool"):
        fit.pop(field, None)
    fit['solver'] = 'Residual footprints, fixed pyramid, bounded continuous refit'
    return fit

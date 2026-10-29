#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "matplotlib>=3.9"]
# ///
"""Frozen light-tree experiment; output timings are not GPU performance evidence.

Run: uv run scripts/light_sampling_experiment.py --output builds/light-sampling/run-01
A: baseline. B: angular bins with empirical blocker-distance CDFs.
C: degree-2 SH of two bounded-distance moments fitted from the same probes.
Binned moments and query-matched origins diagnose compression versus spatial pooling.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VARIANTS = ("baseline", "distance_cdf", "binned_moments", "sh_moments")
SCENES = ("open", "partition", "doorway")
ALPHA = 0.1
DISTANCE_SCALE = 16.0


@dataclass
class Scene:
    name: str
    lights: np.ndarray
    flux: np.ndarray
    boxes: np.ndarray
    queries: np.ndarray
    query_side: int
    bright: int | None


@dataclass
class Tree:
    center: np.ndarray
    radius: np.ndarray
    flux: np.ndarray
    left: np.ndarray
    right: np.ndarray
    light: np.ndarray
    leaves: np.ndarray


def lower_priority() -> None:
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x40):
            raise ctypes.WinError()


def first_hit(origins: np.ndarray, directions: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Unit-length rays; infinity means no hit in the complete synthetic scene."""
    origins, directions = np.broadcast_arrays(origins, directions)
    result = np.full(origins.shape[:-1], np.inf)
    parallel = directions == 0
    denominator = np.where(parallel, 1.0, directions)
    for low, high in boxes:
        near = (low - origins) / denominator
        far = (high - origins) / denominator
        enter = np.max(np.where(parallel, -np.inf, np.minimum(near, far)), axis=-1)
        leave = np.min(np.where(parallel, np.inf, np.maximum(near, far)), axis=-1)
        outside = np.any(parallel & ((origins < low) | (origins > high)), axis=-1)
        hit = ~outside & (leave >= np.maximum(enter, 1e-7))
        distance = np.where(enter > 1e-7, enter, leave)
        result = np.minimum(result, np.where(hit, distance, np.inf))
    return result


def scalar_segment_visible(origin: np.ndarray, light: np.ndarray, boxes: np.ndarray) -> bool:
    """Independent reference: enumerate intersections with box faces, not ray slabs."""
    delta = light - origin
    for low, high in boxes:
        for axis in range(3):
            if delta[axis] == 0:
                continue
            for plane in (low[axis], high[axis]):
                t = (plane - origin[axis]) / delta[axis]
                if 1e-7 < t < 1 - 1e-7:
                    point = origin + t * delta
                    if np.all(point >= low - 1e-10) and np.all(point <= high + 1e-10):
                        return False
    return True


def make_scene(name: str, count: int, side: int) -> Scene:
    rng = np.random.default_rng(20261029 + SCENES.index(name))
    lights = rng.uniform([-24, 2, -24], [40, 24, 40], (count, 3))
    flux = np.exp(rng.uniform(np.log(0.5), np.log(8.0), count))
    boxes = [[[-128, -1, -128], [128, 0, 128]]]
    x = np.linspace(0.5, 15.5, side)
    z = np.linspace(0.5, 15.5, side)
    bright = None
    if name == "partition":
        lights = rng.uniform([0.5, 2, 0.5], [15.5, 10, 15.5], (count, 3))
        lights[:, 0] = np.where(lights[:, 0] < 8, np.minimum(lights[:, 0], 7.5),
                               np.maximum(lights[:, 0], 8.5))
        boxes.extend([
            [[7.9, 0, -0.3], [8.1, 12, 16.3]],
            [[-0.3, 0, -0.3], [0, 12, 16.3]],
            [[16, 0, -0.3], [16.3, 12, 16.3]],
            [[0, 0, -0.3], [16, 12, 0]],
            [[0, 0, 16], [16, 12, 16.3]],
            [[0, 12, 0], [16, 12.3, 16]],
        ])
    elif name == "doorway":
        lights = rng.uniform([-16, 0.5, 16], [32, 12, 72], (count, 3))
        front = max(8, count // 16)
        lights[:front] = rng.uniform([1, 2, 1], [15, 8, 10], (front, 3))
        flux[:front] = 1.0
        bright = count - 1
        lights[bright] = [8, 3, 60]
        flux[bright] = 200_000.0
        boxes.extend([
            [[-64, 0, 12], [7.5, 20, 12.5]],
            [[8.5, 0, 12], [64, 20, 12.5]],
            [[7.5, 3, 12], [8.5, 20, 12.5]],
        ])
        z = np.linspace(2.1, 5.9, side)
    xx, zz = np.meshgrid(x, z)
    queries = np.column_stack([xx.ravel(), np.full(side * side, 0.02), zz.ravel()])
    return Scene(name, lights, flux, np.asarray(boxes, float), queries, side, bright)


def build_tree(lights: np.ndarray, flux: np.ndarray) -> Tree:
    extent = np.ptp(lights, axis=0)
    coords = ((lights - lights.min(axis=0)) / np.maximum(extent, 1e-9) * 1023).astype(np.uint64)
    codes = np.zeros(len(lights), dtype=np.uint64)
    for bit in range(10):
        for axis in range(3):
            codes |= ((coords[:, axis] >> bit) & 1) << (3 * bit + axis)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    centers, radii, powers, lefts, rights, ids = [], [], [], [], [], []
    leaves = np.empty(len(lights), dtype=int)

    def append(start: int, end: int) -> int:
        node = len(ids)
        subset = order[start:end]
        power = flux[subset].sum()
        center = np.sum(lights[subset] * flux[subset, None], axis=0) / power
        centers.append(center)
        radii.append(np.linalg.norm(lights[subset] - center, axis=1).max())
        powers.append(power)
        lefts.append(-1)
        rights.append(-1)
        ids.append(-1)
        if end - start == 1:
            ids[node] = int(subset[0])
            leaves[subset[0]] = node
        else:
            differing_bit = (int(sorted_codes[start]) ^ int(sorted_codes[end - 1])).bit_length() - 1
            if differing_bit < 0:
                split = (start + end) // 2
            else:
                boundary = (int(sorted_codes[start]) >> differing_bit | 1) << differing_bit
                split = int(np.searchsorted(sorted_codes, boundary, side="left"))
            assert start < split < end
            lefts[node] = append(start, split)
            rights[node] = append(split, end)
        return node

    append(0, len(order))
    return Tree(np.asarray(centers), np.asarray(radii), np.asarray(powers),
                np.asarray(lefts), np.asarray(rights), np.asarray(ids), leaves)


def sphere_directions(resolution: int) -> np.ndarray:
    y = -1 + 2 * (np.arange(resolution) + 0.5) / resolution
    phi = 2 * np.pi * (np.arange(2 * resolution) + 0.5) / (2 * resolution)
    yy, pp = np.meshgrid(y, phi, indexing="ij")
    radius = np.sqrt(1 - yy * yy)
    return np.column_stack([(radius * np.cos(pp)).ravel(), yy.ravel(),
                            (radius * np.sin(pp)).ravel()])


def sh_basis(direction: np.ndarray) -> np.ndarray:
    x, y, z = direction.T
    return np.column_stack([
        np.full(len(x), 0.28209479177387814),
        0.4886025119029199 * x, 0.4886025119029199 * y, 0.4886025119029199 * z,
        1.0925484305920792 * x * y, 1.0925484305920792 * y * z,
        1.0925484305920792 * x * z, 0.31539156525252005 * (3 * y * y - 1),
        0.5462742152960396 * (x * x - z * z),
    ])


def train_distances(scene: Scene, seed: int, origins_count: int, directions: np.ndarray, query_matched: bool):
    rng = np.random.default_rng(seed)
    low, high = scene.queries.min(axis=0), scene.queries.max(axis=0)
    origins = scene.queries.copy() if query_matched else rng.uniform(low, high, (origins_count, 3))
    if scene.name == "partition" and not query_matched:
        inside_wall = (origins[:, 0] > 7.9) & (origins[:, 0] < 8.1)
        while inside_wall.any():
            origins[inside_wall, 0] = rng.uniform(low[0], high[0], inside_wall.sum())
            inside_wall = (origins[:, 0] > 7.9) & (origins[:, 0] < 8.1)
    distances = np.empty((len(directions), len(origins)))
    for first in range(0, len(directions), 512):
        distance = first_hit(origins[:, None, :], directions[None, first:first + 512, :], scene.boxes)
        distances[first:first + 512] = distance.T
    transformed = 1 - DISTANCE_SCALE / (distances + DISTANCE_SCALE)
    if query_matched:
        moments = np.stack([transformed, transformed ** 2], axis=-1)
        coefficients = np.linalg.lstsq(sh_basis(directions), moments.reshape(len(directions), -1), rcond=None)[0]
        coefficients = coefficients.reshape(9, len(origins), 2).transpose(1, 0, 2)
    else:
        moments = np.column_stack([transformed.mean(axis=1), (transformed ** 2).mean(axis=1)])
        coefficients = np.linalg.lstsq(sh_basis(directions), moments, rcond=None)[0]
    return origins, distances, coefficients


class Visibility:
    def __init__(self, mode: str, distances: np.ndarray, coefficients: np.ndarray, resolution: int,
                 query_matched: bool):
        self.mode = mode
        self.distances = distances
        self.coefficients = coefficients
        self.resolution = resolution
        self.query_matched = query_matched
        if mode == "binned_moments":
            transformed = 1 - DISTANCE_SCALE / (distances + DISTANCE_SCALE)
            self.bin_moments = (np.stack([transformed, transformed ** 2], axis=-1) if query_matched
                                else np.column_stack([transformed.mean(axis=1), (transformed ** 2).mean(axis=1)]))
        self.evaluations = 0
        self.repairs = 0

    def evaluate(self, directions: np.ndarray, distances: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
        self.evaluations += len(directions)
        if self.mode in ("distance_cdf", "binned_moments"):
            rows = np.clip(((directions[:, 1] + 1) * 0.5 * self.resolution).astype(int),
                           0, self.resolution - 1)
            phi = np.mod(np.arctan2(directions[:, 2], directions[:, 0]), 2 * np.pi)
            cols = np.minimum((phi / (2 * np.pi) * 2 * self.resolution).astype(int),
                              2 * self.resolution - 1)
            bins = rows * (2 * self.resolution) + cols
            if self.mode == "distance_cdf":
                if self.query_matched:
                    return (self.distances[bins, query_indices] >= distances).astype(float)
                return np.mean(self.distances[bins] >= distances[:, None], axis=1)
            raw = self.bin_moments[bins, query_indices] if self.query_matched else self.bin_moments[bins]
        elif self.query_matched:
            raw = np.einsum("ij,ijk->ik", sh_basis(directions), self.coefficients[query_indices])
        else:
            raw = sh_basis(directions) @ self.coefficients
        mean = np.clip(raw[:, 0], 0, 1)
        second = np.clip(raw[:, 1], mean * mean, mean)
        self.repairs += int(np.count_nonzero((np.abs(mean - raw[:, 0]) > 1e-12)
                                          | (np.abs(second - raw[:, 1]) > 1e-12)))
        variance = second - mean * mean
        delta = distances / (distances + DISTANCE_SCALE) - mean
        result = np.ones(len(mean))
        beyond = delta > 0
        result[beyond] = variance[beyond] / (variance[beyond] + delta[beyond] ** 2)
        return result

    def node_factor(self, tree: Tree, node: int, queries: np.ndarray) -> np.ndarray:
        delta = tree.center[node] - queries
        distance = np.linalg.norm(delta, axis=1)
        ratio = tree.radius[node] / distance
        compact = ratio <= 0.5
        result = np.ones(len(queries))
        if not compact.any():
            return result
        direction = delta[compact] / distance[compact, None]
        near_distance = distance[compact] - tree.radius[node]
        query_indices = np.flatnonzero(compact)
        if tree.radius[node] < 1e-12:
            estimate = self.evaluate(direction, near_distance, query_indices)
        else:
            up = np.zeros_like(direction)
            up[:, 1] = 1
            up[np.abs(direction[:, 1]) > 0.95] = [1, 0, 0]
            tangent = np.cross(direction, up)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            bitangent = np.cross(direction, tangent)
            sine = ratio[compact, None]
            cosine = np.sqrt(1 - sine * sine)
            samples = np.stack([
                direction,
                direction * cosine + tangent * sine,
                direction * cosine - tangent * sine,
                direction * cosine + bitangent * sine,
                direction * cosine - bitangent * sine,
            ], axis=1)
            estimate = self.evaluate(samples.reshape(-1, 3), np.repeat(near_distance, 5),
                                     np.repeat(query_indices, 5)).reshape(-1, 5).mean(axis=1)
        strength = (1 - ratio[compact]) ** 2
        result[compact] = 1 - strength + strength * estimate
        return result


def branch_probabilities(tree: Tree, queries: np.ndarray, visibility: Visibility | None) -> np.ndarray:
    delta = tree.center[None, :, :] - queries[:, None, :]
    squared = np.sum(delta * delta, axis=-1)
    score = tree.flux[None, :] * np.maximum(delta[:, :, 1], 0) / np.sqrt(squared) / (squared + 0.25 ** 2)
    base = score.copy()
    if visibility is not None:
        for node in range(1, len(tree.light)):
            score[:, node] *= visibility.node_factor(tree, node, queries)
    probabilities = np.zeros_like(score)
    for node, left in enumerate(tree.left):
        if left < 0:
            continue
        right = tree.right[node]
        baseline = base[:, left] / (base[:, left] + base[:, right])
        denominator = score[:, left] + score[:, right]
        probabilities[:, node] = np.divide(score[:, left], denominator, out=baseline.copy(),
                                          where=denominator > 0)
    return probabilities


def leaf_probabilities(tree: Tree, branch: np.ndarray) -> np.ndarray:
    mass = np.zeros_like(branch)
    mass[:, 0] = 1
    for node, left in enumerate(tree.left):
        if left >= 0:
            mass[:, left] = mass[:, node] * branch[:, node]
            mass[:, tree.right[node]] = mass[:, node] * (1 - branch[:, node])
    return mass[:, tree.leaves]


def sample_tree(tree: Tree, base: np.ndarray, guided: np.ndarray, count: int, seed: int):
    rng = np.random.default_rng(seed)
    use_base = rng.random(count) < ALPHA
    nodes = np.zeros(count, dtype=int)
    base_pdf = np.ones(count)
    guided_pdf = np.ones(count)
    while True:
        active = np.flatnonzero(tree.light[nodes] < 0)
        if not len(active):
            break
        current = nodes[active]
        pb, pg = base[current], guided[current]
        left = rng.random(len(active)) < np.where(use_base[active], pb, pg)
        base_pdf[active] *= np.where(left, pb, 1 - pb)
        guided_pdf[active] *= np.where(left, pg, 1 - pg)
        nodes[active] = np.where(left, tree.left[current], tree.right[current])
    return tree.light[nodes], ALPHA * base_pdf + (1 - ALPHA) * guided_pdf


def contributions(scene: Scene):
    delta = scene.lights[None, :, :] - scene.queries[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    direction = delta / distance[:, :, None]
    visible = first_hit(scene.queries[:, None, :], direction, scene.boxes) >= distance - 1e-7
    value = scene.flux[None, :] * np.maximum(direction[:, :, 1], 0) / (4 * np.pi ** 2 * distance ** 2)
    return value * visible, visible


def moments(contribution: np.ndarray, probability: np.ndarray):
    total = contribution.sum(axis=1)
    variance = np.sum(contribution ** 2 / probability, axis=1) - total ** 2
    assert np.all(variance >= -1e-10 * total ** 2)
    return total, np.maximum(variance, 0)


def weighted_quantile(values: np.ndarray, probabilities: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    index = np.searchsorted(np.cumsum(probabilities[order]), quantile)
    return float(values[order[min(index, len(order) - 1)]])


def validate_primitives() -> dict:
    box = np.array([[[1, -1, -1], [2, 1, 1]]], float)
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], float)
    hit = first_hit(np.zeros(3), directions, box)
    np.testing.assert_array_equal(hit, [1, np.inf, np.inf])
    assert not scalar_segment_visible(np.zeros(3), np.array([3, 0, 0]), box)
    assert scalar_segment_visible(np.zeros(3), np.array([-3, 0, 0]), box)
    q0, qv, value = np.array([0.9, 0.1]), np.array([0.0, 1.0]), np.array([10.0, 1.0])
    mixture = ALPHA * q0 + (1 - ALPHA) * qv
    corrected = float(np.sum(mixture * (value / mixture)))
    wrong_conditional = float(ALPHA * value.sum() + (1 - ALPHA) * value[1])
    assert corrected == 11.0 and wrong_conditional == 2.0
    return {"ray_box_cases": 3, "mixture_expected": corrected,
            "unsupported_conditional_pdf_mutation": wrong_conditional}


def mc_checks(scene: Scene, tree: Tree, branches: dict, probabilities: dict,
              value: np.ndarray, sample_count: int) -> list[dict]:
    truth = value.sum(axis=1)
    _, base_variance = moments(value, probabilities["baseline"])
    _, sh_variance = moments(value, probabilities["sh_moments"])
    chosen = {len(truth) // 2, int(np.argmax(truth)),
              int(np.argmax(sh_variance / np.maximum(base_variance, 1e-30)))}
    if scene.bright is not None:
        visible = np.flatnonzero(value[:, scene.bright] > 0)
        chosen.add(int(visible[len(visible) // 2]))
    rows = []
    for mode in VARIANTS:
        _, variance = moments(value, probabilities[mode])
        for query in sorted(chosen):
            for seed in (101, 503, 997, 2027):
                ids, pdf = sample_tree(tree, branches["baseline"][query], branches[mode][query],
                                      sample_count, seed)
                np.testing.assert_allclose(pdf, probabilities[mode][query, ids], rtol=2e-12, atol=1e-16)
                estimate = float(np.mean(value[query, ids] / pdf))
                se = float(np.sqrt(variance[query] / sample_count))
                z = (estimate - truth[query]) / se if se > 0 else 0.0
                if abs(z) > 7:
                    raise AssertionError(f"MC mean inconsistent: {scene.name} {mode} {query} z={z}")
                rows.append({"scene": scene.name, "mode": mode, "query": query, "seed": seed,
                             "samples": sample_count, "mean": estimate, "reference": float(truth[query]),
                             "analytic_standard_error": se, "z": float(z)})
    return rows


def plot_results(output: Path, plots: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(len(plots), len(VARIANTS), figsize=(4 * len(VARIANTS), 3.7 * len(plots)), squeeze=False,
                             constrained_layout=True)
    for row, (name, data) in enumerate(plots.items()):
        scene, truth, variances = data
        extent = [scene.queries[:, 0].min(), scene.queries[:, 0].max(),
                  scene.queries[:, 2].min(), scene.queries[:, 2].max()]
        reference = axes[row, 0].imshow(truth.reshape(scene.query_side, scene.query_side),
                                       origin="lower", extent=extent, aspect="auto",
                                       norm=LogNorm(), cmap="viridis")
        axes[row, 0].set_title(f"{name}: exact direct lighting")
        fig.colorbar(reference, ax=axes[row, 0], shrink=0.8)
        for col, mode in enumerate(VARIANTS[1:], 1):
            ratio = variances[mode] / np.maximum(variances["baseline"], 1e-30)
            image = axes[row, col].imshow(np.log10(np.maximum(ratio, 1e-5)).reshape(scene.query_side, scene.query_side),
                                         origin="lower", extent=extent, aspect="auto",
                                         cmap="RdBu_r", vmin=-1, vmax=1)
            axes[row, col].set_title(f"{name}: {mode} variance / baseline")
            fig.colorbar(image, ax=axes[row, col], shrink=0.8, label="log10 ratio (clipped at +/-1)")
        for ax in axes[row]:
            ax.set_xlabel("receiver x (blocks)")
            ax.set_ylabel("receiver z (blocks)")
    fig.suptitle("Frozen visibility guidance: exact single-sample variance\n"
                 "Blue = less variance, red = more variance; first training seed; no GPU cost claim")
    fig.savefig(output / "variance-maps.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lights", type=int, default=512)
    parser.add_argument("--query-side", type=int, default=16)
    parser.add_argument("--origins", type=int, default=64)
    parser.add_argument("--origin-mode", choices=("pooled", "query-matched"), default="pooled")
    parser.add_argument("--angular-resolution", type=int, default=64)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--mc-samples", type=int, default=65536)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    if args.lights < 32 or args.query_side < 4 or args.origins < 2 or args.angular_resolution < 4:
        parser.error("Require lights >= 32, query-side >= 4, origins >= 2, angular-resolution >= 4")
    lower_priority()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    shutil.copy2(source, output / source.name)
    directions = sphere_directions(args.angular_resolution)
    query_matched = args.origin_mode == "query-matched"
    checks = validate_primitives()
    summaries, queries_csv, mc_rows, plots = [], [], [], {}
    started = time.perf_counter()
    for name in args.scenes:
        scene = make_scene(name, args.lights, args.query_side)
        for low, high in scene.boxes:
            assert not np.any(np.all((scene.queries > low) & (scene.queries < high), axis=1)), "Receiver inside an occluder"
        tree = build_tree(scene.lights, scene.flux)
        value, visible = contributions(scene)
        # Face enumeration checks the actual scene rays through a different intersection algorithm.
        rng = np.random.default_rng(17)
        for query, light in zip(rng.integers(len(scene.queries), size=128), rng.integers(args.lights, size=128)):
            assert scalar_segment_visible(scene.queries[query], scene.lights[light], scene.boxes) == visible[query, light]
        if name == "doorway":
            assert np.any(visible[:, scene.bright]) and np.any(~visible[:, scene.bright]), "Doorway fixture must include both visibility outcomes"
            assert scalar_segment_visible(np.array([8, 0.02, 4]), scene.lights[scene.bright], scene.boxes)
            assert not scalar_segment_visible(np.array([2, 0.02, 4]), scene.lights[scene.bright], scene.boxes)
        base_branch = branch_probabilities(tree, scene.queries, None)
        base_q = leaf_probabilities(tree, base_branch)
        truth, base_variance = moments(value, base_q)
        assert np.all(truth > 0)
        for seed_index, seed in enumerate(args.training_seeds):
            origin_count = len(scene.queries) if query_matched else args.origins
            print(f"{name} seed={seed}: {origin_count * len(directions):,} {args.origin_mode} training rays", flush=True)
            origins, blocker_distance, coefficients = train_distances(scene, seed, args.origins, directions, query_matched)
            branches = {"baseline": base_branch}
            probabilities = {"baseline": base_q}
            visibility_stats = {"baseline": {"queries": 0, "moment_repairs": 0}}
            for mode in VARIANTS[1:]:
                field = Visibility(mode, blocker_distance, coefficients, args.angular_resolution, query_matched)
                branch = branch_probabilities(tree, scene.queries, field)
                guided_q = leaf_probabilities(tree, branch)
                branches[mode] = branch
                probabilities[mode] = ALPHA * base_q + (1 - ALPHA) * guided_q
                visibility_stats[mode] = {"queries": field.evaluations, "moment_repairs": field.repairs}
            if query_matched:
                np.testing.assert_allclose(probabilities["binned_moments"], probabilities["distance_cdf"],
                                           rtol=2e-12, atol=1e-16)
            variances = {}
            for mode in VARIANTS:
                q = probabilities[mode]
                np.testing.assert_allclose(q.sum(axis=1), 1.0, rtol=0, atol=2e-12)
                assert np.all(q >= ALPHA * base_q - 1e-16)
                _, variance = moments(value, q)
                variances[mode] = variance
                weights = value / q / truth[:, None]
                ratios = variance / np.maximum(base_variance, 1e-30)
                record = {
                    "scene": name, "training_seed": seed, "mode": mode,
                    "variance_ratio": float(variance.sum() / base_variance.sum()),
                    "relative_variance_ratio": float(np.mean(variance / truth ** 2)
                                                     / np.mean(base_variance / truth ** 2)),
                    "median_query_ratio": float(np.median(ratios)), "worst_query_ratio": float(ratios.max()),
                    "mean_visible_selection_probability": float(np.mean(np.sum(q * visible, axis=1))),
                    "max_contribution_over_truth": float(weights.max()),
                    "median_p999_contribution_over_truth": float(np.median([
                        weighted_quantile(weights[i], q[i], 0.999) for i in range(len(truth))
                    ])),
                    **visibility_stats[mode],
                }
                if scene.bright is not None:
                    bright_visible = value[:, scene.bright] > 0
                    record["bright_visible_queries"] = int(bright_visible.sum())
                    record["bright_visible_mean_pmf_ratio"] = float(np.mean(
                        q[bright_visible, scene.bright] / base_q[bright_visible, scene.bright]))
                    record["bright_visible_mean_contribution_fraction"] = float(np.mean(
                        value[bright_visible, scene.bright] / truth[bright_visible]))
                    record["bright_visible_variance_ratio"] = float(
                        variance[bright_visible].sum() / base_variance[bright_visible].sum())
                summaries.append(record)
                for i, query in enumerate(scene.queries):
                    queries_csv.append({
                        "scene": name, "seed": seed, "mode": mode, "query": i,
                        "x": query[0], "z": query[2], "truth": truth[i], "variance": variance[i],
                        "variance_ratio": ratios[i], "relative_variance": variance[i] / truth[i] ** 2,
                        "max_weight_over_truth": weights[i].max(),
                        "visible_probability": np.sum(q[i] * visible[i]),
                    })
                print(f"  {mode}: variance={record['variance_ratio']:.4f}x, "
                      f"relative={record['relative_variance_ratio']:.4f}x, "
                      f"worst query={record['worst_query_ratio']:.3f}x", flush=True)
            np.savez_compressed(
                output / f"{name}-seed-{seed}.npz", lights=scene.lights, flux=scene.flux, boxes=scene.boxes,
                queries=scene.queries, training_origins=origins, probe_directions=directions,
                blocker_distances=blocker_distance, sh_coefficients=coefficients, contributions=value,
                q_baseline=base_q, q_distance_cdf=probabilities["distance_cdf"],
                q_binned_moments=probabilities["binned_moments"],
                q_sh_moments=probabilities["sh_moments"],
            )
            if seed_index == 0:
                mc_rows.extend(mc_checks(scene, tree, branches, probabilities, value, args.mc_samples))
                plots[name] = (scene, truth, variances)
            (output / "checkpoint.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_results(output, plots)
    protocol = {
        "scope": "CPU statistical experiment only; opaque synthetic AABBs, isotropic point lights, diffuse receivers",
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parent,
                                                   text=True).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "alpha": ALPHA, "distance_scale": DISTANCE_SCALE,
        "angular_sampling": "equal-solid-angle bin midpoints; quadrature, no claims of unbiased angular integration",
        "node_rule": "flux-weighted center; bounding sphere; visibility only if radius/distance <= 0.5; "
                     "five angular samples at nearest sphere distance; strength=(1-radius/distance)^2",
        "geometry_rule": "centroid flux*cos/(distance_squared+0.25_squared); exact inverse square in contributions",
        "precision": "float64 statistics and proposal evaluation; no FP32 or GPU acceptance",
        "wall_seconds_at_idle_priority": time.perf_counter() - started,
        "checks": {**checks, "independent_face_checks_per_scene": 128,
                   "mc_trials": len(mc_rows), "max_abs_mc_z": max(abs(row["z"]) for row in mc_rows)},
        "results": summaries,
    }
    (output / "results.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for filename, rows in (("queries.csv", queries_csv), ("monte-carlo.csv", mc_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"COMPLETE: {output}", flush=True)
    print(json.dumps(protocol["checks"]), flush=True)


if __name__ == "__main__":
    main()

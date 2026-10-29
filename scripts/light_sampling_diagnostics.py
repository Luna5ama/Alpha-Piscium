#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "matplotlib>=3.9"]
# ///
"""Diagnose saved runs without retraining; leaf sums and exact visibility are costly reference controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import light_sampling_experiment as experiment


def probabilities_from_scores(tree, score, baseline):
    branch = np.zeros_like(score)
    for node, left in enumerate(tree.left):
        if left >= 0:
            right = tree.right[node]
            total = score[:, left] + score[:, right]
            branch[:, node] = np.divide(score[:, left], total, out=baseline[:, node].copy(), where=total > 0)
    return experiment.leaf_probabilities(tree, branch)


def refine_scores(tree, score, levels):
    for _ in range(levels):
        finer = score.copy()
        for node, left in enumerate(tree.left):
            if left >= 0:
                finer[:, node] = score[:, left] + score[:, tree.right[node]]
        score = finer
    return score


def mixed_leaf_scores(score, base):
    sums = score.sum(axis=1, keepdims=True)
    guided = np.divide(score, sums, out=base.copy(), where=sums > 0)
    return experiment.ALPHA * base + (1 - experiment.ALPHA) * guided


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment.lower_priority()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path(experiment.__file__)):
        shutil.copy2(source, output / source.name)
    records = []
    sources = []
    for folder in args.inputs:
        protocol = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        sources.append({"path": str(folder.resolve()), "source_sha256": protocol["source_sha256"]})
        query_matched = protocol["arguments"]["origin_mode"] == "query-matched"
        resolution = protocol["arguments"]["angular_resolution"]
        for path in sorted(folder.glob("*-seed-*.npz")):
            name, seed = path.stem.rsplit("-seed-", 1)
            data = np.load(path)
            tree = experiment.build_tree(data["lights"], data["flux"])
            query = data["queries"]
            value = data["contributions"]
            base = data["q_baseline"]
            truth, base_variance = experiment.moments(value, base)
            node_delta = tree.center[None] - query[:, None]
            node_squared = (node_delta ** 2).sum(axis=-1)
            node_score = tree.flux[None] * np.maximum(node_delta[:, :, 1], 0) / np.sqrt(node_squared) / (node_squared + 0.25 ** 2)
            base_branch = np.zeros_like(node_score)
            for node, left in enumerate(tree.left):
                if left >= 0:
                    right = tree.right[node]
                    base_branch[:, node] = node_score[:, left] / (node_score[:, left] + node_score[:, right])
            np.testing.assert_allclose(experiment.leaf_probabilities(tree, base_branch), base, rtol=2e-12, atol=1e-16)
            delta = data["lights"][None] - query[:, None]
            distance = np.linalg.norm(delta, axis=-1)
            direction = delta / distance[:, :, None]
            unoccluded = data["flux"][None] * direction[:, :, 1] / (4 * np.pi ** 2 * distance ** 2)
            candidates = {
                "exact_unoccluded_leaf_sum": mixed_leaf_scores(unoccluded, base),
                "exact_visible_leaf_sum": mixed_leaf_scores(value, base),
            }
            finer = refine_scores(tree, node_score, 2)
            candidates["lookahead2_geometry"] = (experiment.ALPHA * base + (1 - experiment.ALPHA)
                                                  * probabilities_from_scores(tree, finer, base_branch))
            for mode in experiment.VARIANTS[1:]:
                field = experiment.Visibility(mode, data["blocker_distances"], data["sh_coefficients"],
                                              resolution, query_matched)
                scores = node_score.copy()
                for node in range(1, len(tree.light)):
                    scores[:, node] *= field.node_factor(tree, node, query)
                finer = refine_scores(tree, scores, 2)
                guided = probabilities_from_scores(tree, finer, base_branch)
                candidates["lookahead2_" + mode] = experiment.ALPHA * base + (1 - experiment.ALPHA) * guided
                leaf_visibility = field.evaluate(
                    direction.reshape(-1, 3), distance.ravel(),
                    np.repeat(np.arange(len(query)), len(data["lights"])),
                ).reshape(distance.shape)
                candidates["leaf_sum_" + mode] = mixed_leaf_scores(unoccluded * leaf_visibility, base)
            arrays = {}
            for label, probability in candidates.items():
                np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=2e-12)
                assert np.all(probability >= experiment.ALPHA * base - 1e-16)
                _, variance = experiment.moments(value, probability)
                row = {
                    "origin_mode": protocol["arguments"]["origin_mode"], "scene": name,
                    "seed": int(seed), "variant": label,
                    "variance_ratio": float(variance.sum() / base_variance.sum()),
                    "relative_variance_ratio": float(np.mean(variance / truth ** 2) / np.mean(base_variance / truth ** 2)),
                    "worst_query_ratio": float(np.max(variance / np.maximum(base_variance, 1e-30))),
                }
                records.append(row)
                arrays[label] = probability
            np.savez_compressed(output / (protocol["arguments"]["origin_mode"] + "-" + path.name), **arrays)
            print(protocol["arguments"]["origin_mode"], path.stem,
                  {row["variant"]: round(row["variance_ratio"], 4)
                   for row in records if row["origin_mode"] == protocol["arguments"]["origin_mode"]
                   and row["scene"] == name and row["seed"] == int(seed)}, flush=True)
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "diagnostics.json").write_text(json.dumps({
        "inputs": sources, "scope": "Exact statistical controls; leaf aggregation and oracle visibility are not runtime proposals",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment_source_sha256": hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest(),
        "results": records,
    }, indent=2), encoding="utf-8")
    print("COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()

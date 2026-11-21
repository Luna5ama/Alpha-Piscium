"""Test whether render-time features can predict weights for the fixed mirror-room candidates."""

import json
import sys

import numpy as np


LUMA = np.array([0.3109383344, 0.6385030652, 0.0505586004], dtype=np.float32)


def blend(features, candidates, valid, coefficients):
    logits = features @ coefficients
    logits = np.where(valid, logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights[..., None] * candidates).sum(axis=1), weights


def fit(features, candidates, valid, reference):
    rng = np.random.default_rng(431)
    coefficients = np.zeros((features.shape[1], 3), dtype=np.float32)
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    for step in range(1, 1001):
        indices = rng.integers(len(reference), size=1024)
        x = features[indices]
        rgb = candidates[indices]
        predicted, weights = blend(x, rgb, valid[indices], coefficients)
        residual = predicted - reference[indices]
        grad_logits = weights * np.einsum("nc,nkc->nk", residual, rgb - predicted[:, None, :])
        grad = x.T @ grad_logits * (2 / (3 * len(indices)))
        first_moment = 0.9 * first_moment + 0.1 * grad
        second_moment = 0.999 * second_moment + 0.001 * grad**2
        update = first_moment / (1 - 0.9**step)
        scale = np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8
        coefficients -= 0.03 * update / scale
    return coefficients


def load_samples(path):
    with np.load(path, allow_pickle=False) as data:
        candidates = np.stack([data[key] for key in ("raw_rgb", "surface_rgb", "reflection_rgb")], axis=-2)
        geometry = data["geometry"].astype(np.float32)
        reference_a = data["reference_96_rgb"].astype(np.float32)
        reference_b = 2 * data["reference_192_rgb"].astype(np.float32) - reference_a
        denoised = data["denoised_rgb"].astype(np.float32)
        valid = np.stack([np.ones_like(data["valid_surface"]), data["valid_surface"], data["valid_reflection"]], axis=-1)

    mask = (geometry[..., 0] <= 0.1) & (geometry[..., 1] == 0) & (geometry[..., 2] > 0)
    xcoord = np.nonzero(mask)[2]
    candidates = candidates[mask].astype(np.float32)
    valid = valid[mask]
    reference_a = reference_a[mask]
    reference_b = reference_b[mask]
    denoised = denoised[mask]
    roughness = geometry[..., 0][mask]

    log_y = np.log1p(candidates @ LUMA)
    raw, surface, reflection = log_y.T
    features = np.column_stack([
        np.ones_like(raw), raw, surface, reflection, np.abs(raw - surface),
        np.abs(raw - reflection), np.abs(surface - reflection), roughness,
    ])

    return dict(x=xcoord, features=features, candidates=candidates, valid=valid,
                reference_a=reference_a, reference_b=reference_b, denoised=denoised)


def score(training, train, evaluation, test):
    target = evaluation["reference_b"][test]

    def rmse(predicted):
        return float(np.sqrt(np.mean((predicted - target) ** 2, dtype=np.float64)))

    scores = {"pixels": len(target), "denoised": rmse(evaluation["denoised"][test])}
    for model, columns in (("constant", slice(0, 1)), ("render_features", slice(None))):
        coefficients = fit(training["features"][train, columns], training["candidates"][train],
                           training["valid"][train], training["reference_a"][train])
        predicted, _ = blend(evaluation["features"][test, columns], evaluation["candidates"][test],
                             evaluation["valid"][test], coefficients)
        scores[model] = rmse(predicted)
    return scores


def evaluate(path, transfer_path=None):
    mirror = load_samples(path)
    left = mirror["x"] < 128
    right = mirror["x"] >= 192
    results = {"test_left": score(mirror, right, mirror, left), "test_right": score(mirror, left, mirror, right)}
    if transfer_path is not None:
        other = load_samples(transfer_path)
        results["transfer"] = score(mirror, slice(None), other, slice(None))
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(*sys.argv[1:]), indent=2))

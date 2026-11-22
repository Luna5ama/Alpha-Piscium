"""Score native reprojected fresh-sample evidence against the two independent GI fixtures."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from gi_specular_writeback_probe import LUMA, captures, half_matches, hdr_tile, measure


def score(fixture, frames, horizon, variant, bundle):
    metadata = json.loads(fixture["metadata"].item())
    pose_ids = metadata.get("pose_ids", metadata.get("frames"))
    geometry = fixture["geometry"].astype(np.float32)
    half_a = fixture["reference_96_rgb"].astype(np.float32) @ LUMA
    half_b = 2 * (fixture["reference_192_rgb"].astype(np.float32) @ LUMA) - half_a
    counts = {k: np.zeros((2, 5), np.int64) for k in ("moving", "held")}
    phases = []
    reset = variant in ("reset", "projection")
    turns = []
    for i, frame in enumerate(frames):
        field = hdr_tile(frame, bundle)
        raw, state, meta = field(0), field(2), field(8)
        age, flags = raw[..., 1], raw[..., 2].astype(np.uint8)
        mean, next_age, prior_mean = np.moveaxis(state, -1, 0)
        weights = meta[..., 2]
        active = next_age > 0
        invalid = active & (flags != 3)
        assert np.isfinite(state).all() and np.isfinite(raw).all()
        assert np.all((weights == 0) | (weights == 0.5))
        assert np.all((age[invalid] == 0) & (next_age[invalid] == 1) & (weights[invalid] == 0))
        assert np.all(weights[(age < 4) | ~active] == 0)
        n = np.minimum(age, horizon - 1)
        np.testing.assert_allclose(next_age[active], (n + 1)[active], atol=0.016, rtol=0.001)
        fresh = field(4) @ LUMA
        reconstructed = (n * prior_mean + fresh) / (n + 1)
        np.testing.assert_allclose(mean[active], reconstructed[active], rtol=0.003, atol=0.004)
        if reset or i <= 8:
            phases.append(dict(frame=i, active=int(active.sum()), accepted=int((weights > 0).sum()),
                               invalid=int(invalid.sum()), occluded=int((active & ((flags & 8) != 0)).sum()),
                               offscreen=int((active & ((flags & 16) != 0)).sum()),
                               reset=int((active & ((flags & 4) != 0)).sum()),
                               median_prior_age=float(np.median(age[active])) if active.any() else None))
        if reset:
            if i == 4:
                endpoint = geometry[-1]
                smooth = (endpoint[..., 0] <= 0.1) & (endpoint[..., 1] == 0) & (endpoint[..., 2] > 0)
                pose = [frame["actual_pose"][k] for k in ("x", "y", "z", "yaw_degrees", "pitch_degrees")]
                np.testing.assert_allclose(pose, fixture["poses"][-1], atol=0.0002, rtol=0)
                assert np.array_equal(smooth, active)
                assert half_matches(field(10)[..., 0][smooth], endpoint[..., 2][smooth])
            if i > 0:
                yaw = frame["actual_pose"]["yaw_degrees"] - frames[i-1]["actual_pose"]["yaw_degrees"]
                if abs((yaw + 180) % 360 - 180) > 90:
                    turns.append(i)
                    if variant == "projection":
                        # Independent receiver projection exported after shading from the displayed depth.
                        previous_clip = field(12)
                        outside = active & ((abs(previous_clip[..., :2]).max(axis=-1) > 1.005) |
                                            (previous_clip[..., 2] < -0.01))
                        phases[-1]["matrix_outside"] = int(outside.sum())
                        phases[-1]["outside_with_prior"] = int((outside & (age > 0)).sum())
                        phases[-1]["outside_with_writeback"] = int((outside & (weights > 0)).sum())
                        phases[-1]["projection_xy"] = field(14)[50, 50, :2].tolist()
                        assert outside.any()
            continue
        pose = min(i, 8)
        if pose not in pose_ids:
            continue
        j = pose_ids.index(pose)
        g = geometry[j]
        solid = (g[..., 2] > 0) & (g[..., 0] <= 0.1) & (g[..., 1] == 0)
        assert np.array_equal(solid, active)
        assert half_matches(meta[..., 0][solid], g[..., 0][solid])
        assert half_matches(field(10)[..., 0][solid], g[..., 2][solid])
        values = [frame["actual_pose"][k] for k in ("x", "y", "z", "yaw_degrees", "pitch_degrees")]
        np.testing.assert_allclose(values, fixture["poses"][j], atol=0.0002, rtol=0)
        if i == 0 or (i > 8 and i < len(frames) - 64):
            continue
        display = field(6) @ LUMA
        sharp = meta[..., 1]
        proposed = solid & (sharp - display > np.maximum(0.001, 0.01 * display))
        accepted = weights > 0
        totals = counts["moving" if i <= 8 else "held"]
        # The feedback writes a half blend. Also score the full pre-blur proposal used by the earlier stationary probe.
        for row, fraction in enumerate((0.5, 1.0)):
            midpoint = display + 0.5 * fraction * (sharp - display)
            helpful = proposed & (np.minimum(half_a[j], half_b[j]) > midpoint)
            harmful = proposed & (np.maximum(half_a[j], half_b[j]) < midpoint)
            totals[row] += [helpful.sum(), (helpful & accepted).sum(), harmful.sum(),
                            (harmful & accepted).sum(), proposed.sum()]

    result = {"frame_diagnostics": phases}
    if reset:
        assert phases[4]["reset"] == phases[4]["active"] > 0 and phases[4]["accepted"] == 0
        assert len(turns) == 2
        for i in turns:
            m = phases[i]
            assert m["offscreen"] > 0
            if metadata["scene"] == "spec-contact-2-720p":
                assert m["offscreen"] == m["active"] and m["accepted"] == 0
        assert phases[3]["accepted"] > 0 and phases[-1]["accepted"] > 0
        return result
    for period, totals in counts.items():
        for key, row in zip((period, "full_proposal_" + period), totals):
            helpful, true_pos, harmful, false_pos, proposed = row
            result[key] = dict(helpful=int(helpful), accepted_helpful=int(true_pos), harmful=int(harmful),
                               accepted_harmful=int(false_pos), proposed=int(proposed))
    result["quality"] = measure(fixture, frames, True, bundle=bundle, raw_luminance=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("job")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archive = args.fixture.parent / f"capture-{args.job}.zip"
    with np.load(args.fixture, allow_pickle=False) as fixture, (ZipFile(archive) if archive.is_file() else nullcontext()) as bundle:
        preset = json.loads(fixture["metadata"].item())["scene"]
        results = {}
        for name, patch, frames, scene in captures(args.job, preset, bundle):
            assert name in ("history8", "history32", "reset", "projection", "control")
            if name == "control":
                results[name] = dict(job=args.job, preset=preset, patch=patch, scene=scene,
                                     quality=measure(fixture, frames, True, bundle=bundle))
                continue
            horizon = 32 if name == "history32" else 8
            results[name] = dict(job=args.job, preset=preset, patch=patch, scene=scene,
                                 **score(fixture, frames, horizon, name, bundle))
    assert results
    text = json.dumps(results, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

"""Measure selective GI history writeback on a matched Vibris HDR capture.

Use a mirror-room or spec-contact reference NPZ followed by completed Vibris job IDs.
Pass --witness for captures that export the fresh pre-reuse samples and blend weight.
"""

import argparse
import hashlib
import io
import json
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from gi_mirror_weight_probe import LUMA


def hdr_tile(frame, bundle):
    artifact = frame["capture"]["artifacts"][0]
    path = Path(artifact["relative_path"])
    pixels = bundle.read(path.name) if bundle else path.read_bytes()
    assert hashlib.sha256(pixels).hexdigest() == artifact["sha256"]
    assert artifact["resource"]["frame_id"] == frame["rendered_frame_id"]
    image = np.asarray(Image.open(io.BytesIO(pixels)).convert("RGB"))[::-1]
    assert image.shape == (720, 1280, 3)

    def tile(index):
        row, col = divmod(index, 4)
        return image[row * 180:(row + 1) * 180, col * 320:(col + 1) * 320]

    ramp = np.arange(320) % 256
    np.testing.assert_array_equal(tile(15)[0], np.column_stack((ramp, 255 - ramp, np.full(320, 199))))

    def field(index):
        bits = tile(index).astype("<u2") | (tile(index + 1).astype("<u2") << 8)
        return bits.view("<f2").astype(np.float32)

    return field


def captures(job, preset, bundle=None):
    job_path = Path(".vibris/jobs", job)
    def read(name):
        return json.loads(bundle.read("job/" + name) if bundle else (job_path / name).read_bytes())

    request = read("request.json")["arguments"]
    source_patches = {source["snapshot_sha256"]: source["patch_sha256"] for source in request["sources"]}
    result = read("result.json")
    while "provenance" not in result:
        result = result["result"]
    provenance = result["provenance"]
    assert provenance["preset_id"] == preset and not provenance["stale"]
    assert not request["unfreeze_world"]
    restore = result["restoration"]
    assert restore["status"] == "RECEIPT_STATUS_OK"
    for state in ("scene_sha256", "settings_sha256", "source_sha256"):
        assert restore["actual_" + state] == restore["expected_" + state]
    for receipt in result["action_receipts"]:
        assert receipt["status"] == "RECEIPT_STATUS_OK"
        if "runtime_mutation" in receipt:
            mutation = receipt["runtime_mutation"]
            patch = source_patches[mutation["source_sha256"]]
            cfg = {s["name"]: s["value"] for s in mutation["effective_settings"]["settings"]}
            assert cfg["SETTING_GI_USE_REFERENCE"] == "0"
            assert all(cfg["SETTING_" + k] == "false" for k in ("TAA", "TAA_JITTER", "RC_ENABLE", "GI_PATH_GUIDING"))
            assert all(cfg["SETTING_DENOISER_" + k] == "true" for k in ("ACCUM", "SPATIAL", "ANTI_FIREFLY"))
        if "shader_inspection" in receipt:
            catalog = receipt["shader_inspection"]["catalog"]["programs"]
            assert catalog and all(p["compile_state"] == p["link_state"] == "COMPILE_STATE_SUCCEEDED" for p in catalog)
        if "camera_motion" in receipt:
            frames = receipt["camera_motion"]["frames"]
            for i, frame in enumerate(frames):
                assert frame["frame_offset"] == i and frame["rendered_frame_id"] == frames[0]["rendered_frame_id"] + i
                for key, actual in frame["actual_pose"].items():
                    error = actual - frame["planned_pose"][key]
                    if key == "yaw_degrees":
                        error = (error + 180) % 360 - 180
                    assert abs(error) < 0.0002
            name = Path(frames[0]["capture"]["artifacts"][0]["relative_path"]).name.split("-")[0]
            if name in ("spatialon", "spatialoff"):
                assert cfg["SETTING_GI_SPATIAL_REUSE"] == ("true" if name == "spatialon" else "false")
            yield name, patch, frames, provenance["scene_sha256"]


def dark_excursions(mask):
    # Durations are clipped to the observation window at either end.
    edges = np.diff(np.pad(mask.astype(np.int8), ((1, 1), (0, 0))), axis=0).T
    start = np.nonzero(edges == 1)[1]
    end = np.nonzero(edges == -1)[1]
    lengths = end - start
    return {"rate": float(mask.mean()), "events": len(lengths),
            "p95_frames": int(np.quantile(lengths, 0.95, method="higher")) if len(lengths) else 0,
            "at_least_8_frames": int(np.count_nonzero(lengths >= 8)),
            "full_window": int(np.count_nonzero(lengths == len(mask)))}


def half_matches(value, reference):
    tolerance = np.maximum(1e-7, np.abs(np.spacing(reference.astype(np.float16)).astype(np.float32)))
    return np.all(np.abs(value - reference) <= tolerance)


def stationary_witness(fixture, frames, bundle):
    """Score past fresh observations on the same 32 held frames for each horizon."""
    geometry = fixture["geometry"][-1].astype(np.float32)
    solid = (geometry[..., 0] <= 0.1) & (geometry[..., 1] == 0) & (geometry[..., 2] > 0)
    a = fixture["reference_96_rgb"][-1].astype(np.float32) @ LUMA
    b = 2 * (fixture["reference_192_rgb"][-1].astype(np.float32) @ LUMA) - a
    sums = np.zeros((65, *a.shape), dtype=np.float64)
    counts = {k: np.zeros(4, dtype=np.int64) for k in (1, 8, 32)}
    assert len(frames) >= 73
    for i, frame in enumerate(frames[-64:]):
        actual = [frame["actual_pose"][key] for key in ("x", "y", "z", "yaw_degrees", "pitch_degrees")]
        np.testing.assert_allclose(actual, fixture["poses"][-1], rtol=0, atol=0.0002)
        tile = hdr_tile(frame, bundle)
        meta = tile(8)
        assert np.all(meta[..., 2][solid] == 0) and half_matches(meta[..., 0][solid], geometry[..., 0][solid])
        assert half_matches(tile(10)[..., 0][solid], geometry[..., 2][solid])
        sums[i+1] = sums[i] + (tile(4) @ LUMA)
        if i < 32:
            continue
        sharp, display = meta[..., 1], tile(6) @ LUMA
        proposed = solid & (sharp - display > np.maximum(0.001, 0.01 * display))
        midpoint = (sharp + display) * 0.5
        helpful = proposed & (np.minimum(a, b) > midpoint)
        harmful = proposed & (np.maximum(a, b) < midpoint)
        for k, total in counts.items():
            accepted = (sums[i] - sums[i-k]) / k > midpoint
            total += [helpful.sum(), (helpful & accepted).sum(), harmful.sum(), (harmful & accepted).sum()]
    return {k: {"confirmed_helpful": int(h), "confirmed_harmful": int(d),
                "true_positive_rate": float(yes/h), "false_positive_rate": float(no/d)}
            for k, (h, yes, d, no) in counts.items()}


def reuse_stages(fixture, frames, bundle, spatial_terms=False):
    """Measure same-frame estimator stages or the terms of the spatial resolve."""
    meta = json.loads(fixture["metadata"].item())
    ids = meta.get("pose_ids", meta.get("frames"))
    cy, cx = meta["target_yx_in_crop"][-1] if "target_yx_in_crop" in meta else (23, 143)
    box = np.s_[cy-3:cy+4, cx-3:cx+4]
    geom = fixture["geometry"].astype(np.float32)
    ref = fixture["reference_192_rgb"][-1].astype(np.float32) @ LUMA
    a = fixture["reference_96_rgb"][-1].astype(np.float32) @ LUMA
    b = 2 * ref - a
    bright = ((geom[-1, ..., 0] <= 0.1) & (geom[-1, ..., 1] == 0) & (geom[-1, ..., 2] > 0) &
              (np.minimum(a, b) > 1) & (abs(a-b) < 0.2 * ref))
    target, population, parity = [], [], []
    for i, frame in enumerate(frames):
        pose = min(i, 8)
        if pose not in ids:
            continue
        j = ids.index(pose)
        actual = [frame["actual_pose"][key] for key in ("x", "y", "z", "yaw_degrees", "pitch_degrees")]
        np.testing.assert_allclose(actual, fixture["poses"][j], rtol=0, atol=0.0002)
        field = hdr_tile(frame, bundle)
        g = geom[j]
        solid = (g[..., 0] <= 0.1) & (g[..., 1] == 0) & (g[..., 2] > 0)
        assert half_matches(field(8)[..., 0][solid], g[..., 0][solid])
        assert half_matches(field(10)[..., 0][solid], g[..., 2][solid])
        raw, before, branch, displayed = [field(n) for n in (0, 2, 4, 6)]
        assert all(np.isfinite(rgb).all() for rgb in (raw, before, branch, displayed))
        parity.append((before[solid] - raw[solid]).ravel())
        if i > 8:
            if spatial_terms:
                weight = field(8)[..., 2]
                assert np.all((weight[solid] >= 0) & (weight[solid] <= 1))
                stages = [before @ LUMA, branch @ LUMA, raw @ LUMA, weight]
            else:
                stages = [rgb @ LUMA for rgb in (branch, before, raw, displayed)]
            target.append([float(y[box].mean()) for y in stages])
            population.append([float(y[bright].mean()) for y in stages])
    error = np.concatenate(parity)
    stages_name = "provisional_ratio_final_reservoir_weight" if spatial_terms else "fresh_temporal_raw_display"
    error_name = "provisional_raw" if spatial_terms else "temporal_raw"
    return {"held_frames": len(target), "bright_pixels": int(bright.sum()),
            "target_reference": float(ref[box].mean()), "bright_reference": float(ref[bright].mean()),
            "target_" + stages_name: np.mean(target, axis=0).tolist(),
            "bright_" + stages_name: np.mean(population, axis=0).tolist(),
            error_name + "_rgb_rmse": float(np.sqrt(np.mean(error**2, dtype=np.float64))),
            error_name + "_abs_p99": float(np.quantile(np.abs(error), 0.99))}


def measure(fixture, frames, witness, maps=None, bundle=None):
    metadata = json.loads(fixture["metadata"].item())
    ids = metadata.get("pose_ids", metadata.get("frames"))
    positions = metadata.get("target_yx_in_crop")
    # The second room has independent references at the two endpoints. The highlight at
    # the held pose was chosen on its first 96 draws before examining the tested outputs.
    held_target = (23, 143) if metadata["scene"] == "spec-contact-2-720p" else positions[-1]
    geometry = fixture["geometry"].astype(np.float32)
    reference = fixture["reference_192_rgb"].astype(np.float32) @ LUMA
    half_a = fixture["reference_96_rgb"].astype(np.float32) @ LUMA
    half_b = 2 * reference - half_a
    assert len(frames) >= 73 and ids[-1] == 8
    last = geometry[-1]
    eligible = (last[..., 0] <= 0.1) & (last[..., 1] == 0) & (last[..., 2] > 0)
    mid = eligible & (reference[-1] >= 0.1) & (reference[-1] <= 0.6) & (abs(half_a[-1] - half_b[-1]) < 0.1)
    dark = eligible & (np.maximum(half_a[-1], half_b[-1]) < 0.05)
    bright = eligible & (np.minimum(half_a[-1], half_b[-1]) > 1) & (abs(half_a[-1] - half_b[-1]) < 0.2 * reference[-1])
    target, target_raw, target_fresh, target_weight, dark_flags, mid_values, raw_signatures = [], [], [], [], [], [], []
    moving_rgb_sq = moving_rgb_n = 0
    total_weights = np.zeros(3)
    mean_errors = np.zeros(2)
    strong_fresh = np.zeros(np.count_nonzero(bright), dtype=int)
    raw_bright = np.zeros(np.count_nonzero(bright))
    fresh_bright = np.zeros_like(raw_bright)
    if maps is not None:
        means = np.zeros((3 if witness else 2, *eligible.shape), dtype=np.float32)
        hit_count = np.zeros(eligible.shape, dtype=np.uint8)
    reference_rgb_b = 2 * fixture["reference_192_rgb"].astype(np.float32) - fixture["reference_96_rgb"].astype(np.float32)

    for i, frame in enumerate(frames):
        pose = min(i, 8)
        if pose not in ids:
            continue
        j = ids.index(pose)
        expected = fixture["poses"][j]
        actual = [frame["actual_pose"][k] for k in ("x", "y", "z", "yaw_degrees", "pitch_degrees")]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0002)
        tile = hdr_tile(frame, bundle)
        meta, depth, raw, display = tile(8), tile(10)[..., 0], tile(0), tile(6)
        g = geometry[j]
        solid = (g[..., 2] > 0) & (g[..., 0] <= 0.1) & (g[..., 1] == 0)
        assert half_matches(meta[..., 0][solid], g[..., 0][solid]) and half_matches(depth[solid], g[..., 2][solid])
        raw_y, y = raw @ LUMA, display @ LUMA
        assert np.isfinite(y).all() and np.isfinite(raw_y).all()
        if witness:
            fresh_y, weight = tile(4) @ LUMA, meta[..., 2]
            assert np.isfinite(fresh_y).all() and np.all((weight[solid] >= 0) & (weight[solid] <= 1))
        if i <= 8:
            residual = display[solid] - reference_rgb_b[j][solid]
            moving_rgb_sq += float(np.square(residual).sum(dtype=np.float64))
            moving_rgb_n += residual.size
        if positions is not None or i >= 8:
            cy, cx = positions[j] if positions is not None else held_target
            box = np.s_[cy-3:cy+4, cx-3:cx+4]
            target.append((i, float(y[box].mean()), float(reference[j][box].mean())))
            target_raw.append(float(raw_y[box].mean()))
            if witness:
                target_fresh.append(float(fresh_y[box].mean()))
                target_weight.append(float(weight[box].mean()))
        if i < len(frames) - 64:
            continue
        raw_signatures.append(hashlib.sha256(raw[box].copy().tobytes()).digest())
        if maps is not None:
            means[0] += raw_y / 64
            means[1] += y / 64
            if witness:
                means[2] += fresh_y / 64
                hit_count += fresh_y > 0.5 * reference[-1]
        dark_flags.append(y[dark] > 0.5)
        mid_values.append(y[mid])
        mean_errors += [np.mean(raw_y[eligible] - reference[-1][eligible]), np.mean(y[eligible] - raw_y[eligible])]
        if witness:
            total_weights += [weight[eligible].mean(), weight[mid].mean(), weight[dark].mean()]
            strong_fresh += fresh_y[bright] > 0.5 * reference[-1][bright]
            raw_bright += raw_y[bright]
            fresh_bright += fresh_y[bright]

    offsets, tracked, refs = np.asarray(target).T
    held = tracked[offsets >= 8]
    end_ref = refs[-1]
    in_band = abs(held / end_ref - 1) <= 0.2
    recovery = np.flatnonzero(in_band[:-2] & in_band[1:-1] & in_band[2:])
    last_raw, last_display = float(np.mean(target_raw[-64:])), float(tracked[-64:].mean())
    cy, cx = held_target
    box = np.s_[cy-3:cy+4, cx-3:cx+4]
    if maps is not None:
        maps["last64_mean_luminance"] = means
        maps["tracked_display_offset_reference"] = np.column_stack((tracked, offsets, refs))
        maps["tracked_raw"] = np.asarray(target_raw)
        if witness:
            maps["last64_fresh_above_half_reference"] = hit_count
            maps["tracked_fresh"] = np.asarray(target_fresh)
            maps["tracked_weight"] = np.asarray(target_weight)
    result = {"frames": len(frames), "moving_rgb_rmse": (moving_rgb_sq/moving_rgb_n)**0.5,
              "mid_pixels": int(mid.sum()), "flicker_rms": float(np.sqrt(np.mean(np.diff(mid_values, axis=0)**2, dtype=np.float64))),
              "dark_pixels": int(dark.sum()), "dark": dark_excursions(np.stack(dark_flags)),
              "target_reference_halves": [float(half_a[-1][box].mean()), float(half_b[-1][box].mean())],
              "target_reference": float(end_ref), "target_raw": last_raw, "target_display": last_display,
              "target_raw_minus_reference": last_raw-float(end_ref), "target_display_minus_raw": last_display-last_raw,
              "first_three_in_20pct_after_stop": int(recovery[0]) if len(recovery) else None,
              "eligible_signed_raw_minus_ref_and_display_minus_raw": (mean_errors/64).tolist(),
              "tracked_raw_digest": hashlib.sha256(b"".join(raw_signatures)).hexdigest()}
    if witness:
        missing = strong_fresh == 0
        result.update(target_fresh=float(np.mean(target_fresh[-64:])), target_weight=float(np.mean(target_weight[-64:])),
                      mean_weight_eligible_mid_dark=(total_weights/64).tolist(),
                      bright_reference_pixels=int(bright.sum()), bright_with_zero_half_strength_fresh=int(missing.sum()),
                      bright_fresh_raw_reference_means=[float((fresh_bright/64).mean()), float((raw_bright/64).mean()),
                                                        float(reference[-1][bright].mean())] if bright.any() else [])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("jobs", nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--witness", action="store_true")
    mode.add_argument("--reuse-stages", action="store_true", help="Read the estimator stage diagnostics.")
    mode.add_argument("--stationary-witness", action="store_true", help="Score prior fresh samples on held production frames.")
    parser.add_argument("--maps", type=Path, help="Save per-capture endpoint maps and highlight traces as NPZ.")
    args = parser.parse_args()
    if args.maps and (args.reuse_stages or args.stationary_witness):
        parser.error("--maps is available for writeback measurements")
    arrays = {}
    with np.load(args.fixture, allow_pickle=False) as fixture:
        preset = json.loads(fixture["metadata"].item())["scene"]
        for job in args.jobs:
            local = args.fixture.parent / f"capture-{job}.zip"
            with ZipFile(local) if local.is_file() else nullcontext() as bundle:
                for name, patch, frames, scene in captures(job, preset, bundle):
                    maps = {} if args.maps else None
                    if args.stationary_witness:
                        result = stationary_witness(fixture, frames, bundle)
                    elif args.reuse_stages:
                        result = reuse_stages(fixture, frames, bundle, spatial_terms=name == "terms")
                    else:
                        result = measure(fixture, frames, args.witness, maps, bundle)
                    if maps is not None:
                        arrays.update({f"{job}/{name}/{key}": value for key, value in maps.items()})
                    print(json.dumps({"job": job, "capture": name, "patch": patch, "scene": scene, **result}, allow_nan=False), flush=True)
    if args.maps:
        with args.maps.open("xb") as out:
            np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    main()

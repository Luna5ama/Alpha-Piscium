"""Preserve and inspect the raw Vibris diffuse-direct sampling experiment."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
from light_sampling_cache import algorithm_identity
from light_sampling_experiment import Visibility, sh_basis, sphere_directions

LUMA = np.array([0.2126, 0.7152, 0.0722])


def read_image(path):
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["scalar_type"] == "FLOAT32"
    assert metadata["channel_count"] == 4
    values = np.fromfile(path, dtype="<f4").reshape(metadata["height"], metadata["width"], 4)
    assert np.isfinite(values).all(), path
    return values, metadata


def tree_stats(path):
    words = np.fromfile(path, dtype="<u4")
    counters = dict(zip(("allocated_nodes", "emitters", "overflow", "exhausted", "traces", "visible", "receivers", "guided_selections"), map(int, words[:8])))
    assert counters["overflow"] == counters["exhausted"] == 0
    assert counters["emitters"] > 0
    nodes = words[8:].reshape(-1, 8)
    floats = nodes.view("<f4")
    pending, visited, leaves = [1], set(), []
    while pending:
        index = pending.pop()
        assert index not in visited
        visited.add(index)
        node = floats[index]
        if node[3] == 0:
            continue
        link = int(nodes[index, 7])
        if link & 0x80000000:
            assert np.all(node[4:7] - node[:3] == 1)
            leaves.append((tuple(node[:3]), link & 0x7fffffff, float(node[3])))
        else:
            assert 0 < link < len(nodes) - 1
            children = floats[link:link + 2]
            assert np.isclose(node[3], children[:, 3].sum(), rtol=2e-6)
            for child in children:
                if child[3] > 0:
                    assert np.all(node[:3] <= child[:3]) and np.all(node[4:7] >= child[4:7])
            pending.extend((link, link + 1))
    assert len(leaves) == counters["emitters"]
    assert len({(p, m) for p, m, _ in leaves}) == len(leaves)
    assert np.isclose(sum(m for _, _, m in leaves), floats[1, 3], rtol=2e-6)
    counters["visible_per_receiver"] = counters["visible"] / counters["receivers"]
    counters["visible_per_trace"] = counters["visible"] / counters["traces"]
    counters["reachable_nodes"] = len(visited)
    counters["leaf_set_sha256"] = hashlib.sha256(repr(sorted(leaves)).encode()).hexdigest()
    return counters


def preserve(job_path, destination):
    job = json.loads(job_path.read_text())
    assert job["success"]
    result = job["result"]
    assert all(r["status"] == "RECEIPT_STATUS_OK" for r in result["action_receipts"])
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in result["artifacts"]:
        source = Path(artifact["relative_path"])
        target = destination / source.name
        if not target.exists():
            shutil.copy2(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == artifact["sha256"]
    if job_path.resolve() != (destination / "job-result.json").resolve():
        shutil.copy2(job_path, destination / "job-result.json")
    return result


def render(values, path, exposure):
    # The linear average is already complete. This common display transform is
    # applied only to the previews, never to the measured data.
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".cache.json")
    identity = dict(input_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                    shape=list(values.shape), dtype=values.dtype.str, exposure=exposure,
                    algorithm=algorithm_identity("light_sampling_runtime.py"),
                    numpy=np.__version__, pillow=pillow_version)
    if path.exists() and cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved["identity"] == identity and saved["output_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest():
                return "reused"
        except (ValueError, KeyError, TypeError):
            pass
    linear = np.maximum(values[::-1, :, :3], 0).astype(np.float64) * 2**exposure
    mapped = linear / (1 + linear)
    encoded = np.where(mapped <= 0.0031308, mapped * 12.92, 1.055 * mapped**(1/2.4) - 0.055)
    Image.fromarray(np.rint(encoded * 255).astype(np.uint8)).save(path)
    cache.write_text(json.dumps(dict(identity=identity, output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                                sort_keys=True), encoding="utf-8")
    return "encoded"


def training_stats(path):
    words = np.fromfile(path, dtype="<u4", count=16)
    names = ("frame", "ready", "probes", "exhausted", "bricks", "origins", "distance_xor", "distance_sum",
             "origin_xor", "origin_sum", "training_runs", "camera_brick_x", "camera_brick_y", "camera_brick_z", "reserved3", "reserved4")
    stats = dict(zip(names, map(int, words)))
    assert stats["ready"] == stats["training_runs"] == 1
    assert stats["exhausted"] == 0 and stats["probes"] == stats["origins"] * 512 > 0
    stats["fingerprint"] = [stats[n] for n in ("bricks", "origins", "probes", "distance_xor", "distance_sum", "origin_xor", "origin_sum")]
    return stats


def validate_field(path, counters):
    data = np.memmap(path, dtype="<u4", mode="r")
    assert data.nbytes == 151568384
    keys = data[:65536].reshape(4096, 16)
    counts = data[65536:69632]
    active = np.flatnonzero(counts)
    distances = data[4337664:].view("<f4").reshape(4096, 512, 16)
    moments = data[143360:4337664].view("<f4").reshape(4096, 512, 2)
    coefficients = data[69632:143360].view("<f4").reshape(4096, 9, 2)
    assert len(active) == counters["bricks"] and counts.sum() == counters["origins"]
    basis = sh_basis(sphere_directions(16))
    projector = np.linalg.pinv(basis)
    max_moment_error, max_sh_error = 0.0, 0.0
    fingerprint_xor, fingerprint_sum = np.uint32(0), 0
    def pcg(value):
        state = np.asarray(value, dtype=np.uint32) * np.uint32(747796405) + np.uint32(2891336453)
        word = ((state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state) * np.uint32(277803737)
        return (word >> np.uint32(22)) ^ word
    for brick in active:
        valid = keys[brick] != 0xffffffff
        assert valid.sum() == counts[brick]
        assert np.all(distances[brick][:, ~valid] == -1)
        transformed = distances[brick][:, valid].astype(np.float64)
        assert np.all((transformed >= 0) & (transformed <= 1))
        expected = np.column_stack((transformed.mean(1), (transformed**2).mean(1)))
        max_moment_error = max(max_moment_error, float(np.max(abs(expected - moments[brick]))))
        max_sh_error = max(max_sh_error, float(np.max(abs(projector @ expected - coefficients[brick]))))
        index = np.arange(brick * 8192, (brick + 1) * 8192, dtype=np.uint32)
        fingerprint = pcg(distances[brick].reshape(-1).view("<u4") ^ pcg(index))
        fingerprint_xor ^= np.bitwise_xor.reduce(fingerprint)
        fingerprint_sum = (fingerprint_sum + int(pcg(fingerprint).sum(dtype=np.uint64))) & 0xffffffff
    assert max_moment_error < 2e-6 and max_sh_error < 2e-5
    assert int(fingerprint_xor) == counters["distance_xor"] and fingerprint_sum == counters["distance_sum"]
    return {"data": data, "keys": keys, "counts": counts, "distances": distances, "coefficients": coefficients,
            "validation": {"max_moment_error": max_moment_error, "max_sh_error": max_sh_error, "fingerprint_recomputed": True}}


def audit_pdfs(tree_path, training_path, field, visibility, lookahead):
    words = np.fromfile(tree_path, dtype="<u4")[8:].reshape(-1, 8)
    nodes = words.view("<f4").astype(np.float64)
    records = np.fromfile(training_path, dtype="<f4", offset=64).reshape(64, 4, 4)
    header = np.fromfile(training_path, dtype="<i4", count=16)
    camera_brick = header[11:14]
    parents, leaves, pending = {}, {}, [1]
    while pending:
        index = pending.pop()
        if nodes[index, 3] == 0:
            continue
        link = int(words[index, 7])
        if link & 0x80000000:
            leaves[(*nodes[index, :3], link & 0x7fffffff)] = index
        else:
            for child in (link, link + 1):
                parents[child] = index
                pending.append(child)
    checked, max_relative_error, boundary_records = 0, 0.0, 0
    angular_grid = sphere_directions(16)
    fp_error = 16 * np.finfo(np.float32).eps
    for record in records:
        if record[2, 3] == 0:
            continue
        origin, normal = record[0, :3].astype(float), record[1, :3].astype(float)
        owner = (np.floor(origin).astype(int) >> 4) - camera_brick + 8
        inside = bool(np.all((owner >= 0) & (owner < 16)))
        brick = sum(((int(owner[a]) >> bit) & 1) << (3 * bit + a) for bit in range(4) for a in range(3)) if inside else 0xffffffff
        assert brick == int(record[3, 3].view("<u4"))
        material = int(record[0, 3].view("<u4")) & 0x7fffffff
        leaf = leaves[(*record[3, :3].astype(float), material)]
        mode = ("baseline", "distance_cdf", "binned_moments", "sh_moments")[visibility]
        estimate = None
        if visibility and inside and field["counts"][brick] > 0:
            valid = field["keys"][brick] != 0xffffffff
            transformed = field["distances"][brick][:, valid].astype(float)
            distance = np.full_like(transformed, np.inf)
            np.divide(16 * transformed, 1 - transformed, out=distance, where=transformed < 1)
            estimate = Visibility(mode, distance, field["coefficients"][brick].astype(float), 16, False)
        cache = {}
        def node_score(index):
            if index in cache:
                return cache[index]
            node = nodes[index]
            if node[3] == 0:
                return np.zeros(4)
            half = (node[4:7] - node[:3]) * 0.5
            delta = node[:3] + half - origin
            squared = delta @ delta
            radius = np.linalg.norm(half)
            cosine = np.clip((normal @ delta + radius) / np.sqrt(max(squared, .25)), .001, 1)
            geometry = node[3] * cosine / max(squared, radius**2, .25)
            factor = lower_factor = upper_factor = 1.0
            ratio = radius / np.sqrt(squared) if squared > 0 else np.inf
            if estimate is not None and ratio <= .5 + fp_error:
                direction = delta / np.sqrt(squared)
                up = np.array([1, 0, 0]) if abs(direction[1]) > .95 else np.array([0, 1, 0])
                tangent = np.cross(direction, up); tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(direction, tangent)
                center = direction * np.sqrt(1 - ratio**2)
                directions = np.array([direction, center + tangent*ratio, center - tangent*ratio,
                                       center + bitangent*ratio, center - bitangent*ratio])
                near = np.sqrt(squared) - radius
                survival = estimate.evaluate(directions, np.full(5, near), np.zeros(5, dtype=int))
                low, high = survival.copy(), survival.copy()
                if visibility in (1, 2):
                    # A CPU double atan is not a unique reference for a stepped
                    # GPU FP32 bin lookup. Bound adjacent bins only where the
                    # coordinate is within the FP32 arithmetic error band.
                    rr = (directions[:, 1] + 1) * 8
                    cc = np.mod(np.arctan2(directions[:, 2], directions[:, 0]), 2*np.pi) * (32/(2*np.pi))
                    threshold = near/(near+16)
                    for k in range(5):
                        row_error = fp_error * max(1.0, abs(rr[k]))
                        col_error = fp_error * max(1.0, abs(cc[k]))
                        row_candidates = {int(np.clip(np.floor(rr[k] + sign*row_error), 0, 15)) for sign in (-1, 1)}
                        col_candidates = {int(np.floor(cc[k] + sign*col_error)) % 32 for sign in (-1, 1)}
                        bins = [a*32+b for a in row_candidates for b in col_candidates]
                        threshold_border = np.any(abs(field["distances"][brick][bins][:, valid] - threshold) <= fp_error)
                        if len(bins) > 1 or threshold_border:
                            t_lo, t_hi = max(0.0, threshold-fp_error), min(1-1e-12, threshold+fp_error)
                            far_query, near_query = 16*t_hi/(1-t_hi), 16*t_lo/(1-t_lo)
                            low[k] = estimate.evaluate(angular_grid[bins], np.full(len(bins), far_query), np.zeros(len(bins), dtype=int)).min()
                            high[k] = estimate.evaluate(angular_grid[bins], np.full(len(bins), near_query), np.zeros(len(bins), dtype=int)).max()
                strength = (1-ratio)**2
                factor = 1-strength+strength*survival.mean() if ratio <= .5 else 1.0
                lower_factor, upper_factor = 1-strength+strength*low.mean(), 1-strength+strength*high.mean()
                if abs(ratio-.5) <= fp_error:
                    lower_factor, upper_factor = min(lower_factor, 1.0), max(upper_factor, 1.0)
            cache[index] = geometry * np.array([1.0, factor, lower_factor, upper_factor])
            return cache[index]
        def score(index, depth):
            link = int(words[index, 7])
            if depth == 0 or nodes[index, 3] == 0 or link & 0x80000000:
                return node_score(index)
            return score(link, depth-1) + score(link+1, depth-1)
        probability = np.ones(4)
        while leaf != 1:
            parent = parents[leaf]; left = int(words[parent, 7])
            a, b = score(left, 2 if lookahead else 0), score(left+1, 2 if lookahead else 0)
            branch = a / np.maximum(a+b, 1e-300)
            if a[1] + b[1] == 0:
                branch[1] = branch[0]
            branch[2] = a[2]/(a[2]+b[3]) if a[2]+b[3] else branch[0]
            branch[3] = a[3]/(a[3]+b[2]) if a[3]+b[2] else branch[0]
            probability *= branch if leaf == left else 1-branch[[0, 1, 3, 2]]
            leaf = parent
        gpu = np.array([record[2, 0], record[2, 1], record[1, 3]])
        expected = np.append(probability[:2], probability[:2] @ np.array([.1, .9]) if visibility else probability[0])
        bounded = probability[3] - probability[2] > max(1e-14, probability[1] * 1e-8)
        assert np.isclose(expected[0], gpu[0], rtol=.005, atol=1e-10)
        if bounded:
            assert probability[2]*(1-.005)-1e-10 <= gpu[1] <= probability[3]*(1+.005)+1e-10, (visibility, origin, probability, gpu)
            boundary_records += 1
            relative_error = abs(expected[0]-gpu[0])/max(expected[0], 1e-30)
        else:
            assert np.allclose(expected, gpu, rtol=.005, atol=1e-10), (visibility, lookahead, origin, normal, brick, material, record[3, :3], expected, gpu)
            relative_error = float(np.max(abs(expected-gpu)/np.maximum(expected, 1e-30)))
        assert np.isclose(gpu[2], gpu[:2] @ np.array([.1, .9]) if visibility else gpu[0], rtol=3e-7)
        max_relative_error = max(max_relative_error, relative_error)
        checked += 1
    assert checked > 0
    return {"checked_pdfs": checked, "max_relative_error": max_relative_error, "fp32_boundary_interval_checks": boundary_records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", help="Validate one scene and write scene-specific result manifests")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specification = json.loads(args.jobs.read_text())
    visibility_names = ("Off", "Directional CDF", "Binned two moments", "SH two moments")
    summary, fields, fingerprints, masks, sources, emitter_sets = [], {}, {}, {}, set(), {}
    image_rows = []
    for group in specification["groups"]:
        if args.scene and group["scene"] != args.scene:
            continue
        raw = args.out / "raw" / group["id"]
        result = preserve(args.jobs.parent / group["job"], raw)
        receipts = result["action_receipts"]
        scene = group["scene"]
        for case in group["cases"]:
            case_id = case["id"]
            mutation = receipts[case["action_index"]]["runtime_mutation"]
            effective = mutation["effective_settings"]
            settings = {s["name"]: s["value"] for s in effective["settings"]}
            for name, value in specification["common_settings"].items():
                assert settings[name] == value, (case_id, name, settings[name], value)
            assert settings["SETTING_DEBUG_LIGHT_SAMPLING"] == ("1" if case["sampler"] == "brdf" else "2")
            assert settings["SETTING_DEBUG_LIGHT_VISIBILITY"] == str(case["visibility"])
            assert settings["SETTING_DEBUG_LIGHT_LOOKAHEAD"] == str(case["lookahead"]).lower()
            single, single_meta = read_image(raw / f"{case_id}-single.bin")
            mean, mean_meta = read_image(raw / f"{case_id}-mean32.mean.bin")
            variance, _ = read_image(raw / f"{case_id}-mean32.variance.bin")
            count, _ = read_image(raw / f"{case_id}-mean32.valid_count.bin")
            incomplete, _ = read_image(raw / f"{case_id}-mean32.incomplete_batches.bin")
            assert np.all(count == 32) and np.all(incomplete == 0)
            assert np.all(variance >= 0) and np.all(variance[:, :, 3] == 0)
            if scene not in masks:
                masks[scene] = mean[:, :, 3].copy()
            assert np.array_equal(masks[scene], mean[:, :, 3])
            accumulation = mean_meta["accumulation"]
            assert accumulation["frames"] == 32
            assert accumulation["end_frame"] - accumulation["start_frame"] == 31
            training = training_stats(raw / f"{case_id}-training.bin")
            if scene not in fingerprints:
                fingerprints[scene] = training["fingerprint"]
            assert fingerprints[scene] == training["fingerprint"], (scene, case_id, fingerprints[scene], training["fingerprint"])
            field_path = raw / f"{case_id}-field.bin"
            if field_path.exists():
                fields[scene] = validate_field(field_path, training)
            assert scene in fields
            tree = tree_stats(raw / f"{case_id}-tree.bin")
            if scene not in emitter_sets:
                emitter_sets[scene] = tree["leaf_set_sha256"]
            assert emitter_sets[scene] == tree["leaf_set_sha256"]
            if case["visibility"] == 0 or case["sampler"] == "brdf":
                assert tree["guided_selections"] == 0
            batch_luma = np.array(accumulation["screen_batch_means"])[:, :3] @ LUMA
            pdf_audit = audit_pdfs(raw / f"{case_id}-tree.bin", raw / f"{case_id}-training.bin", fields[scene],
                                   case["visibility"], case["lookahead"]) if case["sampler"] == "light" else None
            row = {**case, "scene": scene, "preset": group["preset"],
                   "source_sha256": mutation["source_sha256"], "scene_sha256": mutation["scene_sha256"],
                   "settings_sha256": effective["settings_sha256"], "single_frame": single_meta["frame_id"],
                   "mean_frames": [accumulation["start_frame"], accumulation["end_frame"]],
                   "mean_rgb": mean[:, :, :3].mean((0, 1), dtype=np.float64).tolist(),
                   "mean_luminance": float(batch_luma.mean()),
                   "mean_luminance_se": float(batch_luma.std(ddof=1) / np.sqrt(len(batch_luma))),
                   "mean_rgb_variance": float(variance[:, :, :3].sum(axis=2, dtype=np.float64).mean()),
                   "single_nonzero_fraction": float(np.mean(np.max(single[:, :, :3], axis=2) > 0)),
                   "mean_nonzero_fraction": float(np.mean(np.max(mean[:, :, :3], axis=2) > 0)),
                   "tree": tree, "training": training, "field_validation": fields[scene]["validation"], "pdf_audit": pdf_audit}
            sources.add(row["source_sha256"])
            summary.append(row)
            for label, values in (("1 spp single frame", single), ("32 frame mean", mean)):
                filename = f"{scene}-{case_id}-{'single' if label.startswith('1 ') else 'mean32'}.png"
                render(values, args.out / filename, specification["exposure_ev"])
                visibilities = range(4) if case["sampler"] == "brdf" else [case["visibility"]]
                lookaheads = [False, True] if case["sampler"] == "brdf" else [case["lookahead"]]
                for vis in visibilities:
                    for look in lookaheads:
                        image_rows.append({"path": filename, "dimensions": {
                            "Scene": group["label"], "Sampler": "Cosine BRDF (fixed baseline)" if case["sampler"] == "brdf" else "Light tree",
                            "Visibility": visibility_names[vis], "Lookahead": "Two levels" if look else "Off", "Accumulation": label}})
    assert len(sources) == 1
    for scene in fingerprints:
        rows = [r for r in summary if r["scene"] == scene]
        baseline = next((r for r in rows if r["sampler"] == "brdf"), None)
        for row in rows:
            if baseline:
                row["variance_over_brdf"] = row["mean_rgb_variance"] / baseline["mean_rgb_variance"]
                difference = row["mean_luminance"] - baseline["mean_luminance"]
                row["relative_luminance_difference"] = difference / baseline["mean_luminance"]
                se = np.hypot(row["mean_luminance_se"], baseline["mean_luminance_se"])
                row["luminance_difference_z"] = float(difference / se)
            control = next((r for r in rows if r["sampler"] == "light" and r["visibility"] == 0 and r["lookahead"] == row["lookahead"]), None)
            if control:
                row["variance_over_matching_geometry"] = row["mean_rgb_variance"] / control["mean_rgb_variance"]
    suffix = "-" + args.scene if args.scene else ""
    (args.out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    (args.out / f"images{suffix}.json").write_text(json.dumps({"title": "Visibility Guidance and Geometric Lookahead - Initial Diffuse Direct Samples", "images": image_rows}, indent=2))
    print(json.dumps([{k: r[k] for k in ("scene", "id", "mean_luminance", "mean_rgb_variance", "variance_over_brdf", "variance_over_matching_geometry", "luminance_difference_z") if k in r} for r in summary], indent=2))


if __name__ == "__main__":
    main()

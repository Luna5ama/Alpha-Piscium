"""Verify and preserve a completed Vibris action job before artifact deletion."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('I:/code/mcshaders/Alpha-Piscium-4')
RUN = ROOT / 'builds/light-sampling/20261031-weights'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preserve(job_file, destination, revision, preset):
    job = json.loads(job_file.read_text(encoding='utf-8-sig'))
    assert job['success']
    result = job['result']
    provenance = result['provenance']
    assert provenance['resolved_revision'] == revision
    assert provenance['preset_id'] == preset
    assert not provenance['stale'] and not provenance['head_changed']
    assert not provenance.get('dirty_shader_delta_sha256')
    assert all(r['status'] == 'RECEIPT_STATUS_OK' for r in result['action_receipts'])
    assert not destination.exists(), destination
    destination.mkdir(parents=True)
    entries, names = [], set()
    for artifact in result['artifacts']:
        source = Path(artifact['relative_path'])
        if not source.is_absolute():
            source = ROOT / source
        name = source.name
        assert name not in names, name
        names.add(name)
        target = destination / name
        shutil.copy2(source, target)
        actual = digest(target)
        assert actual == artifact['sha256'] and digest(source) == actual, source
        entries.append(dict(original=str(source), relative_path=name, byte_size=target.stat().st_size, sha256=actual))
    shutil.copy2(job_file, destination / 'job-result.json')
    manifest = dict(revision=revision, preset=preset, files=entries)
    (destination / 'preservation.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def preserve_profile(job_file, destination, revision, preset):
    job = json.loads(job_file.read_text(encoding='utf-8-sig'))
    assert job['success'] and job['failed'] == 0
    assert job['passed'] == job['requested_cases'] == job['completed_cases']
    for case in job['cases']:
        p = case['provenance']
        assert case['status'] == 'passed' and case['frames'] == case['warmup_frames'] == 128
        assert p['resolved_revision'] == revision and p['preset_id'] == preset
        assert not p['stale'] and not p['head_changed'] and not p['dirty_shader_delta_sha256']
        settings = {s['name']: s['value'] for s in p['effective_settings']['settings']}
        assert settings['SETTING_DEBUG_LIGHT_AUDIT'] == 'false' and settings['SETTING_DEBUG_LIGHT_ARITY'] == '4'
        assert 'SETTING_DEBUG_LIGHT_WEIGHT' not in settings and 'SETTING_DEBUG_LIGHT_LOOKAHEAD' not in settings
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_file, destination / 'profile.json')
    entries = []
    for i, artifact in enumerate(job['artifacts']):
        source = Path(artifact['relative_path'])
        target = destination / f'{i:03d}-{source.name}'
        shutil.copy2(source, target)
        actual = digest(target)
        assert actual == artifact.get('sha256', actual) and digest(source) == actual
        entries.append(dict(original=str(source), relative_path=target.name, sha256=actual, byte_size=target.stat().st_size))
    (destination / 'preservation.json').write_text(json.dumps(entries, indent=2) + '\n', encoding='utf-8')
    return entries


def bracketed_profile_request(revision, preset, configs, warmup_frames=128, frames=128):
    """Capture identity around timing inside each Core-owned matrix case."""
    resources = {"camera": "iris_ssbo_0", "tree": "iris_ssbo_12",
                 "voxel-meta": "iris_ssbo_3", "voxel-materials": "iris_ssbo_4"}
    def identity(phase):
        return [dict(type="dump_buffer", resource=dict(logical_name=logical),
                     artifact_name=f"{phase}_{name}") for name, logical in resources.items()]
    return dict(worktree_root=str(ROOT), preset_id=preset, execution="async",
                sources=[dict(id="frozen", kind="commit", revision=revision)],
                configs=configs,
                matrix=dict(sources=["frozen"], configs=[config["id"] for config in configs]),
                actions=[dict(type="wait_frames", frames=warmup_frames),
                         *identity("before"), dict(type="get_gpu_metrics", frames=frames),
                         *identity("after")])


def preserve_bracketed_profiles(job_file, destination, revision, preset, warmup_frames=128, frames=128,
                                expected_identity=None, expected_configs=None):
    """Preserve raw evidence, then reject a drifting cohort before another batch."""
    from light_sampling_weight_analysis import leaf_identity, topology_identity, voxel_identity
    job = json.loads(job_file.read_text(encoding="utf-8-sig"))
    for _ in range(4):
        if "cases" in job:
            break
        if "structuredContent" in job:
            job = job["structuredContent"]
        elif "full_result_path" in job:
            full_path = Path(job["full_result_path"]).resolve()
            assert full_path.is_relative_to((ROOT / ".vibris/jobs").resolve()), full_path
            job = json.loads(full_path.read_text(encoding="utf-8-sig"))
        elif isinstance(job.get("result"), dict):
            job = job["result"]
        else:
            raise ValueError("Response has no complete matrix result or durable result reference")
    assert job["success"] and job["cases"], "Matrix did not complete successfully"
    assert not destination.exists(), destination
    destination.mkdir(parents=True)
    shutil.copy2(job_file, destination / "matrix-response.json")
    (destination / "matrix-full.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    identities = []
    for case in job["cases"]:
        name = case["case_id"]
        assert Path(name).name == name
        provenance = case["provenance"]
        assert case["status"] == "passed"
        assert provenance["resolved_revision"] == revision and provenance["preset_id"] == preset
        assert not provenance["stale"] and not provenance["head_changed"]
        assert not provenance.get("dirty_shader_delta_sha256")
        folder = destination / name
        folder.mkdir()
        settings = {x["name"]: x["value"] for x in provenance["effective_settings"]["settings"]}
        if expected_configs is not None:
            requested = expected_configs[case["config_id"]]
            for key, value in requested.items():
                expected = str(value).lower() if isinstance(value, bool) else str(value)
                assert settings.get(key) == expected, f"{name}: setting {key}: requested {expected}, got {settings.get(key)}"
        model = int(settings["SETTING_DEBUG_LIGHT_WEIGHT_MODEL"])
        receipts = case["action_receipts"]
        assert all(x["status"] == "RECEIPT_STATUS_OK" for x in receipts)
        waits = [x["wait_frames"] for x in receipts if x["kind"] == "ACTION_KIND_WAIT_FRAMES"]
        assert len(waits) == 1 and waits[0]["requested_frames"] == waits[0]["completed_frames"] == warmup_frames
        metrics = [x["gpu_metrics"] for x in receipts if x["kind"] == "ACTION_KIND_GET_GPU_METRICS"]
        assert len(metrics) == 1 and metrics[0]["sampled_frames"] == frames and metrics[0]["metrics"]
        initial = [metric for metric in metrics[0]["metrics"]
                   if metric.get("main_source", "").endswith("LightSamplingInitial.comp.glsl")]
        assert initial and all(len(metric["samples_ns"]) == frames for metric in initial), "Incomplete initial-pass samples"
        capture_frames = [x["capture"]["frame_id"] for x in receipts if x["kind"] == "ACTION_KIND_DUMP_BUFFER"]
        assert len(capture_frames) == 8 and len(set(capture_frames[:4])) == len(set(capture_frames[4:])) == 1
        assert capture_frames[-1] > capture_frames[0], "Identity captures did not bracket the measurement"
        copied = {}
        for receipt in receipts:
            for artifact in receipt.get("capture", {}).get("artifacts", []):
                source = Path(artifact["relative_path"])
                if not source.is_absolute():
                    source = ROOT / source
                prefix = name + "--"
                assert source.name.startswith(prefix), source.name
                target = folder / source.name[len(prefix):]
                if target.name in copied:
                    assert copied[target.name] == artifact["sha256"]
                    continue
                shutil.copy2(source, target)
                assert digest(target) == artifact["sha256"] and target.stat().st_size == artifact["byte_size"]
                copied[target.name] = artifact["sha256"]
        from light_sampling_cache import algorithm_identity, cached_json
        import numpy as np
        def analyze():
            phases = []
            for phase in ("before", "after"):
                tree = folder / f"{phase}_tree.bin"
                leaves = leaf_identity(tree, model)
                voxels = voxel_identity(folder, prefix=f"{phase}_", grid_size=int(settings["SETTING_VOXEL_GRID_SIZE"]))[1]
                camera = np.fromfile(folder / f"{phase}_camera.bin", "<f4")
                assert len(camera) >= 112
                # GlobalData layout in the frozen light-sampling sources: previous view inverse,
                # unjittered camera projection and its inverse. Frame/time/history fields are excluded.
                camera_identity = hashlib.sha256(np.concatenate((camera[16:32], camera[80:112])).tobytes()).hexdigest()
                phases.append(dict(camera_sha256=camera_identity, leaf_sha256=hashlib.sha256(repr(leaves).encode()).hexdigest(),
                                   topology_sha256=topology_identity(tree, model, ordered=True), voxels=voxels))
            return phases
        identity = dict(files=copied, model=model, grid=int(settings["SETTING_VOXEL_GRID_SIZE"]),
                        revision=revision, config_sha256=provenance["config_sha256"], numpy=np.__version__,
                        algorithm=algorithm_identity("light_sampling_weight_analysis.py",
                                                     "light_sampling_weight_reference.py",
                                                     "light_sampling_geometry.py", "light_sampling_weight_capture.py"))
        phases, reused = cached_json(destination.parent / ".analysis-cache", identity, analyze)
        record = dict(case_id=name, before=phases[0], after=phases[1], files=copied, analysis_reused=reused,
                      core_request_id=case.get("core_request_id"), timings=case.get("timings"))
        reference = expected_identity if expected_identity is not None else (identities[0]["before"] if identities else phases[0])
        record["differences"] = {key: dict(expected=reference.get(key), before=phases[0].get(key), after=phases[1].get(key))
                                 for key in set(reference) | set(phases[0]) | set(phases[1])
                                 if reference.get(key) != phases[0].get(key) or phases[0].get(key) != phases[1].get(key)}
        (folder / "identity.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        identities.append(record)
        assert not record["differences"], f"Input drift inside {name}; see {folder / 'identity.json'}"
    cohort = [record["before"] for record in identities]
    (destination / "cohort.json").write_text(json.dumps(identities, indent=2), encoding="utf-8")
    assert all(value == cohort[0] for value in cohort), "Matrix cases used different input cohorts"
    return identities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--preset', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--profile', action='store_true')
    mode.add_argument('--bracketed-profile', action='store_true')
    parser.add_argument('--expected-identity', type=Path)
    parser.add_argument('--expected-configs', type=Path)
    parser.add_argument('--run-root', type=Path, default=RUN)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    ready = json.loads((run_root / 'source-ready.json').read_text(encoding='utf-8-sig'))
    assert args.revision in ready.values(), 'Revision must match source-ready.json'
    assert args.destination.resolve().is_relative_to(run_root)
    function = preserve_bracketed_profiles if args.bracketed_profile else (preserve_profile if args.profile else preserve)
    options = {}
    if args.bracketed_profile:
        if args.expected_identity:
            options["expected_identity"] = json.loads(args.expected_identity.read_text(encoding="utf-8"))
        if args.expected_configs:
            options["expected_configs"] = json.loads(args.expected_configs.read_text(encoding="utf-8"))
    print(json.dumps(function(args.job, args.destination, args.revision, args.preset, **options), indent=2))

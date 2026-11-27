"""Check the native 2.1 held emitter after equipping it against a same-item shader reload."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from uuid import UUID

import numpy as np


REGION = {"x": 1660, "y": 90, "width": 130, "height": 110}
# SPBR-21 specular alpha identifies the equipped emitter in this crop.
EMISSION = {"lantern": 204, "torch": 166}
TEXTURES = {
    ("composite/composite13", "colortex8"): ("RGBA32UI", "<u4", 4),
    ("composite/composite13", "colortex9"): ("R32UI", "<u4", 1),
    ("composite/composite41", "colortex0"): ("RGBA16F", "<f2", 4),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_texture(directory, manifest_id, capture, layout):
    resource = capture["resource"]
    fmt, dtype, channels = layout
    require(resource["internal_format"] == fmt and resource["channel_count"] == channels,
            "texture layout changed")
    require((resource["source_width"], resource["source_height"]) == (1920, 1080)
            and resource["region"] == REGION and (resource["width"], resource["height"]) == (130, 110),
            "native resolution or probe region changed")
    artifact, = [a for a in capture["artifacts"] if a["role"] == "ARTIFACT_ROLE_PRIMARY"]
    recorded = Path(artifact["relative_path"].replace("\\", "/"))
    raw = directory / recorded.name
    if not raw.is_file():
        raw = recorded if recorded.is_absolute() else directory / recorded
    if not raw.is_file() and manifest_id:
        raw = directory / "raw" / str(UUID(str(manifest_id))) / recorded.name
    data = raw.read_bytes()
    require(len(data) == artifact["byte_size"] and hashlib.sha256(data).hexdigest() == artifact["sha256"],
            f"artifact differs from receipt: {raw.name}")
    return np.frombuffer(data, dtype=dtype).reshape(110, 130, channels)


def measure(receipt, candidate, emitter):
    job = json.loads(receipt.read_bytes())
    result = job["result"]
    provenance = result["provenance"]
    require(job["success"] and not provenance["stale"] and not provenance["head_changed"],
            "capture did not finish on a stable source")
    require(provenance["resolved_revision"] == candidate and not provenance.get("patch_sha256")
            and not provenance.get("dirty_shader_delta_sha256"), "capture does not match the requested commit")
    settings = {s["name"]: s["value"] for s in provenance["effective_settings"]["settings"]}
    require(settings["SETTING_EXPOSURE_MANUAL_EV_COARSE"] == "-8"
            and float(settings["SETTING_EXPOSURE_MANUAL_EV_FINE"]) == 0
            and settings["SETTING_EXPOSURE_MANUAL"] == "true", "use manual EV -8")
    conditions = tuple(provenance[k] for k in (
        "preset_id", "source_snapshot_sha256", "scene_sha256", "preset_sha256",
        "config_sha256", "pass_mapping_sha256"))
    require(all(conditions), "missing capture provenance")
    textures = {}
    frames = set()
    for action in result["action_receipts"]:
        capture = action.get("capture", {})
        name = capture.get("resource", {}).get("logical_name")
        key = (capture.get("pass_id"), name)
        if key in TEXTURES:
            require(action["status"] == "RECEIPT_STATUS_OK" and name not in textures,
                    f"failed or duplicate readback: {key}")
            frames.add(capture["frame_id"])
            textures[name] = read_texture(receipt.parent, result.get("result_manifest_id"), capture, TEXTURES[key])
    require(len(textures) == len(TEXTURES) and len(frames) == 1, "missing same-frame readbacks")
    hand = (textures["colortex9"][:, :, 0] & (1 << 24)) != 0
    emission = textures["colortex8"][:, :, 1][hand] >> 24
    require(hand.mean() > 0.8 and np.count_nonzero(emission == EMISSION[emitter]) > 0.2 * hand.size,
            f"{emitter} not confirmed in probe region")
    colors = textures["colortex0"][hand, :3].astype(np.float32)
    return conditions, job["job_id"], colors


def check(args):
    conditions = set()
    jobs = set()
    pairs = []
    for emitter in EMISSION:
        arms = []
        for receipt in getattr(args, emitter):
            try:
                setup, job_id, colors = measure(receipt, args.candidate, emitter)
                require(job_id and job_id not in jobs, "switch and reload require separate captures")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise ValueError(f"{receipt.name}: {exc}") from exc
            conditions.add(setup)
            jobs.add(job_id)
            arms.append(colors)
        pairs.append((emitter, *arms))
    require(len(conditions) == 1, "captures used different sources, settings, scenes or pass mappings")
    setup, = conditions
    preset = setup[0]
    native_wall = preset == "ap13-feedback-wall"
    if not native_wall:
        print(f"Measurements for {args.candidate} on preset {preset}")
    failed = False
    for emitter, switched, reloaded in pairs:
        require(np.isfinite(reloaded).all() and np.median(reloaded[:, 1]) > 0,
                f"{emitter} reload does not supply a finite, positive reference")
        reference = float(np.median(reloaded[:, 1]))
        value = float(np.median(switched[:, 1]))
        error = abs(value / reference - 1)
        passed = bool(np.isfinite(switched).all() and error <= 0.10)
        detail = f"{emitter}: switch {value:g}, reload {reference:g}, deviation {error:.2%}"
        if native_wall:
            detail += ": PASS" if passed else ": FAIL"
        print(detail)
        failed |= not passed
    require(native_wall, f"native wall case requires ap13-feedback-wall, captures used {preset}")
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="full Git commit used for the captures")
    for item in EMISSION:
        parser.add_argument("--" + item, required=True, nargs=2, type=Path, metavar=("SWITCH", "RELOAD"))
    try:
        sys.exit(check(parser.parse_args()))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"UNTESTED: {exc}", file=sys.stderr)
        sys.exit(2)

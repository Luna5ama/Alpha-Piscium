"""Print a Vibris run_actions request for one AP 2.0 leaf-turn replay."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="Git revision of the shader pack to replay")
    parser.add_argument("--worktree", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.worktree), "rev-parse", "--verify", "--end-of-options",
         args.revision + "^{commit}"], text=True).strip()
    # Same compact motion as the verified shared-client before/after replay.
    poses = [(0, 0, 0), (8, 0, 0), (26, 5.2, -7.6), (34, 5.2, -7.6),
             (52, 0, 0), (59, 0, 0)]
    motion = {"type": "camera_motion", "artifact_prefix": "leaf-" + revision[:10], "warmup_frames": 144,
              "keyframes": [{"frame_offset": f, "yaw_delta_degrees": yaw, "pitch_delta_degrees": pitch}
                            for f, yaw, pitch in poses]}
    request = {
        "worktree_root": str(args.worktree.resolve()), "preset_id": "ap-leaf-turn-4605290f",
        "sources": [{"id": "candidate", "kind": "commit", "revision": revision}],
        "configs": [{"id": "leaf", "values": {"SETTING_PBR_MATERIAL": "0", "SETTING_EXPOSURE_MANUAL": True,
                   "SETTING_EXPOSURE_MANUAL_EV_COARSE": "-1", "SETTING_EXPOSURE_MANUAL_EV_FINE": "-0.06"}}],
        "actions": [{"type": "load_shader", "source_id": "candidate", "config_id": "leaf"}, motion],
        "execution": "async", "restore_state": {"on_error": True, "on_success": True},
    }
    print(json.dumps(request, indent=2))


if __name__ == "__main__":
    main()

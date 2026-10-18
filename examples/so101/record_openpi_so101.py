# ruff: noqa: PLC0415
#!/usr/bin/env python3
"""Independent full-frame, absolute six-joint recorder. No hardware without --hardware."""

import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from openpi.so101.collector import RecordingSession
from openpi.so101.contract import CAMERAS
from openpi.so101.contract import FPS
from openpi.so101.contract import PROMPT
from openpi.so101.contract import VERSION
from openpi.so101.runtime import AsyncChunkPolicy
from openpi.so101.runtime import BoundedWebsocketClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Explicitly open serial/camera devices")
    parser.add_argument("--follower-port")
    parser.add_argument("--leader-port")
    parser.add_argument("--follower-id", default="so101_follower_main")
    parser.add_argument("--leader-id", default="so101_leader_main")
    parser.add_argument("--main-camera", help="OpenCV index or device path; full front view")
    parser.add_argument("--wrist-camera", help="OpenCV index or device path; full wrist view")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=Path, default=Path("data/recordings"))
    parser.add_argument("--policy-uri", help="Optional ws://localhost:8000; omitted = pure human demos")
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--max-age-s", type=float, default=0.3)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Per-command slew bound in calibrated position units; targets remain absolute",
    )
    parser.add_argument("--max-takeovers", type=int, default=0)
    parser.add_argument("--terminal-controls", action="store_true")
    args = parser.parse_args()
    if not args.hardware:
        print(
            json.dumps(
                {
                    "contract": VERSION,
                    "task": PROMPT,
                    "mode": "offline preflight; no devices opened",
                    "action": "six absolute LeRobot positions, use_degrees=False",
                    "cameras": CAMERAS,
                    "fps": FPS,
                    "roi": False,
                    "frozen_joints": [],
                    "note": "Use scripts/so101/offline_acceptance.py for fake robot validation.",
                },
                indent=2,
            )
        )
        return
    required = [args.follower_port, args.leader_port, args.main_camera, args.wrist_camera]
    if any(v is None for v in required):
        parser.error("Hardware mode requires explicit ports and both camera devices")
    if args.follower_port == args.leader_port or args.main_camera == args.wrist_camera:
        parser.error("Leader/follower ports and main/wrist cameras must differ")
    if args.max_relative_target <= 0:
        parser.error("--max-relative-target must be positive")

    # LeRobot APIs are intentionally imported only after the explicit hardware flag.
    from async_recorder import AsyncEpisodeRecorder
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    def camera_config(device):
        return OpenCVCameraConfig(
            index_or_path=int(device) if device.isdecimal() else Path(device),
            width=args.width,
            height=args.height,
            fps=FPS,
        )

    policy = None
    robot = leader = None
    session = None
    try:
        if args.policy_uri:
            client = BoundedWebsocketClient(args.policy_uri, timeout_s=args.timeout_s)
            policy = AsyncChunkPolicy(
                client, execute_steps=args.execute_steps, timeout_s=args.timeout_s, max_age_s=args.max_age_s
            )
            policy.before_episode()
            # A real request completes before devices or control window are opened.
            import numpy as np

            from openpi.so101.contract import observation_to_input

            warm = {
                "state": np.zeros(6, np.float32),
                **{c: np.zeros((args.height, args.width, 3), np.uint8) for c in CAMERAS},
            }
            client.infer(observation_to_input(warm))
        robot = SOFollower(
            SOFollowerRobotConfig(
                port=args.follower_port,
                id=args.follower_id,
                use_degrees=False,
                max_relative_target=args.max_relative_target,
                cameras={"cam_main": camera_config(args.main_camera), "cam_wrist": camera_config(args.wrist_camera)},
            )
        )
        leader = SOLeader(SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id, use_degrees=False))
        if not robot.calibration or not leader.calibration:
            raise RuntimeError("Existing calibration required; this entry never runs calibration")
        calibration = {
            role: {k: dataclasses.asdict(v) for k, v in device.calibration.items()}
            for role, device in (("follower", robot), ("leader", leader))
        }
        meta = {
            "contract": VERSION,
            "task": PROMPT,
            "batch_id": uuid.uuid4().hex,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "calibration": calibration,
            "camera_spec": {
                c: {
                    "width": args.width,
                    "height": args.height,
                    "full_frame": True,
                    "color": "RGB",
                    "device": args.main_camera if c == "cam_main" else args.wrist_camera,
                }
                for c in CAMERAS
            },
            "max_relative_target": args.max_relative_target,
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip(),
            "code_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], text=True
                ).strip()
            ),
        }
        if policy:
            expected = policy.metadata.get("source_contract", {})
            if expected.get("calibration") != calibration or expected.get("camera_spec") != meta["camera_spec"]:
                raise RuntimeError("Policy checkpoint calibration/camera contract differs from this collector")
        robot.connect(calibrate=False)
        leader.connect(calibrate=False)
        if not robot.is_calibrated or not leader.is_calibrated:
            raise RuntimeError("Motor calibration does not match; abort and calibrate separately")
        out = args.out.resolve() / time.strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=False)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        session = RecordingSession(
            robot,
            leader,
            lambda i: AsyncEpisodeRecorder(out / f"ep_{i:04d}.pkl"),
            policy=policy,
            metadata=meta,
            max_takeovers=args.max_takeovers,
        )
        if args.terminal_controls:
            from so101_keys import NonBlockingKey

            controls = NonBlockingKey()
        else:
            from control_panel import ControlPanel

            controls = ControlPanel()
        with controls as kb:
            if not kb.ok:
                raise RuntimeError("Control input unavailable")
            last_phase = None
            while session.phase not in ("CLOSED", "ERROR"):
                start = time.monotonic()
                if last_phase != session.phase:
                    last_phase = session.phase
                    detail = f"{PROMPT} · {'人工' if session.human else '模型'} · 接管 {session.count}"
                    print(f"{session.phase}: {detail}", flush=True)
                    if hasattr(kb, "set_phase"):
                        kb.set_phase(session.phase, detail)
                if hasattr(kb, "get_max_takeovers"):
                    session.max_takeovers = kb.get_max_takeovers()
                executed = session.tick(kb.read_key)
                if executed and hasattr(kb, "mark_control_step"):
                    kb.mark_control_step(session.human)
                time.sleep(max(0, 1 / FPS - (time.monotonic() - start)))
    finally:
        try:
            if session:
                session.close()
            elif policy:
                policy.close()
        finally:
            for device in (leader, robot):
                if device is not None:
                    try:
                        if device.bus.is_connected:
                            device.disconnect()
                    except Exception as exc:
                        print(f"Disconnect: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

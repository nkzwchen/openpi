"""Testable LeRobot absolute-action recording/takeover state machine.

The robot/leader/keyboard/writer are injected. Importing this module cannot open
serial ports, cameras or ROS. READY and PAUSE send no commands; RETURN is manual.
"""

import copy
import time

import numpy as np

from openpi.so101.contract import CAMERAS
from openpi.so101.contract import FPS
from openpi.so101.contract import JOINTS
from openpi.so101.contract import PROMPT
from openpi.so101.contract import UNITS
from openpi.so101.contract import VERSION
from openpi.so101.contract import array_to_positions
from openpi.so101.contract import observation_to_input
from openpi.so101.contract import positions_to_array


class RecordingSession:
    def __init__(
        self, robot, leader, writer_factory, *, policy=None, metadata=None, max_takeovers=0, clock=time.monotonic
    ):
        if max_takeovers < 0:
            raise ValueError("max_takeovers must be nonnegative")
        self.robot, self.leader, self.writer_factory, self.policy = robot, leader, writer_factory, policy
        self.clock, self.max_takeovers = clock, max_takeovers
        self.base_metadata = metadata or {}
        self.phase = "READY"
        self.human = policy is None
        self.writer = None
        self.episode_index = 0
        self.count = 0
        self._segment_start = True
        self._step = 0
        self.saved = []
        self._excluded = False
        self._safety_error = None

    def _invalidate(self, reason):
        self._segment_start = True
        if self.policy:
            self.policy.on_control_event(reason)

    def _save(self):
        if self.writer is None:
            return
        self._invalidate("episode_end")
        meta = {
            **self.base_metadata,
            "contract": VERSION,
            "task": PROMPT,
            "env_hz": FPS,
            "action_semantics": "absolute_joint_position_6d",
            "observation_semantics": "measured_joint_position_6d",
            "action_units": UNITS,
            "roi": False,
            "frozen_joints": [],
            "action_joints": list(JOINTS),
            "image_keys": list(CAMERAS),
            "operator_excluded": self._excluded,
            "quality_failed": bool(self._safety_error),
            "safety_error": self._safety_error,
            "takeover_count": self.count,
            "max_takeovers": self.max_takeovers,
            "policy": self.policy.metadata if self.policy else None,
            "policy_events": list(self.policy.events) if self.policy else [],
        }
        self.saved.append(self.writer.finish(meta))
        self.writer = None
        self.episode_index += 1

    def key(self, key):
        if key is None:
            return
        key = key.lower()
        if key == "x":
            self._invalidate("quit")
            self._save()
            self.phase = "CLOSED"
        elif self.phase == "READY" and key == "c":
            if self.policy:
                self.policy.before_episode()
            self.writer = self.writer_factory(self.episode_index)
            self.human = self.policy is None
            self.count, self._step = 0, 0
            self._excluded, self._safety_error = False, None
            self._invalidate("record_start")
            self.phase = "RECORD"
        elif self.phase in ("RECORD", "TAKEOVER_PAUSE") and key in ("d", "q"):
            self._excluded = key == "q"
            self._save()
            self.phase = "RETURN"
            self.human = True
        elif self.phase == "RECORD" and key == " " and self.policy:
            if self.human:
                self._invalidate("human_handoff")
                self.human = False
            elif self.max_takeovers and self.count >= self.max_takeovers:
                self._save()
                self.phase = "RETURN"
                self.human = True
            else:
                self._invalidate("takeover_pause")
                self.phase = "TAKEOVER_PAUSE"
        elif self.phase == "TAKEOVER_PAUSE" and key == " ":
            self._invalidate("human_start")
            self.human = True
            self.count += 1
            self.phase = "RECORD"
        elif self.phase == "RETURN" and key == "c":
            self._invalidate("reset")
            self.phase = "READY"

    def tick(self, poll_key=lambda: None):
        """One control iteration; recheck input after inference/leader read before send."""
        try:
            self.key(poll_key())
            if self.phase not in ("RECORD", "RETURN"):
                return False
            phase, human = self.phase, self.human
            started = self.clock()
            raw = self.robot.get_observation()
            obs = {"state": positions_to_array(raw), **{camera: raw[camera] for camera in CAMERAS}}
            observation_to_input(obs)
            if human:
                requested = positions_to_array(self.leader.get_action())
            else:
                requested = self.policy(obs)
                if requested is None:
                    self.key(poll_key())
                    return False
            # Any key event takes priority over the action computed from older input.
            pending = poll_key()
            if pending is not None:
                self.key(pending)
                return False
            if self.phase != phase or self.human != human:
                return False
            sent = positions_to_array(self.robot.send_action(array_to_positions(requested)))
            finished = self.clock()
            if phase == "RETURN":
                return True  # Return motions are deliberately never expert labels.
            limited = not np.allclose(requested, sent, atol=1e-6, rtol=0)
            info = {
                "intervening": human,
                "phase": "record",
                "segment_start": self._segment_start,
                "quality_ok": finished - started <= 0.25,
                "quality_problems": [] if finished - started <= 0.25 else ["slow_step"],
                "observation_monotonic_s": started,
                "command_monotonic_s": finished,
                "joint_control": {
                    "accepted": True,
                    "limited": bool(limited),
                    "requested_action": requested.tolist(),
                    "sent_action": sent.tolist(),
                },
                "policy_execution": copy.deepcopy(self.policy.last_execution) if self.policy and not human else None,
            }
            self.writer.submit(
                {
                    "observations": copy.deepcopy(obs),
                    "actions": sent.copy(),
                    "requested_action": requested.copy(),
                    "infos": info,
                    "dones": False,
                }
            )
            self._step += 1
            self._segment_start = False
            return True
        except Exception as exc:
            self._safety_error = f"{type(exc).__name__}: {exc}"
            self._invalidate("error")
            self._save()
            self.phase = "ERROR"
            raise

    def close(self):
        self._invalidate("close")
        self._save()
        self.phase = "CLOSED"
        if self.policy:
            self.policy.close()

# ruff: noqa: PT009, PT027, SLF001
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from openpi.so101.collector import RecordingSession
from openpi.so101.contract import CAMERAS
from openpi.so101.contract import array_to_positions
from openpi.so101.contract import validate_action
from openpi.so101.export import export_dataset
from openpi.so101.export import select_segments
from openpi.so101.runtime import AsyncChunkPolicy
from openpi.so101.synthetic import payload
from openpi.so101.synthetic import write_fixtures


class FakeClient:
    def __init__(self, gate=None, *, fail=False):
        self.gate = gate
        self.fail = fail
        self.started = threading.Event()

    def infer(self, obs):
        self.started.set()
        if self.gate:
            self.gate.wait(2)
        if self.fail:
            raise ConnectionError("fake disconnect")
        return {"actions": np.tile(np.array([1, 2, 3, 4, 5, 50], np.float32), (10, 1))}


class FakeRobot:
    def __init__(self):
        self.sent = []

    def get_observation(self):
        return {**array_to_positions([0, 0, 0, 0, 0, 50]), **{c: np.zeros((48, 64, 3), np.uint8) for c in CAMERAS}}

    def send_action(self, action):
        self.sent.append(action)
        return action


class FakeLeader:
    def get_action(self):
        return array_to_positions([1, 2, 3, 4, 5, 50])


class FakeWriter:
    def __init__(self):
        self.steps = []
        self.meta = None

    def submit(self, step):
        self.steps.append(copy.deepcopy(step))
        return True

    def finish(self, meta):
        self.meta = meta
        return meta


class BoundaryTests(unittest.TestCase):
    def select(self, p, horizon=3, intervals=None):
        review = {
            "version": p["meta"]["contract"],
            "recording_sha256": "hash",
            "reviewed": True,
            "status": "approved",
            "reviewer": "test",
            "intervals": intervals or [{"start": 0, "end": len(p["transitions"])}],
        }
        return select_segments(p, review, "hash", horizon)

    def test_never_compress_across_policy(self):
        segments, reasons = self.select(payload(mixed=True), 10)
        self.assertEqual(segments, [list(range(12)), list(range(18, 36))])
        self.assertEqual(reasons["not_expert"], 6)

    def test_pause_gap_and_exclusion(self):
        p = payload(16)
        p["transitions"][4]["infos"]["segment_start"] = True
        p["transitions"][8]["infos"]["quality_ok"] = False
        p["transitions"][12]["infos"]["recording_step"] += 1
        segments, _ = self.select(p)
        self.assertEqual(segments, [list(range(4)), list(range(4, 8)), list(range(9, 12)), list(range(13, 16))])

    def test_time_jump_and_terminal(self):
        p = payload(12)
        p["transitions"][4]["dones"] = True
        for t in p["transitions"][8:]:
            t["infos"]["recording_enqueued_monotonic_s"] += 2
        segments, _ = self.select(p)
        self.assertEqual(segments, [list(range(5)), list(range(5, 8)), list(range(8, 12))])

    def test_short_interval_reported(self):
        segments, reasons = self.select(payload(6), 10)
        self.assertFalse(segments)
        self.assertEqual(reasons["short_interval_frames"], 6)

    def test_adjacent_reviews_do_not_merge(self):
        segments, _ = self.select(payload(10), 3, [{"start": 0, "end": 5}, {"start": 5, "end": 10}])
        self.assertEqual(segments, [list(range(5)), list(range(5, 10))])

    def test_ignored_episode_and_wrong_units(self):
        p = payload()
        p["meta"]["operator_excluded"] = True
        self.assertFalse(self.select(p)[0])
        p = payload()
        p["meta"]["action_units"] = "radians"
        self.assertFalse(self.select(p)[0])

    def test_unreviewed_rejected(self):
        self.assertFalse(select_segments(payload(), {}, "hash", 10)[0])

    def test_limited_and_nonfinite_rejected(self):
        p = payload(10)
        p["transitions"][4]["infos"]["joint_control"]["limited"] = True
        p["transitions"][7]["actions"][0] = np.nan
        segments, _ = self.select(p, 3)
        self.assertEqual(segments, [list(range(4))])

    def test_export_split_video_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixtures(Path(tmp) / "raw")
            out = Path(tmp) / "dataset"
            report = export_dataset(paths, out)
            self.assertFalse(report["empty_splits"])
            self.assertEqual(report["splits"]["train"]["valid_starts"], 39)
            self.assertEqual(report["splits"]["validation"]["valid_starts"], 39)
            self.assertTrue(report["synthetic"])
            manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            hashes = {s: {r["source_sha256"] for r in manifest if r["split"] == s} for s in ("train", "validation")}
            self.assertFalse(hashes["train"] & hashes["validation"])
            self.assertEqual(len(list(out.glob("*/videos/*/chunk-*/*.mp4"))), 12)
            with self.assertRaises(FileExistsError):
                export_dataset(paths, out)


class ControlTests(unittest.TestCase):
    def test_absolute_all_six_joints(self):
        robot, writer = FakeRobot(), FakeWriter()
        session = RecordingSession(robot, FakeLeader(), lambda _: writer)
        session.key("c")
        session.tick()
        self.assertEqual(len(robot.sent), 1)
        np.testing.assert_equal(writer.steps[0]["actions"], [1, 2, 3, 4, 5, 50])
        self.assertEqual(set(robot.sent[0]), set(array_to_positions([1, 2, 3, 4, 5, 50])))
        session.key("d")
        session.tick()
        self.assertEqual(len(writer.steps), 1)  # RETURN not recorded
        session.close()

    def test_input_between_observation_and_send_wins(self):
        robot, writer = FakeRobot(), FakeWriter()
        session = RecordingSession(robot, FakeLeader(), lambda _: writer)
        session.key("c")
        keys = iter([None, "d"])
        session.tick(lambda: next(keys))
        self.assertFalse(robot.sent)
        self.assertEqual(session.phase, "RETURN")

    def test_slow_inference_takeover_invalidates_result(self):
        gate = threading.Event()
        client = FakeClient(gate)
        policy = AsyncChunkPolicy(client, max_age_s=2)
        robot, writer = FakeRobot(), FakeWriter()
        session = RecordingSession(robot, FakeLeader(), lambda _: writer, policy=policy)
        try:
            session.key("c")
            start = time.monotonic()
            session.tick()
            self.assertLess(time.monotonic() - start, 0.1)
            self.assertTrue(client.started.wait(1))
            session.key(" ")
            self.assertEqual(session.phase, "TAKEOVER_PAUSE")
            session.tick()
            self.assertFalse(robot.sent)
            session.key(" ")
            self.assertTrue(session.human)
            session.tick()
            self.assertEqual(len(robot.sent), 1)
            session.key(" ")
            self.assertFalse(session.human)
            gate.set()
            for _ in range(100):
                if policy._future[0].done():
                    break
                time.sleep(0.002)
            session.tick()  # stale result must be discarded, new request only
            self.assertEqual(len(robot.sent), 1)
            self.assertTrue(any(e["event"] == "discard_late_or_invalidated" for e in policy.events))
        finally:
            gate.set()
            session.close()

    def test_timeout_no_policy_commands(self):
        gate = threading.Event()
        client = FakeClient(gate)
        now = [0.0]
        policy = AsyncChunkPolicy(client, timeout_s=0.1, clock=lambda: now[0])
        try:
            obs = payload()["transitions"][0]["observations"]
            self.assertIsNone(policy(obs))
            now[0] = 1
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                policy(obs)
            policy.reset()
            with self.assertRaises(RuntimeError):
                policy(obs)
        finally:
            gate.set()
            policy.close()

    def test_network_failure_stops_recording(self):
        client = FakeClient(fail=True)
        policy = AsyncChunkPolicy(client)
        robot, writer = FakeRobot(), FakeWriter()
        session = RecordingSession(robot, FakeLeader(), lambda _: writer, policy=policy)
        try:
            session.key("c")
            session.tick()
            for _ in range(100):
                if policy._future[0].done():
                    break
                time.sleep(0.002)
            with self.assertRaises(RuntimeError):
                session.tick()
            self.assertEqual(session.phase, "ERROR")
            self.assertFalse(robot.sent)
            self.assertIn("ConnectionError", writer.meta["safety_error"])
        finally:
            session.close()

    def test_chunk_consumed_once(self):
        policy = AsyncChunkPolicy(FakeClient(), execute_steps=2, max_age_s=2)
        obs = payload()["transitions"][0]["observations"]
        try:
            self.assertIsNone(policy(obs))
            for _ in range(100):
                if policy._future[0].done():
                    break
                time.sleep(0.002)
            policy(obs)
            first = policy.last_execution.copy()
            policy(obs)
            second = policy.last_execution.copy()
            self.assertEqual((first["chunk_index"], second["chunk_index"]), (0, 1))
            self.assertEqual(first["request_id"], second["request_id"])
            self.assertIsNone(policy(obs))
        finally:
            policy.close()

    def test_nonfinite_and_bad_gripper(self):
        with self.assertRaises(ValueError):
            validate_action([0, 0, 0, 0, 0, -1])
        with self.assertRaises(ValueError):
            validate_action([np.nan] * 6)


if __name__ == "__main__":
    unittest.main()

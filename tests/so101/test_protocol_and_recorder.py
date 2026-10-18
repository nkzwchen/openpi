# ruff: noqa: PT009, PT027, SLF001, PLC0415, E402
"""Exercise the copied writer/key gate and a real loopback websocket (no hardware)."""

import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/so101"
sys.path.insert(0, str(EXAMPLES))
from async_recorder import AsyncEpisodeRecorder
from control_panel import KeyGate
from takeover_settings import load_limit
from takeover_settings import save_limit
from test_offline import FakeClient
from test_offline import FakeLeader
from test_offline import FakeRobot
from test_offline import FakeWriter

from openpi.so101.collector import RecordingSession
from openpi.so101.contract import PROMPT
from openpi.so101.contract import UNITS
from openpi.so101.contract import VERSION
from openpi.so101.runtime import AsyncChunkPolicy
from openpi.so101.runtime import BoundedWebsocketClient
from openpi.so101.synthetic import payload


class CopiedHelpersTests(unittest.TestCase):
    def test_writer_owns_snapshot_and_preserves_original_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ep.pkl"
            writer = AsyncEpisodeRecorder(dest)
            step = payload(1)["transitions"][0]
            expected = step["actions"].copy()
            self.assertTrue(writer.submit(step))
            step["actions"][:] = 99
            writer.finish({"n_intervened": 1})
            with dest.open("rb") as stream:
                saved = pickle.load(stream)
            np.testing.assert_equal(saved["transitions"][0]["actions"], expected)
            self.assertEqual(saved["transitions"][0]["infos"]["recording_step"], 0)
            self.assertTrue(saved["transitions"][0]["dones"])

    def test_gui_rejects_previous_phase_key_and_close_is_immediate(self):
        gate = KeyGate()
        gate.set_phase("RECORD")
        old = gate.generation
        gate.set_phase("TAKEOVER_PAUSE")
        gate.receive(("key", old, time.monotonic(), " "))
        self.assertIsNone(gate.read())
        gate.receive(("key", gate.generation, time.monotonic(), " "))
        self.assertEqual(gate.read(), " ")
        gate.receive(("close",))
        self.assertEqual(gate.read(), "x")

    def test_takeover_preference_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_limit(3, path)
            self.assertEqual(load_limit(path), 3)

    def test_limit_ends_record_without_new_command(self):
        robot, writer = FakeRobot(), FakeWriter()
        policy = AsyncChunkPolicy(FakeClient())
        session = RecordingSession(robot, FakeLeader(), lambda _: writer, policy=policy, max_takeovers=1)
        try:
            session.key("c")
            session.key(" ")
            session.key(" ")
            self.assertEqual(session.count, 1)
            session.key(" ")
            session.key(" ")
            self.assertEqual(session.phase, "RETURN")
            self.assertFalse(robot.sent)
        finally:
            session.close()

    def test_stale_chunks_are_not_executed(self):
        now = [0.0]
        policy = AsyncChunkPolicy(FakeClient(), max_age_s=0.1, clock=lambda: now[0])
        try:
            obs = payload()["transitions"][0]["observations"]
            policy(obs)
            for _ in range(100):
                if policy._future[0].done():
                    break
                time.sleep(0.002)
            now[0] = 0.2
            self.assertIsNone(policy(obs))
            self.assertTrue(any(e["event"] == "discard_late_or_invalidated" for e in policy.events))
        finally:
            policy.close()

    def test_synthetic_endpoint_cannot_arm_policy(self):
        client = FakeClient()
        client.get_server_metadata = lambda: {
            "contract": VERSION,
            "prompt": PROMPT,
            "action_units": UNITS,
            "synthetic": True,
        }
        policy = AsyncChunkPolicy(client)
        try:
            with self.assertRaises(RuntimeError):
                policy.before_episode()
        finally:
            policy.close()


@unittest.skipUnless(importlib.util.find_spec("websockets"), "websockets not installed in this interpreter")
class LoopbackTests(unittest.TestCase):
    def test_wire_encoding_and_receive_deadline(self):
        from openpi_client import msgpack_numpy
        from websockets.sync.server import serve

        release = threading.Event()
        calls = []

        def handler(ws):
            ws.send(msgpack_numpy.packb({"contract": VERSION, "synthetic": True}))
            obs = msgpack_numpy.unpackb(ws.recv())
            calls.append(obs)
            ws.send(msgpack_numpy.packb({"actions": np.ones((10, 6), np.float32)}))
            ws.recv()
            release.wait(2)

        server = serve(handler, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = None
        try:
            port = server.socket.getsockname()[1]
            client = BoundedWebsocketClient(f"ws://127.0.0.1:{port}", timeout_s=0.15)
            obs = {"state": np.arange(6, dtype=np.float32), "image": np.zeros((48, 64, 3), np.uint8)}
            result = client.infer(obs)
            np.testing.assert_equal(calls[0]["state"], obs["state"])
            self.assertEqual(result["actions"].shape, (10, 6))
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                client.infer(obs)
            self.assertLess(time.monotonic() - start, 1)
        finally:
            release.set()
            if client:
                client.close()
            server.shutdown()
            thread.join(1)


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires the cloud OpenPI training environment")
class ServingConfigTests(unittest.TestCase):
    def test_restore_lora_binding_and_warm_before_listening(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        source = EXAMPLES.parents[1] / "scripts/so101/serve.py"
        spec = importlib.util.spec_from_file_location("so101_serve_acceptance", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        info = {
            "export_horizon": 10,
            "source_contract": {
                "calibration": {"test": True},
                "camera_spec": {c: {"height": 48, "width": 64} for c in ("cam_main", "cam_wrist")},
            },
        }
        events = []

        def infer(obs):
            self.assertEqual(obs["observation/cam_main"].shape, (48, 64, 3))
            self.assertEqual(obs["observation/state"].shape, (6,))
            events.append("warm")
            return {"actions": np.zeros((10, 6), np.float32)}

        def server(*args, **kwargs):
            self.assertEqual(events, ["warm"])
            return SimpleNamespace(serve_forever=lambda: events.append("listen"))

        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "experiment"
            checkpoint = experiment / "100"
            checkpoint.mkdir(parents=True)
            (experiment / "so101_contract.json").write_text(
                json.dumps(
                    {
                        "dataset_report_sha256": "digest",
                        "dataset_info": info,
                        "asset_id": "absolute6_test_asset",
                        "lora": True,
                    }
                )
            )
            with (
                patch.object(module, "dataset_contract", return_value=(info, "digest")),
                patch.object(
                    module.policy_config, "create_trained_policy", return_value=SimpleNamespace(infer=infer)
                ) as load,
                patch.object(module, "WebsocketPolicyServer", side_effect=server),
                patch.object(module.subprocess, "check_output", return_value="test-commit\n"),
                patch.object(
                    sys, "argv", ["serve.py", "--checkpoint", str(checkpoint), "--dataset", str(Path(tmp) / "dataset")]
                ),
            ):
                module.main()
            config = load.call_args.args[0]
            self.assertEqual(config.model.paligemma_variant, "gemma_2b_lora")
            self.assertEqual(config.model.action_expert_variant, "gemma_300m_lora")
            self.assertEqual(config.data.assets.asset_id, "absolute6_test_asset")
            self.assertEqual(events, ["warm", "listen"])


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires OpenPI JAX environment")
class FrozenVLMFilterTests(unittest.TestCase):
    def test_freeze_vision_language_but_not_action_expert(self):
        from openpi.so101.training import frozen_vlm_filter

        frozen = frozen_vlm_filter()
        for path in (
            "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel",
            "PaliGemma/llm/embedder/input_embedding",
            "PaliGemma/llm/layers/attn/q_einsum/w",
            "PaliGemma/llm/layers/attn/q_einsum/lora_a",
        ):
            self.assertTrue(frozen(tuple(path.split("/")), None), path)
        for path in (
            "PaliGemma/llm/layers/attn/q_einsum_1/w",
            "PaliGemma/llm/layers/mlp_1/linear",
            "action_in_proj/kernel",
            "action_out_proj/kernel",
            "time_mlp_in/kernel",
        ):
            self.assertFalse(frozen(tuple(path.split("/")), None), path)


if __name__ == "__main__":
    unittest.main()

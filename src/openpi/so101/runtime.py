"""Nonblocking chunk policy with explicit invalidation; no hardware dependencies."""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time

import numpy as np

from openpi.so101.contract import PROMPT, VERSION, observation_to_input, validate_action


class AsyncChunkPolicy:
    def __init__(self, client, *, execute_steps=1, timeout_s=1.0, max_age_s=0.3, prompt=PROMPT, clock=time.monotonic):
        if execute_steps < 1 or timeout_s <= 0 or max_age_s <= 0:
            raise ValueError("Invalid execution window/deadline")
        self.client, self.execute_steps, self.timeout_s, self.max_age_s = client, execute_steps, timeout_s, max_age_s
        self.prompt, self.clock = prompt, clock
        self.metadata = {"contract": VERSION, "prompt": prompt, "execute_steps": execute_steps}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-infer")
        self._future = None
        self._chunk = None
        self._index = 0
        self._generation = 0
        self._request_id = 0
        self._closed = False
        self._fault = None
        self.last_execution = None
        self.events = []
        self._lock = threading.Lock()

    def _log(self, event, **fields):
        self.events.append({"event": event, "monotonic_s": self.clock(), "generation": self._generation, **fields})

    def invalidate(self, reason="reset"):
        with self._lock:
            self._generation += 1
            self._chunk = None
            self._index = 0
            self.last_execution = None
            self._log("invalidate", reason=reason)

    def on_control_event(self, event):
        self.invalidate(event)

    def reset(self):
        # A failed transport is intentionally not revived by reset.
        self.invalidate("reset")

    def _infer(self, inputs, generation, request_id, requested):
        started = self.clock()
        result = self.client.infer(inputs)
        return result, generation, request_id, requested, started, self.clock()

    def before_episode(self):
        if self._closed or self._fault:
            raise RuntimeError(self._fault or "Policy closed")
        self.invalidate("before_episode")
        server_metadata = getattr(self.client, "get_server_metadata", lambda: {})()
        if server_metadata:
            if server_metadata.get("contract") != VERSION or server_metadata.get("prompt") != self.prompt:
                raise RuntimeError("Policy server contract/prompt mismatch")
            self.metadata.update(server_metadata)

    def __call__(self, obs):
        """Return one fresh action or None (hold; recorder must not env.step)."""
        with self._lock:
            if self._closed or self._fault:
                raise RuntimeError(self._fault or "Policy closed")
            now = self.clock()
            if self._future is not None:
                future, generation, request_id, requested = self._future
                if not future.done() and now - requested > self.timeout_s:
                    self._fault = "Inference deadline exceeded; hold and restart client explicitly"
                    self._chunk = None
                    self._log("timeout", request_id=request_id)
                    raise RuntimeError(self._fault)
                if future.done():
                    self._future = None
                    try:
                        result, generation, request_id, requested, started, finished = future.result()
                    except Exception as exc:
                        self._fault = f"Inference failed: {type(exc).__name__}: {exc}"
                        self._log("error", request_id=request_id, error=self._fault)
                        raise RuntimeError(self._fault) from exc
                    if generation == self._generation and now - requested <= self.max_age_s:
                        chunk = np.asarray(result["actions"], dtype=np.float32)
                        if chunk.ndim != 2 or chunk.shape[0] < self.execute_steps or chunk.shape[1] != 6:
                            self._fault = "Invalid chunk shape"
                            raise RuntimeError(self._fault)
                        try:
                            for action in chunk:
                                validate_action(action)
                        except ValueError as exc:
                            self._fault = str(exc)
                            raise RuntimeError(self._fault) from exc
                        self._chunk, self._index = chunk[:self.execute_steps].copy(), 0
                        self._chunk_info = {"request_id": request_id, "chunk_id": request_id,
                                            "observation_monotonic_s": requested, "infer_start": started,
                                            "infer_end": finished, "generation": generation}
                        self._log("result", **{k:v for k,v in self._chunk_info.items() if k != "generation"})
                    else:
                        self._log("discard_late_or_invalidated", request_id=request_id)
            if self._chunk is not None:
                if now - self._chunk_info["observation_monotonic_s"] > self.max_age_s:
                    self._chunk = None
                    self._log("discard_stale_chunk")
                else:
                    action = self._chunk[self._index].copy()
                    self.last_execution = {**self._chunk_info, "chunk_index": self._index,
                                           "policy_action": action.tolist(), "source": "policy"}
                    self._index += 1
                    if self._index == len(self._chunk):
                        self._chunk = None
                    return action
            if self._future is None:
                inputs = observation_to_input(obs, self.prompt)
                self._request_id += 1
                self._future = (self._executor.submit(self._infer, copy.deepcopy(inputs), self._generation,
                                                    self._request_id, now),
                                self._generation, self._request_id, now)
                self._log("request", request_id=self._request_id)
            return None

    def close(self):
        self.invalidate("close")
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        close = getattr(self.client, "close", None)
        if close:
            close()


class BoundedWebsocketClient:
    """Finite connection/receive deadlines; no reconnect or automatic retry."""
    def __init__(self, uri, *, timeout_s=1.0):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy

        self._codec = msgpack_numpy
        self._timeout = timeout_s
        self._ws = connect(uri, compression=None, max_size=None, open_timeout=timeout_s, close_timeout=timeout_s)
        try:
            self._metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=timeout_s))
        except Exception:
            self._ws.close()
            raise

    def get_server_metadata(self):
        return self._metadata

    def infer(self, inputs):
        self._ws.send(self._codec.packb(inputs))
        result = self._ws.recv(timeout=self._timeout)
        if isinstance(result, str):
            raise RuntimeError(result)
        return self._codec.unpackb(result)

    def close(self):
        self._ws.close()

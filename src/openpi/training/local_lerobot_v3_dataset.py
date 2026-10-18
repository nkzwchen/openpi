from __future__ import annotations

import json
import io
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.common.datasets.video_utils import decode_video_frames


def _get_default_video_backend() -> str:
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:
        reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        logging.warning("torchcodec is not usable, falling back to pyav (%s)", reason)
        return "pyav"
    return "torchcodec"


def is_lerobot_v3_dataset(repo_id: str | None) -> bool:
    if repo_id is None:
        return False
    root = Path(repo_id)
    return (
        root.is_dir()
        and (root / "meta" / "info.json").is_file()
        and (root / "meta" / "tasks.parquet").is_file()
        and (root / "meta" / "episodes").is_dir()
    )


class LocalLeRobotV3Dataset(torch.utils.data.Dataset):
    """Minimal reader for local LeRobot v3 parquet datasets.

    This keeps openpi compatible with datasets that use v3 metadata files
    (tasks.parquet and meta/episodes/*.parquet) while the pinned LeRobot
    package still expects v2 jsonl metadata.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int,
        action_sequence_keys: Sequence[str],
        decode_videos: bool = True,
        video_backend: str | None = None,
    ):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.action_sequence_keys = tuple(action_sequence_keys)
        self.decode_videos = decode_videos
        self.video_backend = video_backend or (_get_default_video_backend() if decode_videos else None)

        with (self.root / "meta" / "info.json").open() as f:
            self.info = json.load(f)
        self.fps = int(self.info["fps"])
        self.features = self.info["features"]
        self.video_keys = [key for key, feature in self.features.items() if feature["dtype"] == "video"]
        self.camera_keys = [
            key for key, feature in self.features.items() if feature["dtype"] in {"image", "video"}
        ]

        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self._episode_by_index = self.episodes.set_index("episode_index", drop=False)
        self._data = self._load_data().sort_values("index").reset_index(drop=True)
        self._data_by_global_index = self._data.set_index("index", drop=False)
        self._sample_rows = np.arange(len(self._data))
        if self.info.get("full_horizon_only"):
            from openpi.so101.contract import VERSION

            if self.info.get("openpi_contract") != VERSION:
                raise ValueError("Unknown full-horizon dataset contract")
            if not (self.root.parent / "COMPLETE").is_file() or (self.root.parent / "INCOMPLETE").exists():
                raise ValueError("Dataset export is incomplete")
            if action_horizon != self.info["export_horizon"]:
                raise ValueError("Action horizon must match the reviewed export horizon")
            ends = self._data["episode_index"].map(self._episode_by_index["dataset_to_index"])
            self._sample_rows = np.flatnonzero(self._data["index"].to_numpy() + action_horizon <= ends.to_numpy())
            if len(self._sample_rows) == 0:
                raise ValueError("No full expert action windows")

    def _load_tasks(self) -> dict[int, str]:
        tasks_df = pd.read_parquet(self.root / "meta" / "tasks.parquet")
        if "task" in tasks_df.columns:
            task_texts = tasks_df["task"]
        else:
            task_texts = tasks_df.index.to_series()
        return {int(task_index): str(task) for task, task_index in zip(task_texts, tasks_df["task_index"], strict=True)}

    def _load_episodes(self) -> pd.DataFrame:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode parquet files found under {self.root / 'meta' / 'episodes'}")
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def _load_data(self) -> pd.DataFrame:
        files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No data parquet files found under {self.root / 'data'}")
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def __len__(self) -> int:
        return len(self._sample_rows)

    def __getitem__(self, index: int) -> dict:
        row = self._data.iloc[int(self._sample_rows[int(index)])]
        episode = self._episode_by_index.loc[int(row["episode_index"])]

        item = {
            "timestamp": np.asarray(row["timestamp"], dtype=np.float32),
            "frame_index": np.asarray(row["frame_index"], dtype=np.int64),
            "episode_index": np.asarray(row["episode_index"], dtype=np.int64),
            "index": np.asarray(row["index"], dtype=np.int64),
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
            "task": self.tasks[int(row["task_index"])],
        }

        for key in self.features:
            if key in item or key in self.video_keys:
                continue
            if self.features[key]["dtype"] == "image":
                if not self.decode_videos:
                    item[key] = np.zeros(self.features[key]["shape"], dtype=np.uint8)
                else:
                    value = row[key]
                    source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else self.root / value["path"]
                    with Image.open(source) as image:
                        item[key] = np.asarray(image.convert("RGB"))
                continue
            if key in self.action_sequence_keys:
                item[key] = self._query_sequence(row, episode, key)
            elif key in row:
                item[key] = self._to_array(row[key])

        for video_key in self.video_keys:
            item[video_key] = self._query_video(row, episode, video_key)

        return item

    def _query_sequence(self, row: pd.Series, episode: pd.Series, key: str) -> np.ndarray:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        current = int(row["index"])
        query_indices = [min(end - 1, max(start, current + offset)) for offset in range(self.action_horizon)]
        values = [self._data_by_global_index.loc[query_index, key] for query_index in query_indices]
        return np.stack([self._to_array(value) for value in values], axis=0)

    def _query_video(self, row: pd.Series, episode: pd.Series, video_key: str) -> np.ndarray:
        shape = tuple(self.features[video_key]["shape"])
        if not self.decode_videos:
            return np.zeros(shape, dtype=np.uint8)

        chunk = int(episode[f"videos/{video_key}/chunk_index"])
        file_index = int(episode[f"videos/{video_key}/file_index"])
        video_path = self.root / self.info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk,
            file_index=file_index,
        )
        frame = decode_video_frames(
            video_path,
            [float(row["timestamp"])],
            tolerance_s=(1.0 / self.fps) + 1e-4,
            backend=self.video_backend,
        ).squeeze(0)
        return self._torch_image_to_hwc_uint8(frame)

    @staticmethod
    def _to_array(value) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype == object:
            array = array.astype(np.float32)
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        return array

    @staticmethod
    def _torch_image_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            if array.size and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

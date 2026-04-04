"""Local H5/Parquet dataset for BEHAVIOR-1K training.

Reads from local parquet files (states, actions, metadata) and H5/MP4 files
(camera frames) to produce samples in the same format as BehaviorLeRobotDataset.

This avoids the dependency on the remote HuggingFace LeRobot dataset and provides
much faster data loading via pre-extracted JPEG blobs in H5 files.

Data layout:
  Parquet: {parquet_root}/task-XXXX/episode_XXXXXXXX.parquet
    Columns: index, episode_index, task_index, timestamp,
             observation.state, observation.cam_rel_poses, action, observation.task_info

  H5 frames (preferred, per-camera):
    {h5_root}/task-XXXX/{camera}/episode_XXXXXXXX.h5
  H5 frames (legacy, head-only):
    {h5_root}/task-XXXX/episode_XXXXXXXX.h5
  Keys in each H5: frame_indices (int32), jpeg_blob (uint8), jpeg_offsets (int64)

  Video fallback:
    {video_root}/task-XXXX/observation.images.rgb.{camera}/episode_XXXXXXXX.mp4

Camera names: head, left_wrist, right_wrist
"""

import bisect
import logging
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Camera names used in the BEHAVIOR-1K dataset
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")

# Maps camera name to the LeRobot-style key that downstream transforms expect
CAMERA_KEY_MAP = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}

# Maps camera name to the video subdirectory name
CAMERA_VIDEO_DIR = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}

# 50 BEHAVIOR-1K tasks in canonical order (task_index 0..49)
TASK_NAMES = [
    "turning_on_radio",
    "picking_up_trash",
    "putting_away_Halloween_decorations",
    "cleaning_up_plates_and_food",
    "can_meat",
    "setting_mousetraps",
    "hiding_Easter_eggs",
    "picking_up_toys",
    "rearranging_kitchen_furniture",
    "putting_up_Christmas_decorations_inside",
    "set_up_a_coffee_station_in_your_kitchen",
    "putting_dishes_away_after_cleaning",
    "preparing_lunch_box",
    "loading_the_car",
    "carrying_in_groceries",
    "bringing_in_wood",
    "moving_boxes_to_storage",
    "bringing_water",
    "tidying_bedroom",
    "outfit_a_basic_toolbox",
    "sorting_vegetables",
    "collecting_childrens_toys",
    "putting_shoes_on_rack",
    "boxing_books_up_for_storage",
    "storing_food",
    "clearing_food_from_table_into_fridge",
    "assembling_gift_baskets",
    "sorting_household_items",
    "getting_organized_for_work",
    "clean_up_your_desk",
    "setting_the_fire",
    "clean_boxing_gloves",
    "wash_a_baseball_cap",
    "wash_dog_toys",
    "hanging_pictures",
    "attach_a_camera_to_a_tripod",
    "clean_a_patio",
    "clean_a_trumpet",
    "spraying_for_bugs",
    "spraying_fruit_trees",
    "make_microwave_popcorn",
    "cook_cabbage",
    "chop_an_onion",
    "slicing_vegetables",
    "chopping_wood",
    "cook_hot_dogs",
    "cook_bacon",
    "freeze_pies",
    "canning_food",
    "make_pizza",
]


class _LRUDict(OrderedDict):
    """OrderedDict with a maximum size; evicts LRU entry on overflow."""

    def __init__(self, maxsize: int):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            self.popitem(last=False)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


class _H5Reader:
    """Manages H5 file handles with LRU caching for efficient frame access.

    Each DataLoader worker gets its own _H5Reader instance (created lazily on
    first __getitem__ call) to avoid sharing file descriptors across processes.
    """

    def __init__(self, max_open: int = 128):
        self._get_handle = lru_cache(maxsize=max_open)(self._open_h5)
        # frame_index -> array_position mapping, cached per H5 file
        self._index_cache: dict[str, dict[int, int]] = {}
        # sorted available frame indices per H5 file, for nearest-frame lookup
        self._sorted_indices_cache: dict[str, list[int]] = {}

    @staticmethod
    def _open_h5(path: str) -> h5py.File:
        return h5py.File(path, "r", swmr=True)

    def _build_index(self, h5_path: str) -> None:
        """Build and cache the frame_index -> array_position mapping."""
        h5f = self._get_handle(h5_path)
        fi = h5f["frame_indices"][:]
        mapping = {int(v): i for i, v in enumerate(fi)}
        self._index_cache[h5_path] = mapping
        self._sorted_indices_cache[h5_path] = sorted(mapping.keys())

    def read_frame(self, h5_path: str, frame_idx: int) -> np.ndarray | None:
        """Read and decode a single JPEG frame from an H5 file.

        If the exact frame_idx is not available, uses the nearest available frame.

        Returns:
            RGB uint8 array [H, W, 3] or None on failure.
        """
        if not os.path.isfile(h5_path):
            return None

        # Build index on first access
        if h5_path not in self._index_cache:
            try:
                self._build_index(h5_path)
            except Exception:
                return None

        idx_to_pos = self._index_cache[h5_path]
        pos = idx_to_pos.get(frame_idx)

        if pos is None:
            # Nearest-frame fallback
            available = self._sorted_indices_cache[h5_path]
            if not available:
                return None
            insert_pt = bisect.bisect_left(available, frame_idx)
            if insert_pt == 0:
                nearest = available[0]
            elif insert_pt >= len(available):
                nearest = available[-1]
            else:
                before, after = available[insert_pt - 1], available[insert_pt]
                nearest = before if (frame_idx - before) <= (after - frame_idx) else after
            pos = idx_to_pos[nearest]

        h5f = self._get_handle(h5_path)
        offsets = h5f["jpeg_offsets"]
        start = int(offsets[pos])
        end = int(offsets[pos + 1])
        if end <= start:
            return None

        jpeg_bytes = h5f["jpeg_blob"][start:end]
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close_all(self) -> None:
        """Close all cached file handles."""
        self._get_handle.cache_clear()
        self._index_cache.clear()
        self._sorted_indices_cache.clear()


class _VideoReader:
    """Fallback: decode a single frame from an MP4 file using cv2.

    Much slower than H5 (~4.5 fps random seek vs ~195 fps sequential H5 read).
    Only used when H5 frames are not available for a camera.
    """

    @staticmethod
    def read_frame(
        video_path: str, frame_idx: int, image_size: int | None = None
    ) -> np.ndarray | None:
        """Read a single frame from a video file.

        Returns:
            RGB uint8 array [H, W, 3] or None on failure.
        """
        if not os.path.isfile(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        if image_size is not None:
            frame = cv2.resize(frame, (image_size, image_size))

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Episode info stored in the global sample index (one entry per episode)
# ---------------------------------------------------------------------------

class _EpisodeInfo:
    """Lightweight metadata for a single episode."""
    __slots__ = ("task_idx", "episode_id", "n_rows", "parquet_path")

    def __init__(self, task_idx: int, episode_id: int, n_rows: int, parquet_path: str):
        self.task_idx = task_idx
        self.episode_id = episode_id
        self.n_rows = n_rows
        self.parquet_path = parquet_path


class LocalH5Dataset:
    """Dataset that reads BEHAVIOR-1K data from local parquet + H5/MP4 files.

    Implements the OpenPI Dataset protocol directly (__len__ + __getitem__
    returning dicts of numpy arrays) so it can be passed to TransformedDataset
    and TorchDataLoader without any wrapper.

    Output per sample (numpy arrays, matching BehaviorLeRobotDataset format):
    {
        "observation.images.rgb.head":       float32 [H, W, 3] in [0, 1],
        "observation.images.rgb.left_wrist": float32 [H, W, 3] in [0, 1],
        "observation.images.rgb.right_wrist":float32 [H, W, 3] in [0, 1],
        "observation.state":                 float32 [state_dim],
        "action":                            float32 [action_horizon, action_dim],
        "action_is_pad":                     bool    [action_horizon],
        "timestamp":                         float64,
        "episode_index":                     int64,
        "index":                             int64,
        "task_index":                        int64,
    }

    Args:
        parquet_root: Root directory containing task-XXXX/episode_XXXXXXXX.parquet.
        h5_root: Root directory containing H5 frame files. None = no H5.
        video_root: Root directory containing video files. None = no video fallback.
        tasks: Task names or indices to include. None = all 50.
        action_horizon: Number of future actions per sample.
        image_size: Target square image size. None = native resolution.
        fps: Dataset FPS (default 30).
        h5_cache_size: Max number of open H5 file handles per worker.
        df_cache_size: Max number of cached episode DataFrames per worker.
        shuffle: Whether to shuffle the global sample order at init.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        parquet_root: str,
        h5_root: str | None = None,
        video_root: str | None = None,
        tasks: Sequence[str | int] | None = None,
        action_horizon: int = 30,
        image_size: int | None = None,
        fps: int = 30,
        h5_cache_size: int = 128,
        df_cache_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.parquet_root = Path(parquet_root)
        self.h5_root = Path(h5_root) if h5_root else None
        self.video_root = Path(video_root) if video_root else None
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.fps = fps
        self._df_cache_size = df_cache_size
        self._h5_cache_size = h5_cache_size

        # These are initialized lazily per worker (see _ensure_worker_state)
        self._h5_reader: _H5Reader | None = None
        self._df_cache: _LRUDict | None = None
        self._worker_pid: int | None = None

        # Expose for OpenPI compatibility (e.g., PromptFromLeRobotTask)
        self.tasks = list(TASK_NAMES)

        # ---- Resolve task indices ----
        if tasks is None:
            task_indices = list(range(len(TASK_NAMES)))
        else:
            task_indices = []
            for t in tasks:
                if isinstance(t, int):
                    task_indices.append(t)
                else:
                    try:
                        task_indices.append(TASK_NAMES.index(t))
                    except ValueError:
                        logger.warning(f"Unknown task name '{t}', skipping")
            task_indices = sorted(set(task_indices))

        # ---- Build global sample index using parquet metadata (no data loading) ----
        logger.info(
            f"Scanning parquet metadata from {self.parquet_root} for {len(task_indices)} tasks..."
        )

        episodes: list[_EpisodeInfo] = []
        total_samples = 0
        skipped = 0

        for task_idx in task_indices:
            task_dir = self.parquet_root / f"task-{task_idx:04d}"
            if not task_dir.is_dir():
                logger.warning(f"Task directory not found: {task_dir}")
                continue

            for pf in sorted(task_dir.glob("episode_*.parquet")):
                try:
                    # Read only parquet metadata (no data), ~0.1ms per file
                    meta = pq.read_metadata(str(pf))
                    n_rows = meta.num_rows
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {pf}: {e}")
                    skipped += 1
                    continue

                if n_rows == 0:
                    continue

                episode_id = int(pf.stem.replace("episode_", ""))
                episodes.append(
                    _EpisodeInfo(task_idx, episode_id, n_rows, str(pf))
                )
                total_samples += n_rows

        if total_samples == 0:
            raise ValueError(f"No samples found in {self.parquet_root}")

        # Build flat sample index: (episode_list_idx, row_within_episode)
        # Stored as two numpy arrays for memory efficiency
        self._sample_ep_idx = np.empty(total_samples, dtype=np.int32)
        self._sample_row_idx = np.empty(total_samples, dtype=np.int32)
        offset = 0
        for ep_list_idx, ep in enumerate(episodes):
            end = offset + ep.n_rows
            self._sample_ep_idx[offset:end] = ep_list_idx
            self._sample_row_idx[offset:end] = np.arange(ep.n_rows, dtype=np.int32)
            offset = end

        self._episodes = episodes

        logger.info(
            f"LocalH5Dataset: {total_samples:,} samples, "
            f"{len(episodes):,} episodes, {len(task_indices)} tasks"
            + (f" ({skipped} parquet files skipped)" if skipped else "")
        )

        # Shuffle if requested
        if shuffle:
            rng = np.random.RandomState(seed)
            order = rng.permutation(total_samples)
            self._sample_ep_idx = self._sample_ep_idx[order]
            self._sample_row_idx = self._sample_row_idx[order]

        # ---- Detect H5 layout per camera ----
        self._h5_layout: dict[str, str] = {}  # camera -> "per_camera" | "legacy" | "none"
        if self.h5_root is not None and task_indices:
            first_task = task_indices[0]
            for cam in CAMERA_NAMES:
                per_cam_dir = self.h5_root / f"task-{first_task:04d}" / cam
                if per_cam_dir.is_dir():
                    self._h5_layout[cam] = "per_camera"
                elif cam == "head":
                    legacy_dir = self.h5_root / f"task-{first_task:04d}"
                    if legacy_dir.is_dir() and any(legacy_dir.glob("episode_*.h5")):
                        self._h5_layout[cam] = "legacy"
                    else:
                        self._h5_layout[cam] = "none"
                else:
                    self._h5_layout[cam] = "none"

            logger.info(f"H5 layout: {self._h5_layout}")

    def _ensure_worker_state(self) -> None:
        """Lazily initialize per-worker state (H5 reader, df cache).

        This must be called at the start of __getitem__. When using PyTorch
        DataLoader with num_workers > 0, each worker is a forked process that
        shares the parent's memory but needs its own file handles and caches.
        """
        pid = os.getpid()
        if self._worker_pid != pid:
            self._h5_reader = _H5Reader(max_open=self._h5_cache_size)
            self._df_cache = _LRUDict(maxsize=self._df_cache_size)
            self._worker_pid = pid

    def __len__(self) -> int:
        return len(self._sample_ep_idx)

    def _get_episode_df(self, ep: _EpisodeInfo):
        """Load episode DataFrame with LRU caching."""
        import pandas as pd

        key = (ep.task_idx, ep.episode_id)
        assert self._df_cache is not None
        if key not in self._df_cache:
            self._df_cache[key] = pd.read_parquet(ep.parquet_path)
        return self._df_cache[key]

    def _get_h5_path(self, task_idx: int, episode_id: int, camera: str) -> str | None:
        """Resolve H5 file path for a given camera, or None if unavailable."""
        if self.h5_root is None:
            return None

        layout = self._h5_layout.get(camera, "none")
        if layout == "per_camera":
            p = self.h5_root / f"task-{task_idx:04d}" / camera / f"episode_{episode_id:08d}.h5"
            return str(p) if p.is_file() else None
        elif layout == "legacy" and camera == "head":
            p = self.h5_root / f"task-{task_idx:04d}" / f"episode_{episode_id:08d}.h5"
            return str(p) if p.is_file() else None
        return None

    def _get_video_path(self, task_idx: int, episode_id: int, camera: str) -> str | None:
        """Resolve video file path for a given camera, or None if unavailable."""
        if self.video_root is None:
            return None
        p = (
            self.video_root
            / f"task-{task_idx:04d}"
            / CAMERA_VIDEO_DIR[camera]
            / f"episode_{episode_id:08d}.mp4"
        )
        return str(p) if p.is_file() else None

    def _load_frame(
        self, task_idx: int, episode_id: int, frame_idx: int, camera: str
    ) -> np.ndarray:
        """Load a single camera frame (H5 -> video -> zeros fallback).

        Returns:
            float32 array [H, W, 3] in [0, 1].
        """
        frame = None

        # Try H5 first (fast path)
        h5_path = self._get_h5_path(task_idx, episode_id, camera)
        if h5_path is not None:
            assert self._h5_reader is not None
            frame = self._h5_reader.read_frame(h5_path, frame_idx)

        # Fallback to video decode (slow path)
        if frame is None:
            video_path = self._get_video_path(task_idx, episode_id, camera)
            if video_path is not None:
                frame = _VideoReader.read_frame(video_path, frame_idx, self.image_size)

        # Last resort: zeros
        if frame is None:
            size = self.image_size or 224
            return np.zeros((size, size, 3), dtype=np.float32)

        # Resize if needed
        if self.image_size is not None:
            h, w = frame.shape[:2]
            if (h, w) != (self.image_size, self.image_size):
                frame = cv2.resize(frame, (self.image_size, self.image_size))

        return frame.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict:
        self._ensure_worker_state()

        ep_list_idx = int(self._sample_ep_idx[idx])
        row_idx = int(self._sample_row_idx[idx])
        ep = self._episodes[ep_list_idx]

        df = self._get_episode_df(ep)
        row = df.iloc[row_idx]

        # ---- Metadata ----
        timestamp = float(row["timestamp"])
        episode_index = int(row["episode_index"])
        global_index = int(row["index"])
        task_index = int(row["task_index"])

        # ---- Camera images ----
        frame_idx = row_idx  # 0-based within episode
        item: dict = {}
        for camera in CAMERA_NAMES:
            key = CAMERA_KEY_MAP[camera]
            item[key] = self._load_frame(ep.task_idx, ep.episode_id, frame_idx, camera)

        # ---- Observation state (proprioception) ----
        state_raw = row["observation.state"]
        if isinstance(state_raw, (list, np.ndarray)):
            state = np.asarray(state_raw, dtype=np.float32)
        else:
            state = np.asarray(eval(state_raw), dtype=np.float32)  # noqa: S307
        item["observation.state"] = state

        # ---- Actions with future horizon ----
        n_rows = ep.n_rows
        actions_list: list[np.ndarray] = []
        is_pad_list: list[bool] = []

        for t in range(self.action_horizon):
            future_row = row_idx + t
            if future_row < n_rows:
                action_raw = df.iloc[future_row]["action"]
                if isinstance(action_raw, (list, np.ndarray)):
                    action = np.asarray(action_raw, dtype=np.float32)
                else:
                    action = np.asarray(eval(action_raw), dtype=np.float32)  # noqa: S307
                actions_list.append(action)
                is_pad_list.append(False)
            else:
                # Pad by repeating last valid action
                actions_list.append(actions_list[-1].copy())
                is_pad_list.append(True)

        item["action"] = np.stack(actions_list, axis=0)  # [action_horizon, action_dim]
        item["action_is_pad"] = np.array(is_pad_list, dtype=bool)  # [action_horizon]

        # ---- Scalar metadata ----
        item["timestamp"] = np.float64(timestamp)
        item["episode_index"] = np.int64(episode_index)
        item["index"] = np.int64(global_index)
        item["task_index"] = np.int64(task_index)

        return item


def create_local_h5_dataset(
    parquet_root: str,
    h5_root: str | None,
    video_root: str | None,
    tasks: Sequence[str] | None,
    action_horizon: int = 30,
    image_size: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    h5_cache_size: int = 128,
) -> LocalH5Dataset:
    """Create a LocalH5Dataset instance.

    Args:
        parquet_root: Path to parquet data directory.
        h5_root: Path to H5 frame directory (None = no H5, video-only).
        video_root: Path to video directory (None = no video fallback).
        tasks: List of task names to include. None = all 50.
        action_horizon: Number of future actions per sample.
        image_size: Target square image size (None = native).
        shuffle: Whether to shuffle samples.
        seed: Random seed.
        h5_cache_size: LRU cache size for H5 file handles.

    Returns:
        LocalH5Dataset instance compatible with OpenPI Dataset protocol.
    """
    return LocalH5Dataset(
        parquet_root=parquet_root,
        h5_root=h5_root,
        video_root=video_root,
        tasks=tasks,
        action_horizon=action_horizon,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        h5_cache_size=h5_cache_size,
    )

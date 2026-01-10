"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

# ============================================================================
# CRITICAL FIX: Disable torch.compile to prevent fork bomb in data workers
# ============================================================================
# Problem: OmniGibson's BehaviorLeRobotDataset imports transform_utils.py which has
# ~30 functions decorated with @torch_compile. When imported in each data worker,
# these decorators get activated. On first call, torch.compile spawns ~32 compiler
# worker processes PER function being compiled.
#
# Result: 32 data workers × 30 functions × 32 compiler workers = 30,000+ processes
# This causes a "fork bomb" that maxes out CPU with process management overhead.
#
# Solution: Monkey-patch torch.compile to be a no-op. The OmniGibson functions
# still work correctly (they're just not JIT-compiled), but we avoid the explosion.
# This is safe because:
# 1. We only need data loading, not simulation (no performance-critical ops)
# 2. torch.compile is for inference optimization, not needed for data prep
# 3. The @torch_compile decorator becomes effectively @no_op
# ============================================================================
import os
os.environ['PYTORCH_JIT'] = '0'  # Disable JIT compilation

import torch
# Monkey-patch torch.compile to be a no-op in data workers
_original_compile = torch.compile
def _noop_compile(model, *args, **kwargs):
    """No-op torch.compile for data workers to prevent fork bomb."""
    return model
torch.compile = _noop_compile

import logging
import time

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
    # Grain imports (JAX-native data loading)
    GRAIN_AVAILABLE,
    GrainDataLoader,
    GrainTransformedDataLoader,
    GrainDataSource,
    GrainTransformOp,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader | GrainDataLoader | GrainTransformedDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.
    
    Uses OmniGibson's BehaviorLeRobotDataset for efficient loading of BEHAVIOR-1K data.
    
    Args:
        data_config: Data configuration
        action_horizon: Action horizon for delta timestamps
        seed: Random seed for shuffling. If None, uses random seed based on current time.
    
    Returns:
        Dataset instance with BEHAVIOR-1K data
    """
    from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
    
    # Use random seed if not provided
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for BehaviorLeRobotDataset: {seed}")
    # tasks = [
    # "picking_up_trash", # difficulty: 2
    # "putting_away_Halloween_decorations", # difficulty: 3
    # "cleaning_up_plates_and_food", # difficulty: 3.5
    # "setting_mousetraps", # difficulty: 2
    # "hiding_Easter_eggs", # difficulty: 2
    # "set_up_a_coffee_station_in_your_kitchen", # difficulty: 3
    # "putting_dishes_away_after_cleaning", # difficulty: 3
    # "preparing_lunch_box", # difficulty: 3
    # "loading_the_car", # difficulty: 3.5
    # "carrying_in_groceries", # difficulty: 3.5
    # "turning_on_radio", # difficulty: 1
    # "picking_up_toys", # difficulty: 3.5
    # "can_meat", # difficulty: 3.5
    # "rearranging_kitchen_furniture", # difficulty: 3
    # "putting_up_Christmas_decorations_inside", # difficulty: 2
    # "bringing_in_wood", # difficulty: 1.5
    # "moving_boxes_to_storage", # difficulty: 1.5
    # "bringing_water", # difficulty: 1.5
    # "tidying_bedroom", # difficulty: 2
    # "outfit_a_basic_toolbox", # difficulty: 2,
    # "sorting_vegetables",
    # "collecting_childrens_toys",
    # "putting_shoes_on_rack",
    # "boxing_books_up_for_storage",
    # "storing_food",
    # "clearing_food_from_table_into_fridge",
    # "assembling_gift_baskets",
    # "sorting_household_items",
    # "getting_organized_for_work",
    # "clean_up_your_desk",
    # "setting_the_fire",
    # "clean_boxing_gloves",
    # "wash_a_baseball_cap",
    # "wash_dog_toys",
    # "hanging_pictures",
    # "attach_a_camera_to_a_tripod",
    # "clean_a_patio",
    # "clean_a_trumpet",
    # "spraying_for_bugs",
    # "spraying_fruit_trees",
    # "make_microwave_popcorn",
    # "cook_cabbage",
    # "make_pizza",
    # "chop_an_onion",
    # "slicing_vegetables",
    # "chopping_wood",
    # "canning_food",
    # "cook_hot_dogs",
    # "cook_bacon",
    # "freeze_pies",
    # ]

    tasks = [
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
    
    # Select specific task indices
    tasks = [tasks[i] for i in [2, 3, 5, 6, 10, 11, 13, 14, 15, 19, 23, 24, 25, 28, 29, 34, 42, 44, 47, 48]]
    
    
    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=tasks,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=True,
        seed=seed,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False, model_config=None) -> Dataset:
    """Transform dataset with B1K-specific per-timestamp normalization support.
    
    CRITICAL: This overrides wensi-ai's transform_dataset to pass use_per_timestamp to Normalize.
    wensi-ai's version doesn't support per-timestamp normalization which causes huge action losses!
    
    Args:
        dataset: Base dataset to transform
        data_config: Data configuration
        skip_norm_stats: Whether to skip normalization
        model_config: Model configuration (optional, used for predicate transforms)
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support (wensi-ai's doesn't have it!)
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm  # CRITICAL: Per-timestamp normalization!
        ),
    ]
    
    # Add subtask state computation for PI_BEHAVIOR models (needs dataset reference)
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        # ComputeSubtaskStateFromMeta needs dataset reference to access episode lengths
        if hasattr(transform, '__class__') and transform.__class__.__name__ == 'ComputeSubtaskStateFromMeta':
            # Replace placeholder with dataset-aware version
            from b1k import transforms as b1k_transforms
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                model_transforms.append(b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset))
                logging.info("Added dataset-aware ComputeSubtaskStateFromMeta transform")
            else:
                logging.warning("Skipping subtask state computation - dataset has no meta.episodes")
        else:
            model_transforms.append(transform)
    
    transforms_list.extend(model_transforms)
    
    # Add predicate state transform for PI_BEHAVIOR models
    if model_config is not None and hasattr(model_config, 'predicate_data_path'):
        from b1k import transforms as b1k_transforms
        transforms_list.append(b1k_transforms.ComputePredicateStateFromData(
            predicate_data_path=model_config.predicate_data_path
        ))
        logging.info(f"Added ComputePredicateStateFromData transform (path: {model_config.predicate_data_path})")

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    """Extract episode lengths from B1K dataset metadata.
    
    Args:
        dataset: BehaviorLeRobotDataset instance
        
    Returns:
        Dictionary mapping episode_index to episode_length (in frames)
        
    Raises:
        ValueError: If dataset doesn't have required metadata
    """
    if not hasattr(dataset, 'episode_data_index'):
        raise ValueError("Dataset must have episode_data_index attribute")
    
    episode_data_index = dataset.episode_data_index
    if 'to' not in episode_data_index or 'from' not in episode_data_index:
        raise ValueError("episode_data_index must have 'to' and 'from' keys")
    
    episode_to = episode_data_index['to'] 
    episode_from = episode_data_index['from']
    episodes = dataset.episodes
    
    episode_lengths = {}
    for i, episode_index in enumerate(episodes):
        if i < len(episode_to) and i < len(episode_from):
            episode_length = episode_to[i] - episode_from[i]
            episode_lengths[episode_index] = float(episode_length)
    
    logging.info(f"Extracted {len(episode_lengths)} episode lengths from dataset")
    return episode_lengths


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, model_config=config.model)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)


def create_behavior_data_loader_grain(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    num_workers: int = 4,
    prefetch_buffer_size: int = 2,
) -> DataLoader:
    """Create a JAX-native data loader for BEHAVIOR-1K training using grain.
    
    This is more efficient than create_behavior_data_loader() because:
    1. No main process collation bottleneck - workers write directly to device memory
    2. Transforms run in parallel worker threads
    3. Built-in multi-host support for distributed training
    4. Efficient prefetching and pipelining
    
    Args:
        config: Training configuration
        sharding: JAX sharding spec. If None, uses data parallel sharding.
        shuffle: Whether to shuffle the data.
        num_batches: Number of batches per epoch. If None, iterates indefinitely.
        skip_norm_stats: Whether to skip normalization.
        num_workers: Number of worker threads (recommended: 4-8 per host).
        prefetch_buffer_size: Number of batches to prefetch.
    
    Returns:
        DataLoader that yields (Observation, Actions) tuples.
    """
    import jax
    import time
    
    if not GRAIN_AVAILABLE:
        logging.warning("grain not available, falling back to TorchDataLoader")
        return create_behavior_data_loader(
            config, sharding=sharding, shuffle=shuffle, 
            num_batches=num_batches, skip_norm_stats=skip_norm_stats
        )
    
    # Import grain here to get access to grain module
    import grain.python as grain
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    # Create base dataset (without transforms)
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    
    # Build transform list with B1K-specific per-timestamp normalization
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm
        ),
    ]
    
    # Add subtask state computation for PI_BEHAVIOR models
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        if hasattr(transform, '__class__') and transform.__class__.__name__ == 'ComputeSubtaskStateFromMeta':
            from b1k import transforms as b1k_transforms
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                model_transforms.append(b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset))
                logging.info("Added dataset-aware ComputeSubtaskStateFromMeta transform")
            else:
                logging.warning("Skipping subtask state computation - dataset has no meta.episodes")
        else:
            model_transforms.append(transform)
    transforms_list.extend(model_transforms)
    
    # Add predicate state transform for PI_BEHAVIOR models
    model_config = config.model
    if hasattr(model_config, 'predicate_data_path'):
        from b1k import transforms as b1k_transforms
        transforms_list.append(b1k_transforms.ComputePredicateStateFromData(
            predicate_data_path=model_config.predicate_data_path
        ))
        logging.info(f"Added ComputePredicateStateFromData transform (path: {model_config.predicate_data_path})")
    
    # Set up sharding
    if sharding is None:
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)),
            jax.sharding.PartitionSpec("B"),
        )
    
    # Use GrainTransformedDataLoader
    data_loader = GrainTransformedDataLoader(
        dataset=dataset,
        transforms=transforms_list,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        prefetch_buffer_size=prefetch_buffer_size,
    )
    
    logging.info(f"Created Grain data loader with {num_workers} workers, batch_size={config.batch_size}")
    
    return DataLoaderImpl(data_config, data_loader)

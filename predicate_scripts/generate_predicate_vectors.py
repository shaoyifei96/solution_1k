#!/usr/bin/env python3
"""
Generate predicate state vectors from JSONL predicate data.

Each dimension corresponds to one top-level predicate:
- atomic: Binary 0/1
- forall: count / threshold (normalized 0-1)
- exists: Depends on nested predicate structure
- not: Binary 0/1
- forpairs: count / threshold (normalized 0-1)
- or: Binary 0/1

Output format: numpy arrays compatible with JAX/PyTorch training.
"""

import os
import json
import argparse
import re
import av
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Check for GPU availability
try:
    import torch as th
    HAS_CUDA = th.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    th = None


# =============================================================================
# VISIBILITY UTILITIES
# =============================================================================

# BDDL to scene object category synonyms (WordNet relationships)
# NOTE: This is a fallback. The preferred method is to use inst_to_name from HDF5 metadata.
BDDL_TO_SCENE_SYNONYMS = {
    'ashcan': 'trash_can',
    'radio_receiver': 'radio',
    'caldron': 'cauldron',
    'electric_refrigerator': 'fridge',
    'candle': 'pillar_candle',  # Might need adjustment per task
}

# Camera key mapping from human-readable names to metadata keys
CAMERA_KEYS = {
    'head': 'robot_r1::robot_r1:zed_link:Camera:0::unique_ins_ids',
    'left_wrist': 'robot_r1::robot_r1:left_realsense_link:Camera:0::unique_ins_ids',
    'right_wrist': 'robot_r1::robot_r1:right_realsense_link:Camera:0::unique_ins_ids',
}


def parse_bddl_category(bddl_name: str) -> str:
    """
    Parse BDDL object name to extract the category.
    
    Examples:
        'can__of__soda.n.01_1' -> 'can_of_soda'
        'radio_receiver.n.01_1' -> 'radio_receiver'
        'floor.n.01_1' -> 'floor'
    """
    # Remove the instance suffix (e.g., _1, _2)
    parts = bddl_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        name_without_instance = parts[0]
    else:
        name_without_instance = bddl_name
    
    # Extract category from WordNet format (e.g., 'can__of__soda.n.01')
    if '.n.' in name_without_instance or '.v.' in name_without_instance:
        category = name_without_instance.split('.')[0]
    else:
        category = name_without_instance
    
    # Convert double underscores to single
    return category.replace('__', '_')


def get_scene_category(bddl_category: str) -> str:
    """Map BDDL category to scene object category using synonyms."""
    return BDDL_TO_SCENE_SYNONYMS.get(bddl_category, bddl_category)


def load_inst_to_name_from_hdf5(
    task_id: int,
    episode_id: str,
    rawdata_path: str = "/vast/projects/kumar/lab/yishao/data/behavior_rawdata"
) -> Dict[str, str]:
    """
    Load the BDDL instance to scene object name mapping from HDF5 file.
    
    This is the ground truth mapping used by the simulator, stored in:
    scene_file -> metadata -> task -> inst_to_name
    
    Args:
        task_id: Task ID number
        episode_id: Episode ID string (e.g., "00010010")
        rawdata_path: Path to rawdata directory
        
    Returns:
        Dict mapping BDDL instance names to scene object names, e.g.:
        {
            "ashcan.n.01_1": "trash_can_116",
            "can__of__soda.n.01_1": "can_of_soda_115",
            ...
        }
    """
    hdf5_path = Path(rawdata_path) / f'task-{task_id:04d}' / f'episode_{episode_id}.hdf5'
    
    if not hdf5_path.exists():
        print(f"    Warning: HDF5 file not found at {hdf5_path}")
        return {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            scene_file_str = f['data'].attrs.get('scene_file', b'{}')
            if isinstance(scene_file_str, bytes):
                scene_file_str = scene_file_str.decode()
            
            scene_data = json.loads(scene_file_str)
            inst_to_name = scene_data.get('metadata', {}).get('task', {}).get('inst_to_name', {})
            
            return inst_to_name
    except Exception as e:
        print(f"    Warning: Failed to load inst_to_name from {hdf5_path}: {e}")
        return {}


def is_instance_name(name: str) -> bool:
    """
    Check if a BDDL name is an instance (e.g., 'ashcan.n.01_1') vs a category template (e.g., 'ashcan.n.01').
    
    Instance names end with _N where N is a digit.
    """
    if not isinstance(name, str):
        return False
    if '.n.' not in name and '.v.' not in name:
        return False
    # Check if it ends with _N where N is a digit
    parts = name.rsplit('_', 1)
    return len(parts) == 2 and parts[1].isdigit()


def extract_objects_from_predicate(pred: Dict[str, Any]) -> Set[str]:
    """
    Extract all BDDL object names involved in a predicate.
    
    Args:
        pred: Predicate dict from JSONL
        
    Returns:
        Set of BDDL object names (e.g., {'can__of__soda.n.01_1', 'ashcan.n.01_1'})
    """
    objects = set()
    
    # Check instances (most reliable source) - these are always actual instances
    if 'instances' in pred:
        for inst_name in pred['instances'].keys():
            if is_instance_name(inst_name):
                objects.add(inst_name)
    
    # Check args for target objects - only add if it's an instance name (has _N suffix)
    if 'args' in pred:
        for arg in pred['args']:
            if is_instance_name(arg):
                objects.add(arg)
    
    # Check arg1/arg2 for atomic predicates
    if 'arg1' in pred:
        if is_instance_name(pred['arg1']):
            objects.add(pred['arg1'])
    if 'arg2' in pred:
        if is_instance_name(pred['arg2']):
            objects.add(pred['arg2'])
    
    # Check category field (for forall/exists)
    if 'category' in pred:
        # Category is a template like "can__of__soda.n.01", not an instance
        pass  # Instances are already captured above
    
    # Recursively check nested predicates
    if 'children' in pred:
        for child in pred['children']:
            objects.update(extract_objects_from_predicate(child))
    
    if 'body' in pred and isinstance(pred['body'], dict):
        objects.update(extract_objects_from_predicate(pred['body']))
    
    # Check nested in instances
    if 'instances' in pred:
        for inst_data in pred['instances'].values():
            if isinstance(inst_data, dict) and 'nested' in inst_data:
                objects.update(extract_objects_from_predicate(inst_data['nested']))
    
    return objects


def extract_predicate_structure(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured information about a predicate for visibility computation.
    
    For quantified predicates (forall/exists), separates:
    - quantified_objects: The objects being quantified over (e.g., all cans)
    - target_objects: The fixed target objects (e.g., the trashcan)
    
    For simple predicates, all objects are treated as required.
    
    Args:
        pred: Predicate dict from JSONL
        
    Returns:
        Dict with:
        - type: 'forall', 'exists', 'atomic', etc.
        - quantified_objects: Set of BDDL object names being quantified
        - target_objects: Set of BDDL object names that are fixed targets
        - all_objects: Union of above (for backward compatibility)
    """
    pred_type = pred.get('type', 'atomic')
    quantified_objects = set()
    target_objects = set()
    
    if pred_type in ('forall', 'exists', 'forpairs'):
        # Quantified predicate: instances are quantified, args contain targets
        if 'instances' in pred:
            for inst_name in pred['instances'].keys():
                if is_instance_name(inst_name):
                    quantified_objects.add(inst_name)
        
        # Args contain the target objects (not quantified) - only if they're actual instances
        if 'args' in pred:
            for arg in pred['args']:
                if is_instance_name(arg):
                    target_objects.add(arg)
        # Also check arg1/arg2 for target objects
        if 'arg1' in pred and is_instance_name(pred['arg1']):
            target_objects.add(pred['arg1'])
        if 'arg2' in pred and is_instance_name(pred['arg2']):
            target_objects.add(pred['arg2'])
    else:
        # Atomic/simple predicate: all objects from args are required
        if 'args' in pred:
            for arg in pred['args']:
                if is_instance_name(arg):
                    target_objects.add(arg)
        
        # Also check arg1/arg2 for atomic predicates
        if 'arg1' in pred and is_instance_name(pred['arg1']):
            target_objects.add(pred['arg1'])
        if 'arg2' in pred and is_instance_name(pred['arg2']):
            target_objects.add(pred['arg2'])
        
        # Check instances for atomic predicates too
        if 'instances' in pred:
            for inst_name in pred['instances'].keys():
                if is_instance_name(inst_name):
                    target_objects.add(inst_name)
    
    # Handle nested predicates (children, body)
    if 'children' in pred:
        for child in pred['children']:
            child_struct = extract_predicate_structure(child)
            target_objects.update(child_struct['target_objects'])
            quantified_objects.update(child_struct['quantified_objects'])
    
    if 'body' in pred and isinstance(pred['body'], dict):
        body_struct = extract_predicate_structure(pred['body'])
        target_objects.update(body_struct['target_objects'])
        quantified_objects.update(body_struct['quantified_objects'])
    
    # Also check nested in instances (for exists/forall with nested predicates)
    if 'instances' in pred:
        for inst_data in pred['instances'].values():
            if isinstance(inst_data, dict) and 'nested' in inst_data:
                nested_struct = extract_predicate_structure(inst_data['nested'])
                target_objects.update(nested_struct['target_objects'])
                quantified_objects.update(nested_struct['quantified_objects'])
    
    return {
        'type': pred_type,
        'quantified_objects': quantified_objects,
        'target_objects': target_objects,
        'all_objects': quantified_objects | target_objects,
        'threshold': pred.get('threshold', 1),
    }


def generate_yuv_palette(num_ids: int) -> np.ndarray:
    """Generate equidistant YUV colors (same as encoding)."""
    Y_vals = np.linspace(16, 235, int(np.ceil(num_ids ** (1 / 3))))
    U_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1 / 3))))
    V_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1 / 3))))
    palette = []
    for y in Y_vals:
        for u in U_vals:
            for v in V_vals:
                palette.append([y, u, v])
                if len(palette) >= num_ids:
                    return np.array(palette, dtype=np.uint8)
    return np.array(palette[:num_ids], dtype=np.uint8)


def decode_seg_frame(frame_rgb: np.ndarray, id_list: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Decode RGB frame to instance IDs using nearest palette match (CPU version)."""
    frame_flat = frame_rgb.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    # Use scipy for faster distance computation if available
    try:
        from scipy.spatial.distance import cdist
        distances = cdist(frame_flat, palette_f, metric='euclidean')
    except ImportError:
        distances = np.sqrt(np.sum((frame_flat[:, None, :] - palette_f[None, :, :]) ** 2, axis=2))
    indices = np.argmin(distances, axis=1)
    return id_list[indices].reshape(frame_rgb.shape[:2])


def decode_seg_frame_gpu(frame_rgb: np.ndarray, id_list_th: 'th.Tensor', palette_th: 'th.Tensor') -> set:
    """Decode RGB frame to instance IDs using GPU (returns set of visible IDs)."""
    rgb = th.from_numpy(frame_rgb).float().cuda()  # (H, W, 3)
    rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
    # Use cdist for efficient pairwise distance
    distances = th.cdist(rgb_flat[None, :, :], palette_th[None, :, :], p=2)[0]  # (H*W, N_ids)
    indices = th.argmin(distances, dim=-1)  # (H*W,)
    instance_ids = id_list_th[indices]
    return set(th.unique(instance_ids).cpu().tolist())


def decode_seg_frame_gpu_with_counts(frame_rgb: np.ndarray, id_list_th: 'th.Tensor', palette_th: 'th.Tensor') -> Dict[int, int]:
    """Decode RGB frame to instance IDs using GPU (returns dict of ID -> pixel count)."""
    rgb = th.from_numpy(frame_rgb).float().cuda()  # (H, W, 3)
    rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
    # Use cdist for efficient pairwise distance
    distances = th.cdist(rgb_flat[None, :, :], palette_th[None, :, :], p=2)[0]  # (H*W, N_ids)
    indices = th.argmin(distances, dim=-1)  # (H*W,)
    instance_ids = id_list_th[indices]
    
    # Count pixels per instance ID
    unique_ids, counts = th.unique(instance_ids, return_counts=True)
    return {int(uid): int(cnt) for uid, cnt in zip(unique_ids.cpu().tolist(), counts.cpu().tolist())}


def decode_seg_frame_gpu_full(frame_rgb: np.ndarray, id_list_th: 'th.Tensor', palette_th: 'th.Tensor') -> np.ndarray:
    """Decode RGB frame to instance ID map using GPU (returns full H x W array)."""
    rgb = th.from_numpy(frame_rgb).float().cuda()  # (H, W, 3)
    H, W = rgb.shape[:2]
    rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
    # Use cdist for efficient pairwise distance
    distances = th.cdist(rgb_flat[None, :, :], palette_th[None, :, :], p=2)[0]  # (H*W, N_ids)
    indices = th.argmin(distances, dim=-1)  # (H*W,)
    instance_ids = id_list_th[indices]
    return instance_ids.reshape(H, W).cpu().numpy()


def save_masked_frame_visualization(
    rgb_frame: np.ndarray,
    seg_frame: np.ndarray,
    id_list: np.ndarray,
    palette: np.ndarray,
    task_relevant_ids: Set[int],
    id_to_scene_obj: Dict[int, str],
    output_path: Path,
    timestep: int,
    id_list_th: Optional['th.Tensor'] = None,
    palette_th: Optional['th.Tensor'] = None,
    use_gpu: bool = False
):
    """
    Save a visualization of an RGB frame with task-relevant objects masked.
    
    Args:
        rgb_frame: RGB frame from the video
        seg_frame: Segmentation frame (encoded YUV colors)
        id_list: Array of instance IDs
        palette: YUV palette for decoding
        task_relevant_ids: Set of instance IDs for task-relevant objects
        id_to_scene_obj: Mapping from instance ID to scene object name
        output_path: Path to save the visualization
        timestep: Current timestep for labeling
        id_list_th: GPU tensor of ID list (optional)
        palette_th: GPU tensor of palette (optional)
        use_gpu: Whether to use GPU for decoding
    """
    # Decode segmentation frame to instance IDs
    if use_gpu and id_list_th is not None and palette_th is not None:
        instance_map = decode_seg_frame_gpu_full(seg_frame, id_list_th, palette_th)
    else:
        instance_map = decode_seg_frame(seg_frame, id_list, palette)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original RGB frame
    axes[0].imshow(rgb_frame)
    axes[0].set_title(f'RGB Frame (timestep {timestep})')
    axes[0].axis('off')
    
    # 2. RGB with colored mask overlay
    overlay = rgb_frame.copy().astype(np.float32)
    
    # Define distinct colors for each object
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
    ]
    
    legend_patches = []
    
    # Group instance IDs by scene object name so all parts of the same object get the same color
    scene_obj_to_ids_local = {}
    for inst_id in task_relevant_ids:
        obj_name = id_to_scene_obj.get(inst_id, f"id_{inst_id}")
        if obj_name not in scene_obj_to_ids_local:
            scene_obj_to_ids_local[obj_name] = []
        scene_obj_to_ids_local[obj_name].append(inst_id)
    
    color_idx = 0
    for obj_name in sorted(scene_obj_to_ids_local.keys()):
        obj_ids = scene_obj_to_ids_local[obj_name]
        # Create combined mask for all parts of this object
        combined_mask = np.zeros(instance_map.shape, dtype=bool)
        for inst_id in obj_ids:
            combined_mask |= (instance_map == inst_id)
        
        if combined_mask.any():
            color = colors[color_idx % len(colors)]
            
            # Blend color with original image
            for c in range(3):
                overlay[:, :, c] = np.where(combined_mask, 
                    0.5 * overlay[:, :, c] + 0.5 * color[c], 
                    overlay[:, :, c])
            
            # Add to legend
            legend_patches.append(Patch(
                facecolor=np.array(color) / 255.0,
                label=f'{obj_name} ({np.sum(combined_mask)} px)'
            ))
            color_idx += 1
    
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title('RGB + Mask Overlay')
    axes[1].axis('off')
    if legend_patches:
        axes[1].legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    # 3. Mask only (black background with colored objects)
    mask_only = np.zeros_like(rgb_frame)
    color_idx = 0
    
    for obj_name in sorted(scene_obj_to_ids_local.keys()):
        obj_ids = scene_obj_to_ids_local[obj_name]
        combined_mask = np.zeros(instance_map.shape, dtype=bool)
        for inst_id in obj_ids:
            combined_mask |= (instance_map == inst_id)
        
        if combined_mask.any():
            color = colors[color_idx % len(colors)]
            for c in range(3):
                mask_only[:, :, c] = np.where(combined_mask, color[c], mask_only[:, :, c])
            color_idx += 1
    
    axes[2].imshow(mask_only)
    axes[2].set_title('Task-Relevant Objects Only')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_scene_object_to_ids(ins_id_mapping: Dict[str, str]) -> Dict[str, List[int]]:
    """Build mapping from scene object name to instance IDs."""
    scene_obj_to_ids = {}
    for ins_id, path in ins_id_mapping.items():
        match = re.search(r'/World/scene_0/([^/]+)/', path)
        if match:
            obj_name = match.group(1)
            if obj_name not in scene_obj_to_ids:
                scene_obj_to_ids[obj_name] = []
            scene_obj_to_ids[obj_name].append(int(ins_id))
    return scene_obj_to_ids


def compute_visibility_for_timesteps(
    task_id: int,
    episode_id: str,
    timesteps: List[int],
    predicate_structures: List[Dict[str, Any]],
    b1k_data_path: str = "/vast/projects/kumar/lab/yishao/data/b1k_full"
) -> Tuple[np.ndarray, bool]:
    """
    Compute visibility ratio for each predicate at each timestep.
    
    For quantified predicates (forall/exists):
    - visibility = (visible_quantified / total_quantified) * (1 if all targets visible else 0)
    
    For simple predicates:
    - visibility = 1 if all objects visible, 0 otherwise
    
    Uses GPU acceleration when available for ~10x speedup.
    Only loads frames at the requested timesteps to save memory.
    
    Args:
        task_id: Task ID number
        episode_id: Episode ID string (e.g., "00010010")
        timesteps: List of timesteps to compute visibility for
        predicate_structures: List of dicts from extract_predicate_structure()
        b1k_data_path: Path to b1k_full data directory
        
    Returns:
        - visibility: np.ndarray of shape (num_timesteps, num_predicates)
        - success: Boolean indicating if visibility computation succeeded
    """
    num_timesteps = len(timesteps)
    num_predicates = len(predicate_structures)
    visibility = np.zeros((num_timesteps, num_predicates), dtype=np.float32)
    
    use_gpu = HAS_CUDA
    if use_gpu:
        print(f"    Using GPU acceleration for visibility computation")
    
    # Load episode metadata
    meta_path = Path(b1k_data_path) / 'meta' / 'episodes' / f'task-{task_id:04d}' / f'episode_{episode_id}.json'
    if not meta_path.exists():
        print(f"    Warning: Metadata not found at {meta_path}")
        return visibility, False
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Get instance ID mapping
    ins_id_mapping = json.loads(meta.get('ins_id_mapping', '{}'))
    scene_obj_to_ids = get_scene_object_to_ids(ins_id_mapping)
    
    # Generate per-camera id_list and palette (each camera's video uses its own encoding)
    # The palette depends on the NUMBER of IDs, so each camera needs its own
    camera_id_lists = {}
    camera_palettes = {}
    camera_id_lists_th = {}
    camera_palettes_th = {}
    
    for camera_name, key in CAMERA_KEYS.items():
        ids = meta.get(key, [])
        if isinstance(ids, str):
            ids = json.loads(ids)
        if ids:
            camera_id_list = np.array(sorted(ids))
            camera_palette = generate_yuv_palette(len(camera_id_list))
            camera_id_lists[camera_name] = camera_id_list
            camera_palettes[camera_name] = camera_palette
            if use_gpu:
                camera_id_lists_th[camera_name] = th.from_numpy(camera_id_list).long().cuda()
                camera_palettes_th[camera_name] = th.from_numpy(camera_palette).float().cuda()
            print(f"    {camera_name} camera: {len(camera_id_list)} unique IDs")
    
    if not camera_id_lists:
        print(f"    Warning: No unique IDs found in metadata")
        return visibility, False
    
    # For compatibility, set id_list/palette to head camera (main camera)
    id_list = camera_id_lists.get('head', list(camera_id_lists.values())[0])
    palette = camera_palettes.get('head', list(camera_palettes.values())[0])
    if use_gpu:
        id_list_th = camera_id_lists_th.get('head', list(camera_id_lists_th.values())[0])
        palette_th = camera_palettes_th.get('head', list(camera_palettes_th.values())[0])
    
    # Load BDDL instance to scene name mapping from HDF5 (ground truth from simulator)
    rawdata_path = Path(b1k_data_path).parent / 'behavior_rawdata'
    inst_to_name = load_inst_to_name_from_hdf5(task_id, episode_id, str(rawdata_path))
    
    if inst_to_name:
        print(f"    Loaded inst_to_name mapping from HDF5 ({len(inst_to_name)} entries)")
        for bddl, scene in inst_to_name.items():
            print(f"      {bddl} -> {scene}")
    else:
        print(f"    Warning: No inst_to_name mapping found, falling back to category-based matching")
    
    # Debug: Print scene_obj_to_ids for multi-part objects
    print(f"    scene_obj_to_ids (multi-part objects):")
    for obj_name, ids in sorted(scene_obj_to_ids.items()):
        if len(ids) > 1:
            print(f"      {obj_name}: {sorted(ids)}")
    
    # Build BDDL object -> scene instance IDs mapping
    bddl_to_scene_ids = {}
    bddl_to_scene_name = {}  # Also store the scene object name for each BDDL object
    
    all_bddl_objects = set()
    for pred_struct in predicate_structures:
        all_bddl_objects.update(pred_struct['all_objects'])
    
    for bddl_name in all_bddl_objects:
        scene_name = inst_to_name.get(bddl_name)
        
        if scene_name and scene_name in scene_obj_to_ids:
            # Use ground truth mapping from HDF5
            bddl_to_scene_ids[bddl_name] = set(scene_obj_to_ids[scene_name])
            bddl_to_scene_name[bddl_name] = scene_name
        else:
            # Fallback to category-based matching
            bddl_category = parse_bddl_category(bddl_name)
            scene_category = get_scene_category(bddl_category)
            
            matching_ids = []
            matching_name = None
            for obj_name, ids in scene_obj_to_ids.items():
                if obj_name.startswith(scene_category + '_') or obj_name == scene_category:
                    matching_ids.extend(ids)
                    if matching_name is None:
                        matching_name = obj_name
            
            bddl_to_scene_ids[bddl_name] = set(matching_ids)
            bddl_to_scene_name[bddl_name] = matching_name
            
            if not scene_name:
                print(f"    Warning: No inst_to_name for '{bddl_name}', using category fallback")
    
    # Debug: Print bddl_to_scene_ids mapping
    print(f"    bddl_to_scene_ids mapping:")
    for bddl_name, scene_ids in sorted(bddl_to_scene_ids.items()):
        scene_name = bddl_to_scene_name.get(bddl_name, "?")
        print(f"      {bddl_name} -> {scene_name}: {sorted(scene_ids)}")
    
    # Collect all task-relevant instance IDs for visualization
    all_task_relevant_ids = set()
    for pred_struct in predicate_structures:
        for bddl_name in pred_struct['all_objects']:
            all_task_relevant_ids.update(bddl_to_scene_ids.get(bddl_name, set()))
    
    print(f"    all_task_relevant_ids ({len(all_task_relevant_ids)}): {sorted(all_task_relevant_ids)}")
    
    # Open video containers for cameras (only head camera for now)
    cameras = ['head']  # TODO: add 'left_wrist', 'right_wrist' back if needed
    seg_video_containers = {}
    seg_video_streams = {}
    rgb_video_containers = {}
    rgb_video_streams = {}
    
    for camera in cameras:
        # Segmentation video
        seg_video_path = Path(b1k_data_path) / 'videos' / f'task-{task_id:04d}' / f'observation.images.seg_instance_id.{camera}' / f'episode_{episode_id}.mp4'
        if seg_video_path.exists():
            try:
                container = av.open(str(seg_video_path))
                seg_video_containers[camera] = container
                seg_video_streams[camera] = container.streams.video[0]
            except Exception as e:
                print(f"    Warning: Failed to open seg video {seg_video_path}: {e}")
        
        # RGB video
        rgb_video_path = Path(b1k_data_path) / 'videos' / f'task-{task_id:04d}' / f'observation.images.rgb.{camera}' / f'episode_{episode_id}.mp4'
        if rgb_video_path.exists():
            try:
                container = av.open(str(rgb_video_path))
                rgb_video_containers[camera] = container
                rgb_video_streams[camera] = container.streams.video[0]
            except Exception as e:
                print(f"    Warning: Failed to open RGB video {rgb_video_path}: {e}")
    
    if not seg_video_containers:
        print(f"    Warning: No segmentation videos loaded")
        return visibility, False
    
    # Get frame rate and duration info
    sample_stream = list(seg_video_streams.values())[0]
    fps = float(sample_stream.average_rate)
    total_frames = sample_stream.frames
    
    # Pre-decode only the frames we need (using seek for efficiency)
    timestep_set = set(timesteps)
    max_timestep = max(timesteps)
    
    # Collect segmentation frames for each camera at needed timesteps
    seg_camera_frames = {camera: {} for camera in seg_video_containers.keys()}
    
    for camera, container in seg_video_containers.items():
        stream = seg_video_streams[camera]
        frame_idx = 0
        
        # Decode frames sequentially (seeking is often slower for close frames)
        for frame in container.decode(video=0):
            if frame_idx in timestep_set:
                seg_camera_frames[camera][frame_idx] = frame.to_ndarray(format='rgb24')
            frame_idx += 1
            if frame_idx > max_timestep:
                break
        
        container.close()
    
    # Collect RGB frames for visualization (every ~200 video frames = every 2-3 predicate timesteps since they're sampled at 90 frames)
    # Generate viz at indices 0, 2, 4, 6, ... (every 2nd predicate timestep) to get ~180 frame spacing
    rgb_camera_frames = {camera: {} for camera in rgb_video_containers.keys()}
    VIZ_PRED_INTERVAL = 2  # Every 2nd predicate timestep (~180 video frames apart)
    viz_indices = list(range(0, len(timesteps), VIZ_PRED_INTERVAL))
    if len(timesteps) - 1 not in viz_indices:
        viz_indices.append(len(timesteps) - 1)  # Always include last
    viz_timesteps = [timesteps[i] for i in viz_indices]
    viz_timestep_set = set(viz_timesteps)
    
    # Minimum pixel threshold for visibility
    MIN_VISIBLE_PIXELS = 500
    
    for camera, container in rgb_video_containers.items():
        stream = rgb_video_streams[camera]
        frame_idx = 0
        
        for frame in container.decode(video=0):
            if frame_idx in viz_timestep_set:
                rgb_camera_frames[camera][frame_idx] = frame.to_ndarray(format='rgb24')
            frame_idx += 1
            if frame_idx > max(viz_timesteps):
                break
        
        container.close()
    
    # Build reverse mapping: instance_id -> scene object name
    id_to_scene_obj = {}
    for obj_name, ids in scene_obj_to_ids.items():
        for inst_id in ids:
            id_to_scene_obj[inst_id] = obj_name
    
    # Create output directory for frame visualizations
    viz_output_dir = Path(b1k_data_path).parent / 'predicate_data' / 'predicate_vectors' / 'frame_viz' / f'task-{task_id:04d}'
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each timestep, compute visibility
    for t_idx, timestep in enumerate(timesteps):
        # Get visible instance IDs with pixel counts across all cameras for this frame
        visible_ids = set()
        id_pixel_counts = {}  # instance_id -> pixel count
        
        for camera, frames_dict in seg_camera_frames.items():
            if timestep in frames_dict:
                frame_rgb = frames_dict[timestep]
                # Use camera-specific id_list and palette
                cam_id_list = camera_id_lists.get(camera, id_list)
                cam_palette = camera_palettes.get(camera, palette)
                if use_gpu:
                    cam_id_list_th = camera_id_lists_th.get(camera, id_list_th)
                    cam_palette_th = camera_palettes_th.get(camera, palette_th)
                    counts = decode_seg_frame_gpu_with_counts(frame_rgb, cam_id_list_th, cam_palette_th)
                    visible_ids.update(counts.keys())
                    for inst_id, cnt in counts.items():
                        id_pixel_counts[inst_id] = id_pixel_counts.get(inst_id, 0) + cnt
                else:
                    decoded_ids = decode_seg_frame(frame_rgb, cam_id_list, cam_palette)
                    unique, counts = np.unique(decoded_ids, return_counts=True)
                    visible_ids.update(unique.tolist())
                    for inst_id, cnt in zip(unique.tolist(), counts.tolist()):
                        id_pixel_counts[inst_id] = id_pixel_counts.get(inst_id, 0) + cnt
        
        # Apply minimum pixel threshold - filter visible_ids to only include objects with >= MIN_VISIBLE_PIXELS
        visible_ids_filtered = {inst_id for inst_id in visible_ids 
                                if id_pixel_counts.get(inst_id, 0) >= MIN_VISIBLE_PIXELS}
        
        # Print pixel counts for visualization timesteps (debug)
        # Group by scene object name (not individual instance IDs)
        if timestep in viz_timestep_set:
            print(f"    Timestep {timestep}: task-relevant objects (min {MIN_VISIBLE_PIXELS}px):")
            
            # Group IDs by scene object name
            obj_to_ids = {}
            for inst_id in all_task_relevant_ids:
                obj_name = id_to_scene_obj.get(inst_id, f"unknown_{inst_id}")
                if obj_name not in obj_to_ids:
                    obj_to_ids[obj_name] = []
                obj_to_ids[obj_name].append(inst_id)
            
            for obj_name in sorted(obj_to_ids.keys()):
                obj_ids = obj_to_ids[obj_name]
                total_px = sum(id_pixel_counts.get(inst_id, 0) for inst_id in obj_ids)
                any_visible = any(inst_id in visible_ids_filtered for inst_id in obj_ids)
                
                if total_px >= MIN_VISIBLE_PIXELS:
                    status = f"{total_px} px"
                elif total_px > 0:
                    status = f"{total_px} px (below threshold)"
                else:
                    status = "NOT VISIBLE"
                print(f"      {obj_name}: {status}")
            
            # Save frame visualization
            if 'head' in rgb_camera_frames and timestep in rgb_camera_frames['head']:
                rgb_frame = rgb_camera_frames['head'][timestep]
                seg_frame = seg_camera_frames['head'][timestep]
                
                # Use head camera's id_list and palette
                head_id_list = camera_id_lists.get('head', id_list)
                head_palette = camera_palettes.get('head', palette)
                head_id_list_th = camera_id_lists_th.get('head') if use_gpu else None
                head_palette_th = camera_palettes_th.get('head') if use_gpu else None
                
                viz_path = viz_output_dir / f'episode_{episode_id}_timestep_{timestep:05d}.png'
                save_masked_frame_visualization(
                    rgb_frame=rgb_frame,
                    seg_frame=seg_frame,
                    id_list=head_id_list,
                    palette=head_palette,
                    task_relevant_ids=all_task_relevant_ids,
                    id_to_scene_obj=id_to_scene_obj,
                    output_path=viz_path,
                    timestep=timestep,
                    id_list_th=head_id_list_th,
                    palette_th=head_palette_th,
                    use_gpu=use_gpu
                )
                print(f"      Saved visualization to {viz_path}")
        
        # Compute visibility ratio for each predicate (using filtered IDs with pixel threshold)
        for p_idx, pred_struct in enumerate(predicate_structures):
            pred_type = pred_struct['type']
            quantified_objects = pred_struct['quantified_objects']
            target_objects = pred_struct['target_objects']
            
            if not quantified_objects and not target_objects:
                visibility[t_idx, p_idx] = 1.0  # No objects = fully visible
                continue
            
            if pred_type in ('forall', 'exists', 'forpairs'):
                # Quantified predicate: 
                # visibility = (visible_quantified_instances / total_quantified_instances) * (1 if target visible else 0)
                
                # First check if target objects are visible (required) - using filtered IDs
                target_visible = True
                for bddl_name in target_objects:
                    scene_ids = bddl_to_scene_ids.get(bddl_name, set())
                    if not (scene_ids & visible_ids_filtered):
                        target_visible = False
                        break
                
                if not target_visible:
                    visibility[t_idx, p_idx] = 0.0
                    continue
                
                # Count visible quantified instances
                if not quantified_objects:
                    visibility[t_idx, p_idx] = 1.0
                    continue
                
                # Count how many quantified BDDL objects are visible
                total_visible = 0
                total_expected = len(quantified_objects)
                
                for bddl_name in quantified_objects:
                    scene_ids = bddl_to_scene_ids.get(bddl_name, set())
                    if scene_ids & visible_ids_filtered:
                        total_visible += 1
                
                if total_expected > 0:
                    visibility[t_idx, p_idx] = total_visible / total_expected
                else:
                    visibility[t_idx, p_idx] = 1.0
                    
            else:
                # Simple/atomic predicate: all objects must be visible (using filtered IDs)
                all_visible = True
                for bddl_name in target_objects:
                    scene_ids = bddl_to_scene_ids.get(bddl_name, set())
                    if not (scene_ids & visible_ids_filtered):
                        all_visible = False
                        break
                
                visibility[t_idx, p_idx] = 1.0 if all_visible else 0.0
    
    return visibility, True


# =============================================================================
# PREDICATE VALUE COMPUTATION
# =============================================================================


def compute_predicate_value(pred: Dict[str, Any]) -> float:
    """
    Compute normalized value for a predicate (0.0 to 1.0).
    
    - Binary predicates: 0.0 or 1.0
    - Count predicates: count / threshold
    
    Args:
        pred: Predicate dict from JSONL
        
    Returns:
        Float value between 0.0 and 1.0
    """
    pred_type = pred.get("type", "unknown")
    
    if pred_type == "atomic":
        # Binary: check if the predicate is satisfied
        # For atomic predicates, we need to check _satisfied or infer from context
        # In the data, atomic predicates at top level use the satisfied field
        if "satisfied" in pred:
            return 1.0 if pred["satisfied"] else 0.0
        elif "_satisfied" in pred:
            return 1.0 if pred["_satisfied"] else 0.0
        else:
            # Atomic predicates don't always have satisfied - check parent context
            # For now, return 0.0 as default (will be overridden by parent)
            return 0.0
    
    elif pred_type == "forall":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "exists":
        # Exists: requires at least 1 instance to satisfy the nested condition
        # For exists, we take the MAX over instances (best instance determines progress)
        # This is semantically correct: exists only needs one to succeed
        
        instances = pred.get("instances", {})
        if instances:
            # Check first instance for nested structure
            first_inst = next(iter(instances.values()), {})
            if isinstance(first_inst, dict) and "nested" in first_inst:
                nested = first_inst["nested"]
                nested_type = nested.get("type", "")
                
                # If nested is a quantifier, take MAX ratio across instances
                if nested_type in ["forall", "forn"]:
                    # For each instance, compute its ratio, then take max
                    max_ratio = 0.0
                    for inst_data in instances.values():
                        if isinstance(inst_data, dict) and "nested" in inst_data:
                            n = inst_data["nested"]
                            inst_count = n.get("count", 0)
                            inst_threshold = n.get("threshold", 1)
                            if inst_threshold > 0:
                                inst_ratio = min(inst_count / inst_threshold, 1.0)
                                max_ratio = max(max_ratio, inst_ratio)
                            elif n.get("satisfied", False):
                                max_ratio = 1.0
                    return max_ratio
                elif nested_type in ["and", "or"]:
                    # For logical operators, recursively compute and take max
                    max_ratio = 0.0
                    for inst_data in instances.values():
                        if isinstance(inst_data, dict) and "nested" in inst_data:
                            nested_value = compute_predicate_value(inst_data["nested"])
                            max_ratio = max(max_ratio, nested_value)
                    return max_ratio
                elif nested_type == "atomic":
                    # Count satisfied instances / threshold (for exists, threshold is typically 1)
                    satisfied_count = sum(
                        1 for inst_data in instances.values()
                        if isinstance(inst_data, dict) and inst_data.get("satisfied", False)
                    )
                    threshold = pred.get("threshold", 1)
                    if threshold == 0:
                        return 1.0 if pred.get("satisfied", False) else 0.0
                    return min(satisfied_count / threshold, 1.0)
        
        # Default: use count/threshold (for simple exists without nested structure)
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "not":
        # Binary: satisfied means the negation holds
        return 1.0 if pred.get("satisfied", False) else 0.0
    
    elif pred_type == "forpairs":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "fornpairs":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "forn":
        # Count-based: count / threshold
        count = pred.get("count", 0)
        threshold = pred.get("threshold", 1)
        if threshold == 0:
            return 1.0 if pred.get("satisfied", False) else 0.0
        return min(count / threshold, 1.0)
    
    elif pred_type == "or":
        # Or: at least one child satisfied
        # Use ratio: max of child values (best child determines progress)
        children = pred.get("children", [])
        if children:
            max_value = 0.0
            for child in children:
                child_value = compute_predicate_value(child)
                max_value = max(max_value, child_value)
            return max_value
        return 1.0 if pred.get("satisfied", False) else 0.0
    
    elif pred_type == "and":
        # And: all children must be satisfied
        # Use ratio: count of satisfied children / total children
        children = pred.get("children", [])
        if children:
            satisfied_count = sum(1 for child in children if child.get("satisfied", False))
            return satisfied_count / len(children)
        return 1.0 if pred.get("satisfied", False) else 0.0
    
    else:
        # Unknown type - use satisfied field if available
        if "satisfied" in pred:
            return 1.0 if pred["satisfied"] else 0.0
        elif "_satisfied" in pred:
            return 1.0 if pred["_satisfied"] else 0.0
        return 0.0


def clean_object_name(name: str) -> str:
    """
    Clean up object names by removing instance suffixes and formatting.
    
    Examples:
        "can__of__soda.n.01" -> "can of soda"
        "ashcan.n.01_1" -> "ashcan"
        "pizza.n.01" -> "pizza"
    """
    import re
    # Remove instance suffix like "_1", "_2" at the end
    name = re.sub(r'_\d+$', '', name)
    # Remove WordNet suffix like ".n.01", ".v.02"
    name = re.sub(r'\.[a-z]\.\d+$', '', name)
    # Replace double underscores with spaces
    name = name.replace('__', ' ')
    # Replace remaining underscores with spaces
    name = name.replace('_', ' ')
    return name


def get_predicate_name(pred: Dict[str, Any], idx: int) -> str:
    """
    Generate a human-readable name for a predicate.
    
    Args:
        pred: Predicate dict
        idx: Predicate index
        
    Returns:
        Short descriptive name
    """
    pred_type = pred.get("type", "unknown")
    raw_desc = pred.get("_raw_description", "")
    
    if pred_type == "atomic":
        pred_name = pred.get("predicate", "?")
        arg1 = clean_object_name(pred.get("arg1", ""))
        return f"{pred_name}({arg1})"
    
    elif pred_type == "forall":
        category = clean_object_name(pred.get("category", "?"))
        inner_pred = pred.get("predicate", "")
        args = pred.get("args", [])
        
        # Check if inner_pred is "not" - need to look at nested structure
        if inner_pred == "not":
            # Look at instances to find the actual atomic predicate
            instances = pred.get("instances", {})
            if instances:
                first_inst = next(iter(instances.values()), {})
                if isinstance(first_inst, dict) and "nested" in first_inst:
                    nested = first_inst["nested"]
                    if nested.get("type") == "not" and "child" in nested:
                        child = nested["child"]
                        actual_pred = child.get("predicate", "")
                        if actual_pred:
                            return f"forall({category})->not({actual_pred})"
            return f"forall({category})->not"
        
        # Check if inner_pred is "exists" - need to look at nested structure for actual predicate
        if inner_pred == "exists":
            instances = pred.get("instances", {})
            if instances:
                first_inst = next(iter(instances.values()), {})
                if isinstance(first_inst, dict) and "nested" in first_inst:
                    nested = first_inst["nested"]
                    if nested.get("type") == "exists":
                        nested_cat = clean_object_name(nested.get("category", ""))
                        # Go one level deeper to find atomic predicate
                        nested_insts = nested.get("instances", {})
                        if nested_insts:
                            inner_inst = next(iter(nested_insts.values()), {})
                            if isinstance(inner_inst, dict) and "nested" in inner_inst:
                                atomic = inner_inst["nested"]
                                if atomic.get("type") == "atomic":
                                    atomic_pred = atomic.get("predicate", "")
                                    return f"forall({category})->exists({nested_cat})->{atomic_pred}"
            return f"forall({category})->exists"
        
        if inner_pred and len(args) >= 2:
            target = clean_object_name(args[1])
            return f"forall({category})->{inner_pred}({target})"
        elif inner_pred:
            return f"forall({category})->{inner_pred}"
        return f"forall({category})"
    
    elif pred_type == "exists":
        category = clean_object_name(pred.get("category", "?"))
        # Check for nested predicate
        instances = pred.get("instances", {})
        if instances:
            first_inst = next(iter(instances.values()), {})
            if isinstance(first_inst, dict) and "nested" in first_inst:
                nested = first_inst["nested"]
                nested_type = nested.get("type", "")
                
                if nested_type == "atomic":
                    # Direct atomic predicate
                    nested_pred = nested.get("predicate", "")
                    arg1 = clean_object_name(nested.get("arg1", ""))
                    arg2 = clean_object_name(nested.get("arg2", ""))
                    if arg2:
                        return f"exists({category})->{nested_pred}({arg1},{arg2})"
                    elif arg1:
                        return f"exists({category})->{nested_pred}({arg1})"
                    return f"exists({category})->{nested_pred}"
                    
                elif nested_type in ["forall", "forn"]:
                    # Nested forall quantifier - e.g., exists(fridge)->forall(pizza)->inside
                    nested_cat = clean_object_name(nested.get("category", ""))
                    nested_pred = nested.get("predicate", "")
                    nested_args = nested.get("args", [])
                    if nested_pred and len(nested_args) >= 2:
                        target = clean_object_name(nested_args[1])
                        return f"exists({category})->forall({nested_cat})->{nested_pred}({target})"
                    elif nested_pred:
                        return f"exists({category})->forall({nested_cat})->{nested_pred}"
                    return f"exists({category})->forall({nested_cat})"
        return f"exists({category})"
    
    elif pred_type == "not":
        child = pred.get("child", {})
        if child.get("type") == "atomic":
            child_pred = child.get("predicate", "?")
            child_arg = clean_object_name(child.get("arg1", ""))
            return f"not({child_pred}({child_arg}))"
        return f"not(...)"
    
    elif pred_type == "forpairs":
        cat1 = clean_object_name(pred.get("category1", "?"))
        cat2 = clean_object_name(pred.get("category2", "?"))
        # Extract actual predicate from raw description
        raw_desc = pred.get("_raw_description", "")
        # e.g. "forpairs pizza.n.01 - pizza.n.01 plate.n.04 - plate.n.04 ontop pizza.n.01 plate.n.04"
        # The predicate is typically the word before the last two category mentions
        parts = raw_desc.split()
        pred_name = ""
        if len(parts) >= 3:
            # Look for predicate name (typically after the second "-")
            for i, p in enumerate(parts):
                if p in ["ontop", "inside", "under", "nextto", "touching", "onfloor"]:
                    pred_name = p
                    break
        if pred_name:
            return f"forpairs({cat1},{cat2})->{pred_name}"
        return f"forpairs({cat1},{cat2})"
    
    elif pred_type == "or":
        return f"or(...)"
    
    else:
        return f"pred_{idx}"


def is_binary_predicate(pred: Dict[str, Any]) -> bool:
    """Check if predicate produces binary output."""
    pred_type = pred.get("type", "unknown")
    return pred_type in ["atomic", "not", "or", "and"]


def load_episode_predicates(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Load all timestep records from a predicate JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of records, each containing step, predicates, etc.
    """
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_predicate_vectors_for_episode(
    records: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[str], List[int], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generate state vectors from episode records.
    
    Args:
        records: List of timestep records from JSONL
        
    Returns:
        - states: np.ndarray of shape (num_timesteps, num_predicates)
        - predicate_names: List of predicate names
        - timesteps: List of timestep indices
        - metadata: Dict with task info
        - predicate_structures: List of dicts from extract_predicate_structure()
    """
    if not records:
        return np.array([]), [], [], {}, []
    
    # Get predicate structure from first record
    first_record = records[0]
    predicates = first_record.get("predicates", [])
    num_predicates = len(predicates)
    
    # Generate predicate names
    predicate_names = [get_predicate_name(pred, i) for i, pred in enumerate(predicates)]
    
    # Extract structured info for each predicate (for visibility computation)
    predicate_structures = [extract_predicate_structure(pred) for pred in predicates]
    
    # Also extract flat object sets for backward compatibility
    predicate_objects = [extract_objects_from_predicate(pred) for pred in predicates]
    
    # Determine which are binary vs count-based
    is_binary = [is_binary_predicate(pred) for pred in predicates]
    
    # Extract timesteps and values
    timesteps = []
    state_vectors = []
    
    for record in records:
        step = record.get("step", 0)
        preds = record.get("predicates", [])
        
        # Compute value for each predicate
        values = []
        for pred in preds:
            val = compute_predicate_value(pred)
            values.append(val)
        
        # Pad if predicates changed (shouldn't happen, but safety)
        while len(values) < num_predicates:
            values.append(0.0)
        values = values[:num_predicates]
        
        timesteps.append(step)
        state_vectors.append(values)
    
    states = np.array(state_vectors, dtype=np.float32)
    
    # Metadata
    metadata = {
        "task_name": first_record.get("task_name", "unknown"),
        "episode_id": first_record.get("episode_id", 0),
        "num_predicates": num_predicates,
        "predicate_names": predicate_names,
        "is_binary": is_binary,
        "num_timesteps": len(timesteps),
        "timesteps": timesteps,
        "predicate_objects": [list(obj_set) for obj_set in predicate_objects],  # JSON-serializable
        "predicate_structures": [
            {
                'type': ps['type'],
                'quantified_objects': list(ps['quantified_objects']),
                'target_objects': list(ps['target_objects']),
                'threshold': ps['threshold'],
            }
            for ps in predicate_structures
        ],
    }
    
    return states, predicate_names, timesteps, metadata, predicate_structures


def visualize_predicate_vectors(
    states: np.ndarray,
    predicate_names: List[str],
    timesteps: List[int],
    metadata: Dict[str, Any],
    output_path: Path,
    task_id: Optional[int] = None,
    visibility: Optional[np.ndarray] = None
):
    """
    Visualize predicate vectors as a heatmap with optional visibility overlay.
    
    Args:
        states: np.ndarray of shape (num_timesteps, num_predicates)
        predicate_names: List of predicate names
        timesteps: List of timestep indices
        metadata: Dict with task info
        output_path: Path to save visualization
        task_id: Optional task ID for filename
        visibility: Optional np.ndarray of shape (num_timesteps, num_predicates) for visibility
    """
    if states.size == 0:
        print("No data to visualize")
        return
    
    num_timesteps, num_predicates = states.shape
    task_name = metadata.get("task_name", "unknown")
    episode_id = metadata.get("episode_id", 0)
    
    # Determine if we have visibility data
    has_visibility = visibility is not None and visibility.size > 0
    
    # Number of rows: predicates + visibility rows (interleaved)
    if has_visibility:
        total_rows = num_predicates * 2
    else:
        total_rows = num_predicates
    
    # Create figure
    fig_height = max(4, total_rows * 0.4 + 2)
    fig, ax = plt.subplots(1, 1, figsize=(16, fig_height))
    
    # Create colormaps
    # Green colormap for predicates (white -> dark green)
    green_colors = ['#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#006400']
    cmap_green = LinearSegmentedColormap.from_list('white_to_green', green_colors, N=256)
    
    # Yellow-brown colormap for visibility (white -> yellow -> brown)
    yellow_brown_colors = ['#FFFFFF', '#FFFACD', '#FFD700', '#DAA520', '#8B4513']
    cmap_yellow = LinearSegmentedColormap.from_list('white_to_brown', yellow_brown_colors, N=256)
    
    if has_visibility:
        # Interleave predicate values and visibility values
        combined_data = np.zeros((total_rows, num_timesteps), dtype=np.float32)
        row_labels = []
        row_types = []  # 'pred' or 'vis'
        
        for i in range(num_predicates):
            # Predicate row
            combined_data[i * 2] = states[:, i]
            row_labels.append(predicate_names[i][:40])
            row_types.append('pred')
            
            # Visibility row
            combined_data[i * 2 + 1] = visibility[:, i]
            row_labels.append(f"  └ visibility")
            row_types.append('vis')
        
        # Create two separate images and overlay them
        # First, create mask for each type
        pred_mask = np.array([t == 'pred' for t in row_types])
        vis_mask = np.array([t == 'vis' for t in row_types])
        
        # Create base image (all NaN for transparency)
        pred_data = np.where(pred_mask[:, None], combined_data, np.nan)
        vis_data = np.where(vis_mask[:, None], combined_data, np.nan)
        
        # Plot visibility first (background)
        im_vis = ax.imshow(vis_data, aspect='auto', cmap=cmap_yellow, vmin=0, vmax=1, interpolation='nearest')
        
        # Plot predicates on top
        im_pred = ax.imshow(pred_data, aspect='auto', cmap=cmap_green, vmin=0, vmax=1, interpolation='nearest')
        
        # Set y-axis labels
        ax.set_yticks(range(total_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        
        # Color the labels differently
        for i, (label, rtype) in enumerate(zip(ax.get_yticklabels(), row_types)):
            if rtype == 'vis':
                label.set_color('#8B4513')  # Brown color for visibility labels
                label.set_fontsize(7)
        
        # Add two colorbars
        cbar_pred = plt.colorbar(im_pred, ax=ax, orientation='vertical', pad=0.02, shrink=0.4, 
                                  anchor=(0, 1.0), aspect=20)
        cbar_pred.set_label('Predicate (green)', fontsize=9)
        
        cbar_vis = plt.colorbar(im_vis, ax=ax, orientation='vertical', pad=0.08, shrink=0.4,
                                 anchor=(0, 0.0), aspect=20)
        cbar_vis.set_label('Visibility (brown)', fontsize=9)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, num_timesteps, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, total_rows, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
    else:
        # Original visualization without visibility
        states_display = states.T
        im = ax.imshow(states_display, aspect='auto', cmap=cmap_green, vmin=0, vmax=1, interpolation='nearest')
        
        # Set y-axis labels
        short_names = [name[:40] for name in predicate_names]
        ax.set_yticks(range(num_predicates))
        ax.set_yticklabels(short_names, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Predicate Value (0=not satisfied, 1=satisfied)', fontsize=10)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, num_timesteps, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_predicates, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set x-axis (timesteps)
    num_ticks = min(10, num_timesteps)
    if num_timesteps > 1:
        tick_indices = np.linspace(0, num_timesteps - 1, num_ticks, dtype=int)
        tick_labels = [str(timesteps[i]) for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Predicate', fontsize=12)
    
    task_suffix = f"task{task_id:04d}_" if task_id is not None else ""
    title_suffix = " (with visibility)" if has_visibility else ""
    ax.set_title(f'{task_suffix}{task_name} (Episode {episode_id})\nPredicate State Vectors{title_suffix}', 
                 fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {output_path}")


def process_task(
    task_dir: Path,
    output_dir: Path,
    task_id: int,
    num_episodes: int = 1,
    viz_only_first: bool = True,
    compute_visibility: bool = True,
    b1k_data_path: str = "/vast/projects/kumar/lab/yishao/data/b1k_full",
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Process episodes for a task.
    
    Args:
        task_dir: Path to task-XXXX directory
        output_dir: Output directory for vectors
        task_id: Task ID number
        num_episodes: Number of episodes to process
        viz_only_first: Only visualize first episode
        compute_visibility: Whether to compute visibility data
        b1k_data_path: Path to b1k_full data for visibility computation
        
    Returns:
        Dict with processing results
    """
    # Find all JSONL files
    jsonl_files = sorted(task_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  No JSONL files found in {task_dir}")
        return {"status": "no_data", "task_id": task_id}
    
    # Limit to requested number of episodes
    files_to_process = jsonl_files[:num_episodes]
    
    task_name = None
    episodes_processed = []
    total_predicates = 0
    total_timesteps = 0
    
    # Create output directory
    task_output_dir = output_dir / f"task-{task_id:04d}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    skipped_count = 0
    for ep_idx, jsonl_file in enumerate(files_to_process):
        episode_name = jsonl_file.stem  # e.g., episode_00020010_predicates
        
        # Extract episode ID from filename (e.g., "episode_00020010_predicates" -> "00020010")
        episode_id_match = re.search(r'episode_(\d+)', episode_name)
        episode_id_str = episode_id_match.group(1) if episode_id_match else ""
        
        # Skip if output already exists
        npz_path = task_output_dir / f"{episode_name}_vectors.npz"
        if skip_existing and npz_path.exists():
            skipped_count += 1
            continue
        
        print(f"  Processing {jsonl_file.name}...")
        
        # Load and process
        records = load_episode_predicates(jsonl_file)
        if not records:
            print(f"  Empty file: {jsonl_file}")
            continue
        
        states, predicate_names, timesteps, metadata, predicate_structures = generate_predicate_vectors_for_episode(records)
        
        # Add task_id to metadata
        metadata["task_id"] = task_id
        metadata["source_file"] = str(jsonl_file)
        
        if task_name is None:
            task_name = metadata["task_name"]
        
        # Compute visibility if requested
        visibility = None
        if compute_visibility and episode_id_str:
            print(f"    Computing visibility for {len(timesteps)} timesteps...")
            visibility, vis_success = compute_visibility_for_timesteps(
                task_id=task_id,
                episode_id=episode_id_str,
                timesteps=timesteps,
                predicate_structures=predicate_structures,
                b1k_data_path=b1k_data_path
            )
            if vis_success:
                print(f"    Visibility computed successfully")
                metadata["has_visibility"] = True
            else:
                print(f"    Visibility computation failed, using empty visibility")
                visibility = None
                metadata["has_visibility"] = False
        else:
            metadata["has_visibility"] = False
        
        # Save as numpy arrays
        npz_path = task_output_dir / f"{episode_name}_vectors.npz"
        save_data = {
            'states': states,
            'timesteps': np.array(timesteps, dtype=np.int32),
            'predicate_names': np.array(predicate_names, dtype=object),
            'is_binary': np.array(metadata["is_binary"], dtype=bool),
        }
        if visibility is not None:
            save_data['visibility'] = visibility
        np.savez(npz_path, **save_data)
        print(f"    Saved vectors to {npz_path}")
        
        # Save metadata as JSON
        meta_path = task_output_dir / f"{episode_name}_metadata.json"
        # Convert numpy types for JSON serialization
        meta_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metadata.items()}
        with open(meta_path, 'w') as f:
            json.dump(meta_json, f, indent=2)
        
        # Visualize (first episode only, or all if requested)
        if viz_only_first:
            viz_dir = output_dir / "predicate_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            viz_path = viz_dir / f"predicate_vectors_task{task_id:04d}_{episode_name}.png"
            visualize_predicate_vectors(states, predicate_names, timesteps, metadata, viz_path, task_id, visibility)
        
        episodes_processed.append(episode_name)
        total_predicates = metadata["num_predicates"]
        total_timesteps += metadata["num_timesteps"]
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing episodes")
    
    if not episodes_processed:
        if skipped_count > 0:
            return {"status": "all_skipped", "task_id": task_id, "skipped": skipped_count}
        return {"status": "empty", "task_id": task_id}
    
    return {
        "status": "success",
        "task_id": task_id,
        "task_name": task_name,
        "num_predicates": total_predicates,
        "num_episodes": len(episodes_processed),
        "total_timesteps": total_timesteps,
        "episodes": episodes_processed,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate predicate state vectors from JSONL data")
    parser.add_argument("--data-dir", type=str,
                        default="/vast/projects/kumar/lab/yishao/data/predicate_data",
                        help="Directory containing task-XXXX subdirectories")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data_dir/predicate_vectors)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Process only this task ID (default: process first available)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Process all available tasks")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to process per task (default: 1)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--viz-only", action="store_true",
                        help="Only generate visualizations from existing .npz files (no visibility computation)")
    parser.add_argument("--no-visibility", action="store_true",
                        help="Skip visibility computation (faster, but no visibility overlay)")
    parser.add_argument("--b1k-data-path", type=str,
                        default="/vast/projects/kumar/lab/yishao/data/b1k_full",
                        help="Path to b1k_full data directory for visibility computation")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip episodes that already have output files")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "predicate_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle viz-only mode
    if args.viz_only:
        print("="*80)
        print("VISUALIZATION ONLY MODE")
        print("="*80)
        print(f"Output directory: {output_dir}")
        
        # Find task output directories (npz files)
        task_output_dirs = sorted(output_dir.glob("task-????"))
        if not task_output_dirs:
            print("No processed task directories found!")
            return
        
        # Filter by task_id if specified
        if args.task_id is not None:
            task_output_dirs = [d for d in task_output_dirs if int(d.name.split("-")[1]) == args.task_id]
            if not task_output_dirs:
                print(f"No data found for task {args.task_id}")
                return
        elif not args.all_tasks:
            # Only first task
            task_output_dirs = task_output_dirs[:1]
        
        viz_dir = output_dir / "predicate_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        total_visualized = 0
        for task_output_dir in task_output_dirs:
            task_id = int(task_output_dir.name.split("-")[1])
            print(f"\n--- Task {task_id:04d} ---")
            
            # Find all npz files
            npz_files = sorted(task_output_dir.glob("*_vectors.npz"))[:args.num_episodes]
            
            for npz_file in npz_files:
                episode_name = npz_file.stem.replace("_vectors", "")
                print(f"  Visualizing {episode_name}...")
                
                # Load data
                data = np.load(npz_file, allow_pickle=True)
                states = data['states']
                timesteps = data['timesteps'].tolist()
                predicate_names = data['predicate_names'].tolist()
                visibility = data.get('visibility', None)
                
                # Load metadata
                meta_path = task_output_dir / f"{episode_name}_metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        "task_name": "unknown",
                        "episode_id": episode_name,
                    }
                
                # Generate visualization
                viz_path = viz_dir / f"predicate_vectors_task{task_id:04d}_{episode_name}.png"
                visualize_predicate_vectors(states, predicate_names, timesteps, metadata, viz_path, task_id, visibility)
                total_visualized += 1
        
        print(f"\n{'='*80}")
        print(f"Visualized {total_visualized} episode(s)")
        print(f"Output saved to: {viz_dir}")
        print("="*80)
        return
    
    print("="*80)
    print("PREDICATE VECTOR GENERATION")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Compute visibility: {not args.no_visibility}")
    if not args.no_visibility:
        print(f"b1k_full data path: {args.b1k_data_path}")
    
    # Find task directories
    task_dirs = sorted(data_dir.glob("task-????"))
    if not task_dirs:
        print("No task directories found!")
        return
    
    print(f"Found {len(task_dirs)} task directories")
    
    # Determine which tasks to process
    if args.task_id is not None:
        # Process specific task
        task_dir = data_dir / f"task-{args.task_id:04d}"
        if not task_dir.exists():
            print(f"Task directory not found: {task_dir}")
            return
        tasks_to_process = [(args.task_id, task_dir)]
    elif args.all_tasks:
        # Process all tasks
        tasks_to_process = []
        for td in task_dirs:
            tid = int(td.name.split("-")[1])
            tasks_to_process.append((tid, td))
    else:
        # Process first task only
        td = task_dirs[0]
        tid = int(td.name.split("-")[1])
        tasks_to_process = [(tid, td)]
    
    print(f"\nProcessing {len(tasks_to_process)} task(s)...")
    
    results = []
    for task_id, task_dir in tasks_to_process:
        print(f"\n--- Task {task_id:04d} ---")
        result = process_task(
            task_dir, 
            output_dir, 
            task_id,
            num_episodes=args.num_episodes,
            viz_only_first=not args.no_viz,
            compute_visibility=not args.no_visibility,
            b1k_data_path=args.b1k_data_path,
            skip_existing=args.skip_existing
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get("status") == "success"]
    print(f"Successfully processed: {len(successful)}/{len(results)} tasks")
    
    for r in successful:
        num_eps = r.get('num_episodes', 1)
        total_ts = r.get('total_timesteps', r.get('num_timesteps', 0))
        print(f"  Task {r['task_id']:04d} ({r['task_name']}): "
              f"{r['num_predicates']} predicates, {num_eps} episode(s), {total_ts} total timesteps")
    
    print(f"\nOutput saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

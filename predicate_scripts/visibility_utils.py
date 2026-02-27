"""
Utility functions for checking object visibility in BEHAVIOR-1K episodes.

This module provides functions to determine which BDDL objects are visible
to the robot's cameras based on episode metadata. Visibility is determined
at the episode level (was the object ever visible in any frame).

Camera mapping:
- head: ZED camera (robot_r1::robot_r1:zed_link:Camera:0)
- left_wrist: Left RealSense (robot_r1::robot_r1:left_realsense_link:Camera:0)  
- right_wrist: Right RealSense (robot_r1::robot_r1:right_realsense_link:Camera:0)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

# BDDL to scene object category synonyms (WordNet relationships)
BDDL_TO_SCENE_SYNONYMS = {
    'ashcan': 'trash_can',
    'radio_receiver': 'radio',
    'caldron': 'cauldron',
    'armchair': 'armchair',
    'electric_refrigerator': 'fridge',
    # Add more synonyms as discovered
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
    
    Args:
        bddl_name: Full BDDL object name with WordNet suffix and instance number
        
    Returns:
        Category name with double underscores converted to single
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
    """
    Map BDDL category to scene object category using synonyms.
    
    Args:
        bddl_category: Category extracted from BDDL name
        
    Returns:
        Corresponding scene object category
    """
    return BDDL_TO_SCENE_SYNONYMS.get(bddl_category, bddl_category)


def get_scene_object_mapping(ins_id_mapping: Dict[str, str]) -> Dict[str, List[int]]:
    """
    Build mapping from scene object name to instance IDs.
    
    Args:
        ins_id_mapping: Dictionary mapping instance ID (as str) to prim path
        
    Returns:
        Dictionary mapping scene object name to list of instance IDs
    """
    scene_obj_to_ids = {}
    for ins_id, path in ins_id_mapping.items():
        match = re.search(r'/World/scene_0/([^/]+)/', path)
        if match:
            obj_name = match.group(1)
            if obj_name not in scene_obj_to_ids:
                scene_obj_to_ids[obj_name] = []
            scene_obj_to_ids[obj_name].append(int(ins_id))
    return scene_obj_to_ids


def get_unique_ids_per_camera(meta: Dict[str, Any]) -> Dict[str, Set[int]]:
    """
    Extract unique instance IDs visible per camera from episode metadata.
    
    Args:
        meta: Episode metadata dictionary
        
    Returns:
        Dictionary mapping camera name to set of visible instance IDs
    """
    unique_ids = {}
    for camera_name, key in CAMERA_KEYS.items():
        val = meta.get(key, '[]')
        if isinstance(val, str):
            ids = json.loads(val)
        else:
            ids = val
        unique_ids[camera_name] = set(ids)
    return unique_ids


def get_bddl_object_visibility(
    meta: Dict[str, Any],
    bddl_objects: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Check which cameras can see each BDDL object.
    
    Note: For categories with multiple instances (e.g., 3 soda cans),
    all BDDL instances map to all scene objects of that category.
    This means if any scene object of that category is visible,
    all BDDL instances are considered visible.
    
    Args:
        meta: Episode metadata dictionary
        bddl_objects: List of BDDL object names to check
        
    Returns:
        Dictionary mapping BDDL object name to visibility info:
        {
            'bddl_category': str,
            'scene_category': str, 
            'scene_objects': List[str],
            'instance_ids': List[int],
            'visible_on': List[str],  # camera names
        }
    """
    ins_id_mapping = json.loads(meta.get('ins_id_mapping', '{}'))
    scene_obj_to_ids = get_scene_object_mapping(ins_id_mapping)
    unique_ids = get_unique_ids_per_camera(meta)
    
    results = {}
    for bddl_name in bddl_objects:
        bddl_category = parse_bddl_category(bddl_name)
        scene_category = get_scene_category(bddl_category)
        
        # Find matching scene objects
        matching_objs = []
        matching_ids = []
        for obj_name, ids in scene_obj_to_ids.items():
            if obj_name.startswith(scene_category + '_') or obj_name == scene_category:
                matching_objs.append(obj_name)
                matching_ids.extend(ids)
        
        # Check visibility per camera
        visible_on = [
            cam for cam, ids in unique_ids.items() 
            if any(ins_id in ids for ins_id in matching_ids)
        ]
        
        results[bddl_name] = {
            'bddl_category': bddl_category,
            'scene_category': scene_category,
            'scene_objects': matching_objs,
            'instance_ids': matching_ids,
            'visible_on': visible_on,
        }
    
    return results


def get_predicate_visibility(
    meta: Dict[str, Any],
    predicates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Augment predicates with visibility information.
    
    For each predicate, adds a 'visibility' field indicating which cameras
    can see the objects involved in that predicate.
    
    Args:
        meta: Episode metadata dictionary
        predicates: List of predicate dictionaries with 'predicate' and 'objects' keys
        
    Returns:
        Same predicates with added 'visibility' field
    """
    # Collect all unique objects
    all_objects = set()
    for pred in predicates:
        all_objects.update(pred.get('objects', []))
    
    # Get visibility for all objects
    visibility = get_bddl_object_visibility(meta, list(all_objects))
    
    # Augment predicates
    for pred in predicates:
        obj_names = pred.get('objects', [])
        pred_visibility = {
            'all_visible_on': None,
            'any_visible_on': None,
            'per_object': {},
        }
        
        if obj_names:
            all_cameras = set(['head', 'left_wrist', 'right_wrist'])
            any_cameras = set()
            
            for obj in obj_names:
                obj_vis = visibility.get(obj, {}).get('visible_on', [])
                pred_visibility['per_object'][obj] = obj_vis
                if obj_vis:
                    any_cameras.update(obj_vis)
                    if pred_visibility['all_visible_on'] is None:
                        all_cameras = set(obj_vis)
                    else:
                        all_cameras &= set(obj_vis)
            
            pred_visibility['all_visible_on'] = list(all_cameras) if all_cameras else []
            pred_visibility['any_visible_on'] = list(any_cameras)
        
        pred['visibility'] = pred_visibility
    
    return predicates


def load_episode_metadata(task_id: int, episode_id: int, base_path: str = None) -> Dict[str, Any]:
    """
    Load metadata for a specific episode.
    
    Args:
        task_id: Task number (e.g., 0, 1, 2)
        episode_id: Episode number within task (e.g., 10, 20, 30)
        base_path: Base path to b1k_full data directory
        
    Returns:
        Episode metadata dictionary
    """
    if base_path is None:
        base_path = '/vast/projects/kumar/lab/yishao/data/b1k_full'
    
    episode_str = f'{task_id:04d}{episode_id:04d}'
    meta_path = Path(base_path) / 'meta' / 'episodes' / f'task-{task_id:04d}' / f'episode_{episode_str}.json'
    
    with open(meta_path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Demo: Check visibility for task 1 (throwing_away_trash)
    import argparse
    
    parser = argparse.ArgumentParser(description='Check BDDL object visibility')
    parser.add_argument('--task', type=int, default=1, help='Task ID')
    parser.add_argument('--episode', type=int, default=10, help='Episode ID within task')
    args = parser.parse_args()
    
    # Load metadata
    meta = load_episode_metadata(args.task, args.episode)
    
    # Get task_obs_keys to find BDDL objects
    task_obs_keys = meta.get('task_obs_keys', [])
    if isinstance(task_obs_keys, str):
        task_obs_keys = json.loads(task_obs_keys)
    
    # Extract unique object names
    bddl_objects = set()
    for key in task_obs_keys:
        # Keys are like 'radio_receiver.n.01_1_pos', 'radio_receiver.n.01_1_orn'
        # Extract the object name (everything before _pos, _orn, etc.)
        for suffix in ['_pos', '_orn', '_quat']:
            if key.endswith(suffix):
                obj_name = key[:-len(suffix)]
                bddl_objects.add(obj_name)
                break
    
    print(f"Task {args.task}, Episode {args.episode}")
    print(f"BDDL objects: {sorted(bddl_objects)}")
    print()
    
    visibility = get_bddl_object_visibility(meta, list(bddl_objects))
    
    for obj, info in sorted(visibility.items()):
        visible_str = ', '.join(info['visible_on']) if info['visible_on'] else 'NOT VISIBLE'
        print(f"{obj}")
        print(f"  Category: {info['bddl_category']} -> {info['scene_category']}")
        print(f"  Scene objects: {info['scene_objects']}")
        print(f"  Visible on: {visible_str}")
        print()

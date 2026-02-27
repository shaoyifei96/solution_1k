# Object Visibility in BEHAVIOR-1K Dataset

This document describes how to determine which BDDL objects are visible to the robot's cameras in the b1k_full dataset.

## Overview

The goal is to filter predicates based on whether the objects involved are visible to the robot. This is useful for training models that should only predict predicates for objects the robot can actually see.

## Data Sources

### Episode Metadata

Located at: `/vast/projects/kumar/lab/yishao/data/b1k_full/meta/episodes/task-XXXX/episode_XXXXXXXX.json`

Key fields:
- `ins_id_mapping`: JSON string mapping instance ID → prim path
- Camera visibility keys (see below)
- `task_obs_keys`: List of BDDL object observation keys

### Camera Visibility Keys

The metadata contains per-camera visibility information with these exact keys:

| Camera | Metadata Key |
|--------|-------------|
| Head (ZED) | `robot_r1::robot_r1:zed_link:Camera:0::unique_ins_ids` |
| Left Wrist (RealSense) | `robot_r1::robot_r1:left_realsense_link:Camera:0::unique_ins_ids` |
| Right Wrist (RealSense) | `robot_r1::robot_r1:right_realsense_link:Camera:0::unique_ins_ids` |

Each key contains a JSON-encoded list of instance IDs that were visible on that camera **at any point during the episode**.

> **Note**: This is episode-level visibility, not per-frame. An object is marked visible if it appeared in any frame.

## BDDL to Scene Object Mapping

### Naming Convention

BDDL objects use WordNet naming: `category.pos.sense_instance`
- Example: `can__of__soda.n.01_1` (noun, sense 01, instance 1)
- Double underscores (`__`) represent spaces in multi-word categories

Scene objects use: `category_id`
- Example: `can_of_soda_113`

### Category Extraction

```python
def parse_bddl_category(bddl_name: str) -> str:
    # 'can__of__soda.n.01_1' -> 'can_of_soda'
    parts = bddl_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        name_without_instance = parts[0]
    else:
        name_without_instance = bddl_name
    
    if '.n.' in name_without_instance:
        category = name_without_instance.split('.')[0]
    else:
        category = name_without_instance
    
    return category.replace('__', '_')
```

### Synonym Mapping

BDDL uses WordNet synsets, so different words may refer to the same object type:

| BDDL Category | Scene Category |
|---------------|----------------|
| `ashcan` | `trash_can` |
| `radio_receiver` | `radio` |
| `caldron` | `cauldron` |
| `electric_refrigerator` | `fridge` |

Add more synonyms to `BDDL_TO_SCENE_SYNONYMS` in `visibility_utils.py` as discovered.

## Instance ID Mapping

The `ins_id_mapping` field maps instance IDs to prim paths:

```json
{
  "7": "/World/scene_0/trash_can_116/base_link/visuals",
  "8": "/World/scene_0/can_of_soda_115/base_link/visuals",
  "9": "/World/scene_0/can_of_soda_114/base_link/visuals",
  "10": "/World/scene_0/can_of_soda_113/base_link/visuals"
}
```

Extract scene object name from path using regex: `/World/scene_0/([^/]+)/`

## Visibility Check Algorithm

```python
def get_bddl_object_visibility(meta, bddl_objects):
    # 1. Parse ins_id_mapping to get scene_obj -> [instance_ids]
    # 2. Get unique_ins_ids per camera
    # 3. For each BDDL object:
    #    a. Extract category (can__of__soda.n.01_1 -> can_of_soda)
    #    b. Apply synonym mapping (ashcan -> trash_can)
    #    c. Find matching scene objects (can_of_soda_113, can_of_soda_114, ...)
    #    d. Check if any instance ID is in camera's unique_ins_ids
    # 4. Return visibility per camera
```

## Example Results

### Task 1: throwing_away_trash

| BDDL Object | Scene Objects | Visible On |
|-------------|---------------|------------|
| `can__of__soda.n.01_1` | `can_of_soda_113/114/115` | head, left_wrist, right_wrist |
| `can__of__soda.n.01_2` | `can_of_soda_113/114/115` | head, left_wrist, right_wrist |
| `can__of__soda.n.01_3` | `can_of_soda_113/114/115` | head, left_wrist, right_wrist |
| `ashcan.n.01_1` | `trash_can_116` | head, left_wrist, right_wrist |
| `floor.n.01_1` | (none) | NOT VISIBLE |
| `agent.n.01_1` | (none) | NOT VISIBLE |

## Known Limitations

### 1. 1:N Mapping Issue

When multiple BDDL instances of the same category exist (e.g., 3 soda cans), we cannot determine which BDDL instance maps to which scene object:

- `can__of__soda.n.01_1` → could be any of `can_of_soda_113/114/115`
- `can__of__soda.n.01_2` → could be any of `can_of_soda_113/114/115`
- `can__of__soda.n.01_3` → could be any of `can_of_soda_113/114/115`

**Workaround**: For visibility purposes, treat all instances as visible if any scene object of that category is visible.

### 2. Objects Without Scene Representation

Some BDDL objects don't have corresponding scene objects:
- `floor.n.01_*` - floors are part of room geometry
- `agent.n.01_*` - the robot itself

### 3. Episode-Level Only

The `unique_ins_ids` is cumulative over the entire episode. We cannot determine per-frame visibility from metadata alone.

### 4. Segmentation Video Decoding

The instance segmentation videos (`seg_instance_id/*.mp4`) use lossy compression which adds noise to the YUV colors. However, **they CAN be decoded** using the `SegVideoLoader` class or the palette-based nearest-neighbor matching.

See [Frame-Level Visibility](#frame-level-visibility) section below for details.

## Usage

### Command Line

```bash
python visibility_utils.py --task 1 --episode 10
```

### In Code

```python
from visibility_utils import (
    load_episode_metadata,
    get_bddl_object_visibility
)

meta = load_episode_metadata(task_id=1, episode_id=10)
bddl_objects = ['can__of__soda.n.01_1', 'ashcan.n.01_1']
visibility = get_bddl_object_visibility(meta, bddl_objects)

for obj, info in visibility.items():
    print(f"{obj}: visible on {info['visible_on']}")
```

### Augmenting Predicates

```python
from visibility_utils import get_predicate_visibility

predicates = [
    {'predicate': 'inside', 'objects': ['can__of__soda.n.01_1', 'ashcan.n.01_1']},
]
predicates = get_predicate_visibility(meta, predicates)
# Each predicate now has 'visibility' field with:
# - all_visible_on: cameras where ALL objects are visible
# - any_visible_on: cameras where ANY object is visible
# - per_object: visibility per individual object
```

## File Location

Utility module: `b1k_2/predicate_scripts/visibility_utils.py`

---

## Predicate Vector Generation with Visibility

The predicate vector generation pipeline computes per-frame visibility for each predicate using GPU-accelerated segmentation decoding.

### Scripts

| Script | Purpose | Where to Run |
|--------|---------|--------------|
| `jobs/generate_visibility.sh` | Compute visibility data (GPU-intensive) | SLURM job |
| `jobs/visualize_predicates.sh` | Generate visualization plots | Login node |

### Workflow

**Step 1: Generate visibility data (GPU)**

```bash
# Run all 50 tasks
sbatch jobs/generate_visibility.sh

# Run specific tasks
sbatch --array=0-9 jobs/generate_visibility.sh   # Tasks 0-9
sbatch --array=5 jobs/generate_visibility.sh     # Task 5 only
```

This generates `.npz` files with visibility matrices in `data/predicate_data/predicate_vectors/task-XXXX/`.

**Step 2: Generate visualizations (login node)**

```bash
# Visualize first task, first episode
./jobs/visualize_predicates.sh

# Visualize specific task
./jobs/visualize_predicates.sh 5           # Task 5, 1 episode
./jobs/visualize_predicates.sh 5 3         # Task 5, 3 episodes

# Visualize all tasks
./jobs/visualize_predicates.sh --all       # All tasks, 1 episode each
./jobs/visualize_predicates.sh --all 5     # All tasks, 5 episodes each
```

Visualizations are saved to `data/predicate_data/predicate_vectors/predicate_viz/`.

### Python Script Options

The underlying script `predicate_scripts/generate_predicate_vectors.py` supports:

```bash
# Generate visibility only (no visualization) - for GPU jobs
python generate_predicate_vectors.py --task-id 5 --num-episodes 10 --no-viz --skip-existing

# Visualization only (from existing .npz files) - for login node
python generate_predicate_vectors.py --task-id 5 --num-episodes 3 --viz-only --no-visibility

# Full pipeline (visibility + visualization)
python generate_predicate_vectors.py --task-id 5 --num-episodes 3
```

### Output Format

Each episode produces:
- `episode_XXXXXXXX_vectors.npz` - NumPy archive with:
  - `states`: (T, P) predicate values
  - `visibility`: (T, P) visibility ratios (0-1)
  - `timesteps`: (T,) timestep indices
  - `predicate_names`: (P,) predicate labels
- `episode_XXXXXXXX_metadata.json` - Task info and predicate structure

---

## Future Work

1. **Discover more synonyms**: Run across all tasks to find BDDL categories that don't match scene objects
2. **Resolve 1:N mapping**: May need to access original HDF5 data or BDDL scope definitions

---

## Frame-Level Visibility

### Overview

While episode-level visibility uses `unique_ins_ids` from metadata, **per-frame visibility** can be extracted from the segmentation videos using a palette-based decoding approach.

### How It Works

1. **Encoding**: During data collection, instance IDs are mapped to YUV colors via an equidistant palette
2. **Compression**: Videos are compressed with lossy H.265, adding noise (153 IDs become ~17,000 unique colors)
3. **Decoding**: For each pixel, find the nearest palette color using L2 distance to recover the original ID

### Existing Utility

OmniGibson provides `SegVideoLoader` in [obs_utils.py](../BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/obs_utils.py#L352):

```python
from omnigibson.learning.utils.obs_utils import SegVideoLoader

loader = SegVideoLoader(
    data_path='/path/to/b1k_full',
    task_id=1,
    camera_id='head',  # or 'left_wrist', 'right_wrist'
    demo_id='00010010',
    id_list=th.tensor(sorted(unique_ins_ids)),  # from metadata
)

for frame_ids in loader:  # (1, H, W) tensor of instance IDs
    # Process per-frame visibility
    pass
```

> **Note**: `SegVideoLoader` requires CUDA. For CPU-only usage, see the standalone decoder below.

### Standalone Decoder (CPU)

```python
import json
import av
import numpy as np

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

def decode_seg_frame(frame_rgb, id_list, palette):
    """Decode RGB frame to instance IDs using nearest palette match."""
    frame_flat = frame_rgb.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    distances = np.sqrt(np.sum((frame_flat[:, None, :] - palette_f[None, :, :]) ** 2, axis=2))
    indices = np.argmin(distances, axis=1)
    return id_list[indices].reshape(frame_rgb.shape[:2])

# Usage
meta = json.load(open('meta/episodes/task-0001/episode_00010010.json'))
head_ids = meta['robot_r1::robot_r1:zed_link:Camera:0::unique_ins_ids']
id_list = np.array(sorted(head_ids))
palette = generate_yuv_palette(len(id_list))

container = av.open('videos/task-0001/observation.images.seg_instance_id.head/episode_00010010.mp4')
for frame in container.decode(video=0):
    rgb = frame.to_ndarray(format='rgb24')
    instance_ids = decode_seg_frame(rgb, id_list, palette)
    # instance_ids is (H, W) array with original instance IDs
```

### Example Results

For Task 1 (throwing_away_trash), episode 00010010:

| Frame | trash_can_116 | can_of_soda_115 | can_of_soda_114 | can_of_soda_113 |
|-------|---------------|-----------------|-----------------|-----------------|
| 0 | 731 px | 66 px | 7 px | not visible |
| 1916 | 5970 px | 116 px | 32 px | 8 px |
| 3833 | 8051 px | 313 px | 1047 px | 637 px |
| 5750 | 7501 px | 1785 px | 425 px | 563 px |
| 7666 | 3342 px | 1357 px | 201 px | 1749 px |

### Accuracy

- Compression adds ~17,000 unique colors from original 153 IDs
- Nearest-neighbor matching recovers ~115/153 IDs correctly in a single frame
- Task-relevant objects (soda cans, trash can) are reliably decoded
- Some small objects or edge pixels may map to wrong IDs due to compression artifacts

### Performance Considerations

- Full video has 7667 frames
- Decoding is computationally expensive (L2 distance to 153 colors for each pixel)
- Consider subsampling frames for training data filtering

**GPU Acceleration:**
- `SegVideoLoader` uses CUDA with `th.cdist()` for ~10x speedup
- Current cluster GPUs (B200, sm_100) are NOT compatible with PyTorch 2.7.1+cu126 (supports up to sm_90)
- Use CPU with scipy.cdist as fallback

**Optimization Options:**
1. **scipy.cdist**: Faster than pure numpy (~2-3x)
2. **LUT-based**: Precompute RGB→ID lookup table for quantized colors (fastest for many frames)
3. **Upgrade PyTorch**: Newer versions may support sm_100 architecture

```python
# Fast LUT-based approach (precompute once, use for all frames)
from scipy.spatial.distance import cdist

def create_rgb_to_id_lut(id_list, palette, quantization=6):
    """Create lookup table for quantized RGB -> ID mapping."""
    levels = 256 >> (8 - quantization)  # 64 levels for quant=6
    shift = 8 - quantization
    
    r = np.arange(levels) << shift
    g = np.arange(levels) << shift
    b = np.arange(levels) << shift
    rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
    all_rgb = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1).astype(np.float32)
    
    distances = cdist(all_rgb, palette.astype(np.float32))
    indices = np.argmin(distances, axis=1)
    return id_list[indices].reshape(levels, levels, levels), shift

def decode_fast(frame_rgb, lut, shift):
    """Decode using LUT (returns set of visible IDs)."""
    r, g, b = frame_rgb[:,:,0] >> shift, frame_rgb[:,:,1] >> shift, frame_rgb[:,:,2] >> shift
    return set(np.unique(lut[r, g, b]).tolist())
```

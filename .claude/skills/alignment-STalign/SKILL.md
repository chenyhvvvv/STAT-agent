---
name: alignment-stalign
title: Spatial Alignment (STalign)
slug: alignment-stalign
description: Align two cell-level spatial transcriptomics slices using STalign. User provides matching landmark points between source and target slices. Computes initial affine from landmarks, then refines with LDDMM. Creates a new aligned slice with transformed x,y coordinates.

filter_requirements:
  num_slices: 2
  modalities: [gene, gene]
  data_levels: [cell, cell]

prerequisites:
  - Source slice ID (the slice to be transformed)
  - Target slice ID (the reference slice, stays unchanged)
  - Matching landmark points between source and target (at least 3 pairs, as (source_x, source_y, target_x, target_y))
---

# Spatial Alignment Using STalign

Align two cell-level spatial slices using **STalign** LDDMM with user-provided landmark points. The source slice is transformed to match the target coordinate space. A **new aligned slice** is created; originals unchanged.

**Output**: A new slice with transformed `obs['x']`, `obs['y']`. Originals in `obs['x_original']`, `obs['y_original']`.

---

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import torch
import warnings

print("=" * 60)
print("STAGE 1: Load and Validate")
print("=" * 60)

# IMPORTANT: User specifies source, target, and landmark points
source_slice_id = 0  # <-- MODIFY: slice to transform
target_slice_id = 1  # <-- MODIFY: reference slice (unchanged)

# REQUIRED: landmark pairs [(src_x, src_y, tgt_x, tgt_y), ...]
# At least 3 pairs needed for affine initialization
landmarks = [
    (100, 200, 110, 195),
    (300, 400, 305, 410),
    (500, 150, 510, 160),
]  # <-- MODIFY: matching points between source and target

assert len(landmarks) >= 3, "Need at least 3 landmark pairs for alignment"

source_obj = session.get_slice(source_slice_id)
target_obj = session.get_slice(target_slice_id)

assert source_obj.is_cell_level, f"STalign requires cell-level data, slice {source_slice_id} is spot-level"
assert target_obj.is_cell_level, f"STalign requires cell-level data, slice {target_slice_id} is spot-level"

source_adata = source_obj.adata.copy()
target_adata = target_obj.adata.copy()

print(f"  Source (slice {source_slice_id}): {source_adata.n_obs} cells")
print(f"  Target (slice {target_slice_id}): {target_adata.n_obs} cells")
print(f"  Landmark pairs: {len(landmarks)}")
```

### Stage 2: Rasterize and Prepare Points

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import STalign.STalign as ST

print("\n" + "=" * 60)
print("STAGE 2: Rasterize and Prepare Points")
print("=" * 60)

dx = 30.0  # Pixel size for rasterization (same units as x,y)

source_x = source_adata.obs['x'].values.astype(np.float64)
source_y = source_adata.obs['y'].values.astype(np.float64)
target_x = target_adata.obs['x'].values.astype(np.float64)
target_y = target_adata.obs['y'].values.astype(np.float64)

source_coords = np.column_stack([source_x, source_y])
target_coords = np.column_stack([target_x, target_y])

# Rasterize using STalign (returns channels-first image)
XI, YI, I = ST.rasterize(source_x, source_y, dx=dx, blur=1.0, draw=0)
XJ, YJ, J = ST.rasterize(target_x, target_y, dx=dx, blur=1.0, draw=0)
xI = [YI, XI]  # LDDMM expects [row_locs, col_locs] = [Y, X]
xJ = [YJ, XJ]

# Convert landmarks to STalign's (y, x) = (row, col) order
landmarks_arr = np.array(landmarks, dtype=np.float64)
pointsI = np.column_stack([landmarks_arr[:, 1], landmarks_arr[:, 0]])  # (y, x)
pointsJ = np.column_stack([landmarks_arr[:, 3], landmarks_arr[:, 2]])  # (y, x)

# Compute initial affine transform from landmark points
L, T = ST.L_T_from_points(pointsI, pointsJ)

print(f"  Source image: {I.shape}, Target image: {J.shape}")
print(f"  Initial affine computed from {len(landmarks)} landmarks")
```

### Stage 3: Run LDDMM

```python
print("\n" + "=" * 60)
print("STAGE 3: Run LDDMM")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = ST.LDDMM(
        xI=xI, I=I, xJ=xJ, J=J,
        L=L, T=T,
        pointsI=pointsI, pointsJ=pointsJ,
        niter=300, epV=100,
        sigmaM=1.5, sigmaB=1.0, sigmaA=1.1,
        device='cpu', dtype=torch.float64,
    )
plt.close('all')

A = result['A']
v = result['v']
xv = result['xv']
print(f"  LDDMM complete")
```

### Stage 4: Transform and Create Aligned Slice

```python
print("\n" + "=" * 60)
print("STAGE 4: Transform and Create Aligned Slice")
print("=" * 60)

# Transform source coords (STalign uses row,col = y,x order)
source_pts_yx = torch.tensor(np.column_stack([source_y, source_x]), dtype=torch.float64)
transformed_yx = ST.transform_points_source_to_target(xv, v, A, source_pts_yx)
transformed_yx = transformed_yx.detach().numpy()
transformed_x = transformed_yx[:, 1]
transformed_y = transformed_yx[:, 0]

# Create aligned adata
aligned_adata = source_adata.copy()
aligned_adata.obs['x'] = transformed_x
aligned_adata.obs['y'] = transformed_y
aligned_adata.obs['x_original'] = source_x
aligned_adata.obs['y_original'] = source_y

# Create new DataSlice
from stat_agent.core.data_slice import DataSlice

new_slice_id = max(session.slice_ids) + 1
aligned_slice = DataSlice(
    slice_id=new_slice_id,
    modality=source_obj.modality,
    data_level=source_obj.data_level,
    adata=aligned_adata,
    images={},
    metadata={'tissue_name': f'aligned_slice{source_slice_id}_to_{target_slice_id}'},
)
session.add_slice(aligned_slice)

displacement = np.sqrt((transformed_x - source_x)**2 + (transformed_y - source_y)**2).mean()
print(f"  Transformed {source_adata.n_obs} cells, mean displacement: {displacement:.1f}")
print(f"  Created new slice {new_slice_id}")
```

## Visualization

### Before/After Alignment

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

axes[0].scatter(target_coords[:, 0], target_coords[:, 1], s=1, alpha=0.4, c='blue', label='Target')
axes[0].scatter(source_coords[:, 0], source_coords[:, 1], s=1, alpha=0.4, c='red', label='Source')
axes[0].set_title('Before Alignment')
axes[0].legend(markerscale=5); axes[0].set_aspect('equal')

axes[1].scatter(target_coords[:, 0], target_coords[:, 1], s=1, alpha=0.4, c='blue', label='Target')
axes[1].scatter(transformed_x, transformed_y, s=1, alpha=0.4, c='red', label='Aligned Source')
axes[1].set_title('After Alignment')
axes[1].legend(markerscale=5); axes[1].set_aspect('equal')

dx_d = transformed_x - source_x
dy_d = transformed_y - source_y
n_arrows = min(500, len(source_coords))
idx = np.random.choice(len(source_coords), n_arrows, replace=False)
axes[2].quiver(source_coords[idx, 0], source_coords[idx, 1],
               dx_d[idx], dy_d[idx], angles='xy', scale_units='xy', scale=1, alpha=0.5, width=0.003)
axes[2].set_title('Displacement Field')
axes[2].set_aspect('equal')

for ax in axes:
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('STalign Spatial Alignment')
plt.tight_layout()
plt.show()
```

---

## Notes

- **Pairwise only**: Aligns exactly 2 slices (source → target).
- **Cell-level only**: Not for spot-level (Visium) data.
- **Landmarks required**: At least 3 matching (x, y) point pairs between source and target.
- **CPU only**: GPU has known issues; always uses `device='cpu'`.
- **New slice created**: Originals preserved in `obs['x_original']`, `obs['y_original']`.
- **Requires**: `STalign`, `torch`.

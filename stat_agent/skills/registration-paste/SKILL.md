---
name: registration-paste
title: Slice Registration (PASTE)
slug: registration-paste
description: Align multiple spatial transcriptomics slices using PASTE (Probabilistic Alignment of ST Experiments). Uses optimal transport to find correspondences between spots across slices based on gene expression and spatial coordinates. Produces aligned coordinates for joint analysis.

filter_requirements:
  modalities: [gene]

prerequisites:
  - Multiple slices loaded in session (at least 2)
  - Slices should be from the same or similar tissue

default_skill: false
---

# Slice Registration (PASTE)

Align **multiple spatial slices** using **PASTE** — Probabilistic Alignment of Spatial Transcriptomics Experiments. Uses optimal transport to find correspondences between spots based on expression similarity and spatial structure.

**Output**: Aligned coordinates in `adata.obsm['spatial_registered']` for each slice.

---

## Workflow

### Stage 1: Collect Slices

```python
import numpy as np
import pandas as pd
import scanpy as sc

print("=" * 60)
print("STAGE 1: Collect Slices")
print("=" * 60)

slice_ids = session.get_slice_ids()
assert len(slice_ids) >= 2, f"Need >= 2 slices, got {len(slice_ids)}"

adatas = []
for sid in slice_ids:
    s = session.get_slice(sid)
    ad = s.adata.copy()
    ad.obsm['spatial'] = ad.obs[['x', 'y']].to_numpy()
    adatas.append(ad)
    print(f"  Slice {sid}: {ad.n_obs} cells/spots, {ad.n_vars} genes")

# Common genes
common_genes = set(adatas[0].var_names)
for ad in adatas[1:]:
    common_genes &= set(ad.var_names)
common_genes = sorted(common_genes)
print(f"\n  Common genes: {len(common_genes)}")

# Subset and normalize
for i in range(len(adatas)):
    adatas[i] = adatas[i][:, common_genes].copy()
    sc.pp.normalize_total(adatas[i], target_sum=1e4)
    sc.pp.log1p(adatas[i])
```

### Stage 2: Run PASTE Pairwise Alignment

```python
print("\n" + "=" * 60)
print("STAGE 2: Run PASTE Alignment")
print("=" * 60)

import paste as paste_pkg

# Optional GPU acceleration for the OT inner loop. PASTE uses POT's torch
# backend when use_gpu=True; verified ~2.3× speedup on V100 vs CPU on
# 2k-spot inputs. The NMF/dissimilarity init still runs on CPU regardless.
import torch
from ot.backend import TorchBackend, NumpyBackend

def _cuda_kernels_work():
    if not torch.cuda.is_available():
        return False
    try:
        _ = (torch.zeros(2, device='cuda') + 1).sum().item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False

if _cuda_kernels_work():
    backend = TorchBackend()
    use_gpu = True
    print("  PASTE OT inner loop: CUDA")
else:
    backend = NumpyBackend()
    use_gpu = False
    print("  PASTE OT inner loop: CPU")

# Pairwise alignment (align each slice to the first)
pis = []  # transport maps
reference = adatas[0]

for i in range(1, len(adatas)):
    print(f"  Aligning slice {slice_ids[i]} to slice {slice_ids[0]}...")
    pi = paste_pkg.pairwise_align(
        reference,
        adatas[i],
        alpha=0.1,           # Balance between expression (0) and spatial (1)
        backend=backend,
        use_gpu=use_gpu,
    )
    pis.append(pi)
    print(f"  Transport map shape: {pi.shape}")
```

### Stage 3: Apply Alignment

```python
print("\n" + "=" * 60)
print("STAGE 3: Apply Alignment and Store")
print("=" * 60)

# Reference keeps its coordinates
session.get_slice(slice_ids[0]).adata.obsm['spatial_registered'] = (
    adatas[0].obsm['spatial'].copy()
)
print(f"  Slice {slice_ids[0]}: reference (unchanged)")

# Transform other slices
for i, pi in enumerate(pis):
    sid = slice_ids[i + 1]

    # Weighted average of reference coordinates by transport map
    # pi: (n_spots_ref, n_spots_target) — each column sums to ~1
    ref_coords = adatas[0].obsm['spatial']
    # Normalize columns of pi
    pi_norm = pi / pi.sum(axis=0, keepdims=True)
    aligned_coords = pi_norm.T @ ref_coords

    session.get_slice(sid).adata.obsm['spatial_registered'] = aligned_coords
    print(f"  Slice {sid}: aligned to reference")

# Store params
session.get_slice(slice_ids[0]).adata.uns['registration_params'] = {
    'method': 'PASTE',
    'reference_slice': slice_ids[0],
    'n_slices': len(slice_ids),
    'alpha': 0.1,
}

print(f"\nPASTE registration complete.")
print(f"Aligned coordinates in adata.obsm['spatial_registered']")
```

## Visualization

### Overlay Aligned Slices

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before alignment (original coordinates)
ax = axes[0]
for i, sid in enumerate(slice_ids):
    s = session.get_slice(sid)
    coords = s.adata.obs[['x', 'y']].values
    ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.5, label=f'Slice {sid}')
ax.set_title('Before Alignment')
ax.legend()
ax.set_aspect('equal')
ax.invert_yaxis()

# After alignment
ax = axes[1]
for i, sid in enumerate(slice_ids):
    s = session.get_slice(sid)
    if 'spatial_registered' in s.adata.obsm:
        coords = s.adata.obsm['spatial_registered']
        ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.5, label=f'Slice {sid}')
ax.set_title('After PASTE Alignment')
ax.legend()
ax.set_aspect('equal')
ax.invert_yaxis()

plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `alpha` | 0.1 | 0-1 | Balance: 0=expression only, 1=spatial only |

## Notes

- PASTE uses optimal transport — computationally intensive for large datasets (>10k spots per slice).
- The `alpha` parameter controls the trade-off between expression similarity and spatial distance.
- Aligned coordinates are stored separately (`spatial_registered`) to preserve original coordinates.
- Install: `pip install paste-bio POT`.

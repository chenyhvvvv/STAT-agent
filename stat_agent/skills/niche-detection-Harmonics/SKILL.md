---
name: niche-detection
title: Spatial Niche Detection
slug: niche-detection
description: Identify spatial cellular niches using Harmonics hierarchical model. Detects microenvironments based on cell type composition and assigns niche labels to cells.

filter_requirements:
  modalities: [gene]
  data_levels: [cell]

prerequisites:
  - "Cell type annotations in target slice (adata.obs['celltype'])"
default_skill: true
---

# Spatial Niche Detection Using Harmonics Model

Identify and characterize spatial cellular niches (microenvironments) in your spatial transcriptomics data. This skill uses the Harmonics hierarchical model to detect tissue niches based on cell type composition and spatial proximity patterns.




## Workflow

**Copy and execute this complete workflow (all 7 stages):**

```python
# IMPORTANT: import the pacakges
from niche_analysis_lib.model import Harmonics_Model

# ============================================================
# STAGE 1: DATA PREPARATION
# ============================================================

# Get the correct AnnData object (use explicit slice access)
# Always use session.get_slice(slice_id) for correct API
slice_id = 0  # Default to first slice, or use session.current_slice_id if available
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()
slice_name = f"slice{slice_id}"

# Verify required columns and create spatial coordinates
assert 'x' in adata.obs.columns and 'y' in adata.obs.columns, "Missing spatial coordinates"
assert 'celltype' in adata.obs.columns, "Missing celltype annotations"
adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()

# Prepare data for Harmonics
adata_list = [adata]
slice_name_list = [slice_name]

print(f"✓ Stage 1: Data prepared ({adata.n_obs} cells, {adata.obs['celltype'].nunique()} types)")

# ============================================================
# STAGE 2: INITIALIZE HARMONICS MODEL
# ============================================================

model = Harmonics_Model(
    adata_list,
    slice_name_list,
    concat_label='slice_name',
    seed=1234,
    parallel=True,
    verbose=True
)

print(f"✓ Stage 2: Model initialized")

# ============================================================
# STAGE 3: PREPROCESSING
# ============================================================

model.preprocess(
    ct_key='celltype',
    spatial_key='spatial',
    method='joint',
    n_step=3,
    n_neighbors=20,
    cut_percentage=99
)

print(f"✓ Stage 3: Preprocessing complete")

# ============================================================
# STAGE 4: INITIALIZE CLUSTERS
# ============================================================

model.initialize_clusters(
    dim_reduction=True,
    explained_var=None,
    n_components=None,
    n_components_max=100,
    standardize=True,
    method='kmeans',
    Qmax=20
)

print(f"✓ Stage 4: Clusters initialized")

# ============================================================
# STAGE 5: HIERARCHICAL REFINEMENT
# ============================================================

model.hier_dist_match(
    assign_metric='jsd',
    weighted_merge=True,
    max_iters=100,
    tol=1e-4,
    test_kmeans=False
)

print(f"✓ Stage 5: Hierarchical refinement complete")

# ============================================================
# STAGE 6: SELECT SOLUTION
# ============================================================

adata_list, _ = model.select_solution(
    n_niche=None,
    niche_key='niche_label',
    auto=True,
    metric='jsd',
    threshold=0.1,
    return_adata=True,
    plot=True,
    save=False
)

print(f"✓ Stage 6: Solution selected")

# ============================================================
# STAGE 7: EXTRACT RESULTS AND UPDATE SESSION
# ============================================================

adata_result = adata_list[0]

# transfer niche_label back to slice adata
slice_obj.adata.obs['niche_label'] = adata_result.obs['niche_label'].values

n_niches = slice_obj.adata.obs['niche_label'].nunique()
print(f"✓ Stage 7: Niche detection complete!")
print(f"✓ Found {n_niches} distinct niches")
print(f"✓ Added 'niche_label' column to slice_obj.adata.obs")

# Show niche distribution
print(f"\nNiche distribution:")
print(slice_obj.adata.obs['niche_label'].value_counts().sort_index())
```

## Visualization

### Visualize Niche Spatial Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create color palette
n_niches = adata.obs['niche_label'].nunique()
palette = sns.color_palette('tab10', n_niches)

# Spatial plot
fig, ax = plt.subplots(figsize=(12, 10))
for niche_id in sorted(adata.obs['niche_label'].unique()):
    mask = adata.obs['niche_label'] == niche_id
    ax.scatter(
        adata.obs.loc[mask, 'x'],
        adata.obs.loc[mask, 'y'],
        c=[palette[int(niche_id)]],
        s=1,
        alpha=0.8,
        label=f'Niche {niche_id}'
    )

ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Spatial Niche Distribution')
ax.legend(markerscale=5, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()
```

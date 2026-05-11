---
name: pathway-enrichment-compare
title: Two-Group Pathway Enrichment Comparison
slug: pathway-enrichment-compare
description: Compare pathway / gene-set enrichment between two user-provided gene lists (typically markers of two cell populations, clusters, or conditions) and draw a mirrored bar plot — one group per side of a shared axis. Takes a dict of two gene lists in adata.uns['enrichment_genes_groups'] and tests each against a gene-set library via gseapy.enrich (Fisher's exact test + BH FDR). Supports MSigDB Hallmark, GO, Reactome, KEGG, and any other Enrichr library. Requires species specification.

filter_requirements:
  modalities: [gene]

prerequisites:
  - "Two gene lists stored as a dict in adata.uns['enrichment_genes_groups'], e.g. {group_a: [...], group_b: [...]}, or similar way to define the two gene groups to compare."
  - Species (human or mouse)
---

# Two-Group Pathway Enrichment Comparison

Compares enrichment between two gene lists and draws a mirrored bar plot. One group's enriched terms stick out to one side of the axis, the other group's to the opposite side. Useful whenever you want to ask "which pathways distinguish these two gene sets".

**Input**: a dict of two gene lists in `adata.uns['enrichment_genes_groups']`:
```python
adata.uns['enrichment_genes_groups'] = {
    "group_a": ["GENE1", "GENE2", ...],
    "group_b": ["GENE3", "GENE4", ...],
}
```
The gene lists are typically positive markers from a differential-expression step (e.g. `rank_genes_groups` / `FindAllMarkers` filtered by `padj` and `logfc`).

**Output**:
- `adata.uns['enrichment_results_groups']` — dict `{group_name: DataFrame}` with full per-term stats
- `adata.uns['enrichment_params']` — parameters used
- A mirrored bar plot figure

---

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import pandas as pd
import gseapy as gp

slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata

assert 'enrichment_genes_groups' in adata.uns, (
    "Expected adata.uns['enrichment_genes_groups'] = "
    "{'group_a': [...], 'group_b': [...]}"
)
groups = {g: list(v) for g, v in dict(adata.uns['enrichment_genes_groups']).items()}
assert len(groups) == 2, f"Need exactly 2 groups, got {len(groups)}"
group_names = list(groups.keys())
for g, genes in groups.items():
    assert len(genes) >= 5, f"Group '{g}' needs >=5 genes, got {len(genes)}"
    print(f"  {g}: {len(genes)} genes")

species = 'human'  # <-- SET SPECIES: 'human' or 'mouse'
```

### Stage 2: Choose Gene-Set Library

```python
# Common choices:
#   'MSigDB_Hallmark_2020'       (50 curated pathways, compact)
#   'GO_Biological_Process_2023' (~5000 terms, broad coverage)
#   'Reactome_2022'
#   'KEGG_2021_Human'
#   'WikiPathways_2023_Human'
library = 'MSigDB_Hallmark_2020'  # <-- SET LIBRARY
```

### Stage 3: Run Enrichment per Group

```python
# gp.enrichr (online Enrichr API) does set-size filtering and stats server-side
# — no need to download the library locally (which can hit a bytes/str decode
# bug in some gseapy versions). organism must be lowercase: 'human' or 'mouse'.
results_by_group = {}
for group_name, gene_list in groups.items():
    if len(gene_list) < 5:
        print(f"[{group_name}] only {len(gene_list)} genes — skipping")
        results_by_group[group_name] = pd.DataFrame()
        continue

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=library,
        organism=species,
        outdir=None, no_plot=True,
    )
    df = enr.results.copy()
    df['group'] = group_name
    df['hit'] = df['Overlap'].astype(str).str.split('/').str[0].astype(int)
    df['set_size'] = df['Overlap'].astype(str).str.split('/').str[1].astype(int)
    df['neg_log10_padj'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-50))
    results_by_group[group_name] = df
    print(f"[{group_name}] {len(df)} terms  "
          f"raw P<1e-3: {int((df['P-value'] < 1e-3).sum())}  "
          f"FDR<0.05: {int((df['Adjusted P-value'] < 0.05).sum())}")
```

### Stage 4: Store Results

```python
slice_obj.adata.uns['enrichment_results_groups'] = results_by_group
slice_obj.adata.uns['enrichment_params'] = {
    'group_names': group_names,
    'species': species,
    'library': library,
    'background_size': len(slice_obj.adata.var_names),
}

# Top terms per group
for group_name, df in results_by_group.items():
    print(f"\nTop 10 {group_name} terms:")
    if len(df) == 0:
        print("  (none)")
        continue
    for _, row in df.sort_values('P-value').head(10).iterrows():
        print(f"  {row['Term'][:55]:55s}  P={row['P-value']:.1e}  "
              f"padj={row['Adjusted P-value']:.1e}  hit={row['Overlap']}")
```

### Stage 5: Mirrored Bar Plot

```python
import matplotlib.pyplot as plt

p_threshold = 1e-3         # <-- tune: raw P-value cutoff for the plot
top_n = 15                 # <-- tune: top terms per group
left_color = "#D35D05"     # <-- tune: left group color
right_color = "#01702E"    # <-- tune: right group color

results_by_group = slice_obj.adata.uns['enrichment_results_groups']
group_names = list(results_by_group.keys())
left_name, right_name = group_names[0], group_names[1]

def _top(df, n):
    if len(df) == 0:
        return df
    d = df[df['P-value'] < p_threshold].copy()
    return d.sort_values('neg_log10_padj', ascending=False).head(n)

left_df = _top(results_by_group[left_name], top_n)
right_df = _top(results_by_group[right_name], top_n)
left_df = left_df.assign(signed=-left_df['neg_log10_padj'], side=left_name)
right_df = right_df.assign(signed=right_df['neg_log10_padj'], side=right_name)
plot_df = pd.concat([left_df, right_df], ignore_index=True)
plot_df = plot_df.sort_values('signed', ascending=True).reset_index(drop=True)

if len(plot_df) == 0:
    print("No terms pass the threshold — nothing to plot.")
else:
    # Split into left/right DataFrames, sorted shortest→longest within each block
    ldf = plot_df[plot_df['side'] == left_name].sort_values('neg_log10_padj')
    rdf = plot_df[plot_df['side'] == right_name].sort_values('neg_log10_padj')
    n_r, n_l = len(rdf), len(ldf)
    n_total = n_r + n_l

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * n_total + 1.5)))

    # Right block (positive x) at positions 0..n_r-1
    for i, (_, row) in enumerate(rdf.iterrows()):
        ax.barh(i, row['neg_log10_padj'], color=right_color,
                edgecolor='none', height=0.7)

    # Left block (negative x) at positions n_r..n_total-1
    for i, (_, row) in enumerate(ldf.iterrows()):
        ax.barh(n_r + i, -row['neg_log10_padj'], color=left_color,
                edgecolor='none', height=0.7)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel('log (adj. P value)')
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)

    xmax = max(
        rdf['neg_log10_padj'].max() if n_r else 1,
        ldf['neg_log10_padj'].max() if n_l else 1,
    )
    ax.set_xlim(-xmax * 1.15, xmax * 1.15)
    ax.set_ylim(-1.0, n_total + 0.5)

    # Labels on the OPPOSITE side from their bars (using blank space):
    #   Right bars → labels on the left side
    for i, (_, row) in enumerate(rdf.iterrows()):
        label = f"{row['Term']} ({row['hit']}/{row['set_size']})"
        ax.text(-0.4, i, label, ha='right', va='center', fontsize=8)
    #   Left bars → labels on the right side
    for i, (_, row) in enumerate(ldf.iterrows()):
        label = f"{row['Term']} ({row['hit']}/{row['set_size']})"
        ax.text(0.4, n_r + i, label, ha='left', va='center', fontsize=8)

    # Group name labels
    ax.text(-xmax * 0.5, n_total + 0.3, left_name, ha='center',
            va='bottom', fontsize=11, fontweight='bold',
            color=left_color, fontstyle='italic')
    ax.text(xmax * 0.5, n_total + 0.3, right_name, ha='center',
            va='bottom', fontsize=11, fontweight='bold',
            color=right_color, fontstyle='italic')

    ax.set_title(f"{left_name} vs {right_name} — {library}", fontsize=10)
    plt.tight_layout()
    plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|---|---|---|---|
| `library` | `'MSigDB_Hallmark_2020'` | Any Enrichr library name | Gene-set library to test |
| `species` | `'human'` | `'human'`, `'mouse'` | Must match gene-name casing |
| `p_threshold` | `1e-3` | 1e-2 to 1e-4 | Raw P cutoff for plot |
| `top_n` | `15` | 5–30 | Top terms per side |
| `left_color` | `"#D35D05"` | Any hex | Left bars |
| `right_color` | `"#01702E"` | Any hex | Right bars |

## Notes

- **Why raw P-value for the plot cutoff?** On small compact libraries like MSigDB Hallmark (50 pathways), BH-adjusted P is close to raw P and a raw cutoff is more intuitive. For larger libraries (GO has ~5000 terms), switch to `Adjusted P-value < 0.05`.
- **Background**: `gp.enrichr` uses the full human/mouse genome (~20k genes) as the background — matches R's `enrichR` package.
- **Gene-name casing**: pass mouse-style genes (`Cd8a`) with `species='mouse'`; pass human-style (`CD8A`) with `species='human'`. Crossing them gives empty hits.
- **Input gene list size**: for meaningful enrichment, expect at least 50 genes per group. Fewer than 20 will typically return no significant hits on Hallmark. If your DE is returning very few markers, consider loosening the DE filter (e.g. `padj < 0.1`) before calling this skill.
- **Requires gseapy**: `pip install gseapy`. First run downloads the library (cached afterwards).

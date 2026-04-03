---
name: pathway-go-enrichment
title: GO Enrichment Analysis
slug: pathway-go-enrichment
description: Find enriched Gene Ontology (GO) terms for a user-provided gene list. Takes a list of genes stored in adata.uns['go_genes'] and tests which GO biological processes, molecular functions, or cellular components are over-represented. Uses Fisher's exact test via gseapy with Benjamini-Hochberg FDR correction. Requires species specification (human or mouse).

filter_requirements:
  modalities: [gene]

prerequisites:
  - A list of gene provided by user, or a gene list stored in adata.uns['go_genes']
  - Species (human or mouse)
---

# GO Enrichment Analysis

Find which **Gene Ontology terms** are enriched in a user-provided gene list. Given a set of genes of interest, this skill tests whether specific biological processes, molecular functions, or cellular components appear more often than expected by chance.

**Input**: A list of gene provided by user, or a gene list stored in `adata.uns['go_genes']`.

**Output**: Enriched GO terms with p-values, odds ratios, and overlapping genes in `adata.uns['go_results']`.

**How the gene list gets there**: The user either:
Tells the agent specific genes: *"Run GO enrichment on genes: FOXP3, CD4, IL2RA, CTLA4, IL10"*
   → Agent stores them in `adata.uns['go_genes']` then runs this skill
Or know the .uns['go_genes'] is already stored in a specific slice adata
---

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import pandas as pd
import gseapy as gp

print("=" * 60)
print("STAGE 1: Load and Validate")
print("=" * 60)

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata

# Store the user input
user_input_genes = [] # Fill in with genes in the query
if user_input_genes:
    adata.uns['go_genes'] = user_input_genes

# Get gene list
assert 'go_genes' in adata.uns, (
    "No gene list found in adata.uns['go_genes']. "
    "Please store a list of gene names first, e.g.: "
    "adata.uns['go_genes'] = ['FOXP3', 'CD4', 'IL2RA']"
)

gene_list = list(adata.uns['go_genes'])
assert len(gene_list) >= 2, f"Need at least 2 genes, got {len(gene_list)}"

print(f"  Gene list: {len(gene_list)} genes")
if len(gene_list) <= 20:
    print(f"  Genes: {gene_list}")
else:
    print(f"  First 20: {gene_list[:20]}")
    print(f"  ... and {len(gene_list) - 20} more")

# IMPORTANT: Species — must match gene name format
species = 'human'  # <-- SET SPECIES: 'human' or 'mouse'

print(f"  Species: {species}")
```

### Stage 2: Load GO Gene Sets

```python
print("\n" + "=" * 60)
print("STAGE 2: Load GO Gene Sets")
print("=" * 60)

# IMPORTANT: GO aspect
# Options: 'GO_Biological_Process_2023', 'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023'
go_library = 'GO_Biological_Process_2023'  # <-- SET GO ASPECT

organism = 'Human' if species == 'human' else 'Mouse'

print(f"  Loading {go_library} for {organism}...")
gene_sets = gp.get_library(go_library, organism=organism)
print(f"  Loaded {len(gene_sets)} GO terms")

# Filter by size (too small = noisy, too large = uninformative)
min_size = 10
max_size = 500
gene_sets_filtered = {
    name: genes for name, genes in gene_sets.items()
    if min_size <= len(genes) <= max_size
}
print(f"  After size filter ({min_size}-{max_size}): {len(gene_sets_filtered)} GO terms")
```

### Stage 3: Run Enrichment

```python
print("\n" + "=" * 60)
print("STAGE 3: Run GO Enrichment")
print("=" * 60)

# Check gene name overlap with GO database
all_go_genes = set()
for genes in gene_sets_filtered.values():
    all_go_genes.update(genes)

overlap = set(gene_list) & all_go_genes
print(f"  Genes found in GO database: {len(overlap)}/{len(gene_list)}")

# If poor overlap, try uppercase conversion (mouse Title case -> human UPPER)
if len(overlap) < len(gene_list) * 0.3:
    gene_list_upper = [g.upper() for g in gene_list]
    overlap_upper = set(gene_list_upper) & all_go_genes
    if len(overlap_upper) > len(overlap):
        print(f"  Auto-converting to uppercase improved overlap: {len(overlap_upper)}/{len(gene_list)}")
        gene_list = gene_list_upper
        overlap = overlap_upper

assert len(overlap) >= 2, (
    f"Only {len(overlap)} genes found in GO database. "
    f"Check species setting (current: {species}) and gene name format."
)


# Run enrichment using gseapy enrich (local Fisher's exact test)
enr = gp.enrich(
    gene_list=gene_list,
    gene_sets=gene_sets_filtered,
    outdir=None,
    no_plot=True,
    verbose=False,
)

results_df = enr.results
print(f"  Tested {len(results_df)} GO terms")

# Filter significant results
fdr_threshold = 0.05
significant = results_df[results_df['Adjusted P-value'] < fdr_threshold]
print(f"  Significant (FDR < {fdr_threshold}): {len(significant)} GO terms")
```

### Stage 4: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Store Results")
print("=" * 60)

# Sort by adjusted p-value
results_df = results_df.sort_values('Adjusted P-value')

# Store full results
slice_obj.adata.uns['go_results'] = results_df
slice_obj.adata.uns['go_params'] = {
    'gene_list': gene_list,
    'n_genes': len(gene_list),
    'species': species,
    'go_library': go_library,
    'fdr_threshold': fdr_threshold,
    'n_significant': len(significant),
    'background_size': len(background),
}

# Print top results
print(f"\nTop enriched GO terms:")
print("-" * 80)
top_n = min(20, len(significant))
if top_n == 0:
    print("  No significant GO terms found (FDR < 0.05).")
    print("  Consider: more genes, different GO aspect, or relaxed threshold.")
else:
    for i, (_, row) in enumerate(significant.head(top_n).iterrows()):
        term = row['Term']
        pval = row['Adjusted P-value']
        overlap_str = row['Overlap']
        genes_str = row.get('Genes', '')
        gene_preview = ';'.join(genes_str.split(';')[:5]) if isinstance(genes_str, str) else ''
        print(f"  {i+1}. {term}")
        print(f"     FDR={pval:.2e}, Overlap={overlap_str}, Genes: {gene_preview}")

print(f"\nResults stored in adata.uns['go_results'] (DataFrame with {len(results_df)} rows)")
print(f"Parameters stored in adata.uns['go_params']")
```

## Visualization

### Bar Plot of Top GO Terms

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = slice_obj.adata.uns['go_results']
sig = results[results['Adjusted P-value'] < 0.05].head(20)

if len(sig) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.4)))

    # -log10(FDR) for visualization
    sig = sig.copy()
    sig['-log10(FDR)'] = -np.log10(sig['Adjusted P-value'].clip(lower=1e-50))

    # Shorten long GO term names
    labels = sig['Term'].apply(lambda x: x[:60] + '...' if len(str(x)) > 60 else x)

    ax.barh(range(len(sig)), sig['-log10(FDR)'].values, color='steelblue')
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(FDR)')
    ax.set_title(f'Top Enriched GO Terms ({len(sig)} significant)')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='FDR=0.05')
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No significant GO terms to plot.")
```

### Dot Plot (Overlap Size + Significance)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = slice_obj.adata.uns['go_results']
sig = results[results['Adjusted P-value'] < 0.05].head(20).copy()

if len(sig) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.4)))

    sig['-log10(FDR)'] = -np.log10(sig['Adjusted P-value'].clip(lower=1e-50))

    # Parse overlap counts (format: "3/50")
    sig['overlap_count'] = sig['Overlap'].apply(
        lambda x: int(str(x).split('/')[0]) if '/' in str(x) else 0
    )

    labels = sig['Term'].apply(lambda x: x[:60] + '...' if len(str(x)) > 60 else x)

    scatter = ax.scatter(
        sig['-log10(FDR)'],
        range(len(sig)),
        s=sig['overlap_count'] * 30,
        c=sig['-log10(FDR)'],
        cmap='Reds',
        edgecolors='black',
        linewidths=0.5,
    )

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(FDR)')
    ax.set_title('GO Enrichment Dot Plot')
    plt.colorbar(scatter, ax=ax, label='-log10(FDR)', shrink=0.7)

    # Size legend
    for s in [3, 5, 10]:
        ax.scatter([], [], s=s*30, c='gray', edgecolors='black', linewidths=0.5, label=f'{s} genes')
    ax.legend(title='Overlap', loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.show()
else:
    print("No significant GO terms to plot.")
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `species` | `'human'` | `'human'`, `'mouse'` | Must match gene name format in your data |
| `go_library` | `'GO_Biological_Process_2023'` | See below | Which GO aspect to test |
| `min_size` | `10` | 5-50 | Minimum genes per GO term |
| `max_size` | `500` | 100-2000 | Maximum genes per GO term |
| `fdr_threshold` | `0.05` | 0.01-0.25 | Significance cutoff |

**GO library options**:
- `GO_Biological_Process_2023` — biological processes (most common)
- `GO_Molecular_Function_2023` — molecular functions
- `GO_Cellular_Component_2023` — cellular compartments

## Notes

- **Gene list input**: The skill reads from `adata.uns['go_genes']`. The agent should store the gene list there before invoking this skill. The list is just plain gene name strings, e.g., `['CD4', 'FOXP3', 'IL2RA']`.
- **Background genes**: Uses all genes in `adata.var_names` as background for the statistical test. This is the standard approach for spatial transcriptomics data.
- **Gene name format**: GO databases use uppercase for human (e.g., `FOXP3`) and title case for mouse (e.g., `Foxp3`). The skill auto-detects and converts if needed.
- **Fisher's exact test**: Tests whether the overlap between your gene list and each GO term is greater than expected by chance. P-values are corrected for multiple testing using Benjamini-Hochberg FDR.
- **Requires gseapy**: Install with `pip install gseapy`. First run downloads gene set databases (cached afterwards).

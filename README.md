<!-- BANNER -->

<div align="center">

# STAT — Spatial Transcriptomics Analytical agenT

Ask in natural language, get a planned, verified, and executed analysis of spatial omics data.

[![PyPI version](https://img.shields.io/pypi/v/stat-agent.svg?color=blue)](https://pypi.org/project/stat-agent/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.05.01.722244-b31b1b.svg)](https://doi.org/10.64898/2026.05.01.722244)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗_Demo-HF_Spaces-yellow.svg)](https://huggingface.co/spaces/CyhVVVV/stat-agent-demo)

</div>


## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data format](#data-format)
- [Built-in skills](#built-in-skills)
- [LLM providers](#llm-providers)
- [Reproducing the paper](#reproducing-the-paper)
- [License](#license)

## Installation

Stable release from PyPI:

```bash
pip install stat-agent
```

With the full set of analysis skill dependencies (squidpy, scvi-tools, torch, liana, cell2location, …):

```bash
pip install "stat-agent[skills]"
```

Some skills require packages that aren't on PyPI; install separately as needed:

```bash
# STAGATE (requires PyG ecosystem wheels matching your torch + CUDA version)
pip install torch_geometric
pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html
pip install git+https://github.com/QIFEIDKN/STAGATE_pyG.git
```

(Replace `${TORCH_VER}` and `${CUDA_VER}` with your installed torch/CUDA — e.g. `2.4.1+cu121`.)

> **GPU note:** the `torch` and CUDA versions should be adjusted to match your hardware. See [pytorch.org](https://pytorch.org/get-started/locally/).

## Quick start

### Web interface

```bash
stat-web                    # serves on http://localhost:8889
# or
./start_web.sh              # also starts a Jupyter Lab alongside
```

In the UI:

1. Enter the path to your dataset directory.
2. Configure your LLM provider and paste an API key.
3. Click **Load Dataset**.
4. Ask questions in the chat panel:
   - *"Annotate cell types using the breast-cancer reference."*
   - *"Find spatially variable genes."*
   - *"Show CD8A expression in slice 1."*
   - *"Run RCTD deconvolution and overlay the dominant cell type."*


## Data format

STAT auto-detects your data layout. Place files in a single directory.

**Single-slice**

```text
dataset/
├── tissue.h5ad          # Required: AnnData with x, y in obs
└── he.tif               # Optional: H&E image (pixel coords = cell coords)
```

**Multi-slice**

```text
dataset/
├── tissue_slice_0.h5ad
├── he_slice_0.tif
├── tissue_slice_1.h5ad
└── he_slice_1.tif
```

**Multi-omics (gene + protein)**

```text
dataset/
├── tissue.h5ad          # Gene expression
├── tissue_protein.h5ad  # Protein expression
├── he.tif
└── protein_CD3.tif
``` 

**Coordinate convention.** Cell coordinates `(x, y)` in `adata.obs` map directly to image pixel `(x, y)`. No coordinate transformation. Note the array indexing swap: image array `img[y, x]` corresponds to cell `(x, y)`.

**Required AnnData fields:** `adata.obs['x']`, `adata.obs['y']`, and the expression matrix `adata.X`. `adata.obs['celltype']` is *optional* — annotation skills will populate it.

## Built-in skills

Skills are auto-discovered from `stat_agent/skills/{slug}/SKILL.md`. Each skill carries metadata (modalities, data level, prerequisites) and a templated code body. The current catalog:

<!-- SKILLS-TABLE-START -->
### Cell type annotation

| Skill | Summary |
| --- | --- |
| [Cell Type Annotation with scANVI](stat_agent/skills/celltype-annotation-scANVI/SKILL.md) | Annotate cell types in spatial transcriptomics data using scANVI transfer learning from a reference scRNA-seq dataset. |
| [Fast Cell Type Annotation (Clustering + LLM)](stat_agent/skills/celltype-annotation-GPT/SKILL.md) | Annotate cell types using unsupervised clustering, marker genes, and LLM-based annotation. |
| [Cell Type Annotation via Spatial Mapping (Tangram)](stat_agent/skills/annotation-tangram/SKILL.md) | Map single-cell reference annotations onto spatial transcriptomics data using Tangram deep learning alignment. |

### Spot deconvolution

| Skill | Summary |
| --- | --- |
| [Cell Type Deconvolution (RCTD)](stat_agent/skills/celltype-deconvolution-RCTD/SKILL.md) | Perform cell type deconvolution (or annotation on spot) on spatial transcriptomics data (Visium spots) using RCTD with a single-cell refere… |
| [Bayesian Cell Type Deconvolution (Cell2location)](stat_agent/skills/deconvolution-cell2location/SKILL.md) | Reference-based Bayesian deconvolution of spot-level spatial transcriptomics using Cell2location. |
| [Fast Spot Deconvolution (FlashDeconv)](stat_agent/skills/deconvolution-flashdeconv/SKILL.md) | Ultra-fast reference-based cell type deconvolution for spot-level spatial data using FlashDeconv. |

### Spatial domains

| Skill | Summary |
| --- | --- |
| [Spatial Domain Detection (SpaGCN)](stat_agent/skills/spatial-domain-SpaGCN/SKILL.md) | Identify spatial domains in spot-level spatial transcriptomics data using SpaGCN, integrating gene expression, spatial location, and H&E hi… |
| [Spatial Domain Detection (STAGATE)](stat_agent/skills/spatial-domain-STAGATE/SKILL.md) | Identify spatial domains using STAGATE (Spatial-Transcriptomics Graph Attention Auto-Encoder). |
| [Spatial Domain Detection (GraphST)](stat_agent/skills/spatial-domain-GraphST/SKILL.md) | Identify spatial domains in spot-level data using GraphST (Graph Self-supervised Transformer). |

### Spatial statistics & niches

| Skill | Summary |
| --- | --- |
| [Spatial Statistics Analysis](stat_agent/skills/spatial-statistics-squidpy/SKILL.md) | Compute spatial statistics including Moran's I (spatial autocorrelation of genes), Ripley's K (spatial point pattern of cell types), co-occ… |
| [Neighborhood Enrichment Analysis](stat_agent/skills/spatial-stats-neighborhood-enrichment/SKILL.md) | Compute neighborhood enrichment z-scores to identify which cell types are spatially co-localized or depleted from each other's neighborhood… |
| [Spatial Niche Detection](stat_agent/skills/niche-detection-Harmonics/SKILL.md) | Identify spatial cellular niches using Harmonics hierarchical model. |
| [Spatially Variable Genes (SpatialDE)](stat_agent/skills/svg-SpatialDE/SKILL.md) | Identify spatially variable genes using SpatialDE Gaussian process regression. |

### Differential expression & pathway

| Skill | Summary |
| --- | --- |
| [Differential Gene Expression Analysis](stat_agent/skills/differential-expression/SKILL.md) | Find differentially expressed marker genes between groups using scanpy rank_genes_groups with Wilcoxon test. |
| [GO Enrichment Analysis](stat_agent/skills/pathway-GO-enrichment/SKILL.md) | Find enriched Gene Ontology (GO) terms for a user-provided gene list. |
| [Over-Representation & Pathway Enrichment Analysis (ORA)](stat_agent/skills/enrichment-ora-ssgsea/SKILL.md) | Test whether a gene list is enriched for specific pathways or gene sets using Over-Representation Analysis (Fisher's exact test). |
| [Per-Cell Pathway Activity Scoring (ssGSEA)](stat_agent/skills/pathway-ssgsea/SKILL.md) | Compute per-cell pathway activity scores using single-sample Gene Set Enrichment Analysis (ssGSEA). |
| [Two-Group Pathway Enrichment Comparison](stat_agent/skills/pathway-enrichment-compare/SKILL.md) | Compare pathway / gene-set enrichment between two user-provided gene lists (typically markers of two cell populations, clusters, or conditi… |

### Cell-cell communication

| Skill | Summary |
| --- | --- |
| [Cell-Cell Communication Analysis (LIANA+)](stat_agent/skills/cell-communication-LIANA/SKILL.md) | Analyze cell-cell communication using LIANA+ to identify significant ligand-receptor interactions between cell types. |
| [Cell-Cell Communication Analysis (CellPhoneDB)](stat_agent/skills/cell-communication-CellPhoneDB/SKILL.md) | Analyze cell-cell communication using CellPhoneDB statistical method to identify significant ligand-receptor interactions between cell type… |

### Multi-slice integration

| Skill | Summary |
| --- | --- |
| [Batch Integration (Harmony)](stat_agent/skills/integration-Harmony/SKILL.md) | Integrate multiple spatial transcriptomics slices using Harmony batch correction. |
| [Batch Integration (BBKNN)](stat_agent/skills/integration-bbknn/SKILL.md) | Correct batch effects across multiple slices using BBKNN (Batch Balanced K-Nearest Neighbors). |
| [Batch Integration (Scanorama)](stat_agent/skills/integration-scanorama/SKILL.md) | Correct batch effects across multiple slices using Scanorama panoramic stitching. |

### Slice alignment & registration

| Skill | Summary |
| --- | --- |
| [Spatial Alignment (STalign)](stat_agent/skills/alignment-STalign/SKILL.md) | Align two cell-level spatial transcriptomics slices using STalign. |
| [Slice Registration (PASTE)](stat_agent/skills/registration-paste/SKILL.md) | Align multiple spatial transcriptomics slices using PASTE (Probabilistic Alignment of ST Experiments). |

### Trajectory inference

| Skill | Summary |
| --- | --- |
| [Pseudotime Trajectory Analysis (Palantir / DPT)](stat_agent/skills/trajectory-palantir-dpt/SKILL.md) | Infer cell developmental trajectories and pseudotime ordering using expression-based methods. |

<!-- SKILLS-TABLE-END -->

**Adding a new skill.** Create `stat_agent/skills/<your-slug>/SKILL.md` with YAML frontmatter (`name`, `title`, `description`, `filter_requirements`, `prerequisites`, optional `default_skill`), then write the analysis instructions and code template in the body. The registry will pick it up at startup. <!-- TODO: link to a CONTRIBUTING_SKILLS.md once written -->

## LLM providers

STAT supports five providers via a unified `LLMBackend`. In the web UI's *Configure LLM* panel, pick a **Provider** from the dropdown, then type the bare **Model ID** as it appears at that provider's API — no prefix needed. (Older saved configs that include a prefix like `anthropic/…` still work for backward compatibility.)

For programmatic use, export the corresponding environment variable before launching `stat-web`. Every model ID below has been verified end-to-end against the live provider API.

| Provider | Where to get a key | Env var | Default model | Other verified IDs |
| --- | --- | --- | --- | --- |
| **OpenAI** | <https://platform.openai.com/api-keys> | `OPENAI_API_KEY` | `gpt-5.4` | `gpt-5.5`, `gpt-4o` |
| **Anthropic** | <https://console.anthropic.com/settings/keys> | `ANTHROPIC_API_KEY` | `claude-opus-4-7` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| **Google Gemini** | <https://aistudio.google.com/app/apikey> | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` | `gemini-2.5-pro` |
| **DeepSeek** | <https://platform.deepseek.com/api_keys> | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | `deepseek-v4-flash` |
| **Poe** (multi-model gateway) | <https://poe.com/api_key> | `POE_API_KEY` | `claude-sonnet-4.5` | `claude-opus-4.7`, `gpt-5.5`, `gemini-3.1-pro`, `deepseek-v4-pro-el` |

> **Poe caveat.** `claude-opus-4.6` and `claude-sonnet-4.6` on Poe force extended-thinking on the bot side and are not yet supported through STAT — use `claude-opus-4.7` instead, or switch to the direct Anthropic provider.

> **Tip.** For long-context analysis (multi-slice integration, large reference profiles), prefer models with 200 k+ context: `claude-opus-4-7`, `claude-opus-4-6`, `gpt-5.5`, `gemini-3.1-pro-preview`.

> **Verify before a long run.** Use the *Test Connection* button in the *Configure LLM* panel — it sends a one-token round-trip through the same `LLMBackend` code path as the agent and reports the exact error if anything is off.

## Reproducing the paper

The analyses, figures, and benchmarks from the STAT paper live in a separate repository: `https://github.com/chenyhvvvv/STAT-PaperRepro`


## License

[BSD-3-Clause](LICENSE) © STAT contributors.

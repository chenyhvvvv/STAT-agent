# STAT

**Spatial Transcriptomics Analytical agenT**

An AI-powered platform for spatial omics analysis with multi-format support, interactive visualization, and intelligent code generation.

## Features

- **AI Agent**: Natural language interface for spatial transcriptomics analysis — ask questions, get results
- **Multi-format support**: Single-slice, multi-slice, and multi-omics (gene + protein) datasets
- **Interactive viewer**: Canvas-based spatial visualization with zoom/pan, ROI drawing, and cell overlays
- **Skill system**: Extensible analysis skills (cell type annotation, deconvolution, spatial domains, etc.)
- **Code execution**: Agent generates and runs analysis code in a sandboxed environment
- **Multi-provider LLM**: Works with OpenAI, Anthropic, Google, Deepseek, and Poe

## Installation

```bash
pip install stat-agent
```

With all optional dependencies:

```bash
pip install "stat-agent[all]"
```

Or install specific extras:

```bash
pip install "stat-agent[web]"      # Flask web interface
pip install "stat-agent[llm]"      # LLM providers
pip install "stat-agent[skills]"   # Analysis skill dependencies
```

## Quick Start

### Web Interface

```bash
stat-web
# Open http://localhost:8889
```

Or with the startup script (includes Jupyter Lab):

```bash
./start_web.sh
```

### In the web UI:

1. Enter path to your dataset directory
2. Configure LLM (API key, model)
3. Click "Load Dataset"
4. Ask questions in the chat panel: *"Annotate cell types"*, *"Find spatially variable genes"*, *"Show BRCA1 expression"*

## Data Format

STAT auto-detects your data layout. Place files in a single directory:

**Single-slice:**
```
dataset/
├── tissue.h5ad          # Required: AnnData with x, y coordinates in obs
└── he.tif               # Optional: H&E image (pixel coords = cell coords)
```

**Multi-slice:**
```
dataset/
├── tissue_slice_0.h5ad
├── he_slice_0.tif
├── tissue_slice_1.h5ad
└── he_slice_1.tif
```

**Multi-omics:**
```
dataset/
├── tissue.h5ad          # Gene expression
├── tissue_protein.h5ad  # Protein expression
├── he.tif
└── protein_CD3.tif
```

**Key**: Cell coordinates `(x, y)` in `adata.obs` map directly to image pixels `(x, y)`. No coordinate transformation needed.

## Built-in Skills

| Skill | Description |
|-------|-------------|
| Cell Type Annotation (GPT) | Unsupervised clustering + LLM-based annotation |
| Cell Type Annotation (scANVI) | Transfer learning from scRNA-seq reference |
| Deconvolution (RCTD) | Spot-level cell type deconvolution |
| Spatial Domains (SpaGCN) | Graph-based spatial domain identification |
| SVG (SpatialDE) | Spatially variable gene detection |
| Neighborhood Enrichment | Cell type co-localization analysis |
| Cell Communication (LIANA+) | Ligand-receptor interaction analysis |
| Cell Communication (CellPhoneDB) | Permutation-based interaction testing |
| GO Enrichment | Gene Ontology pathway analysis |
| Niche Detection (Harmonics) | Spatial niche identification |
| Integration (Harmony) | Multi-slice batch correction |
| Alignment (STalign) | Spatial slice alignment |

## Architecture

```
User Query → QueryPlanner → SkillFilter → LLM Matching → SkillVerifier → Code Generation → Execution
```

- **QueryPlanner**: Determines target slices, breaks complex queries into steps
- **SkillFilter**: Programmatic filtering by modality, data level, number of slices
- **SkillVerifier**: Checks prerequisites, requests missing information
- **SpatialAgent**: Generates analysis code using skill instructions + session context
- **CodeExecutor**: Sandboxed execution with state change detection

## Project Structure

```
stat_agent/
├── core/                  # Data layer
│   ├── session.py         # Multi-slice/multi-omics session
│   ├── data_slice.py      # Single data slice wrapper
│   └── roi_manager.py     # ROI geometry management
├── agent/                 # Agent pipeline
│   ├── spatial_agent_core.py
│   ├── conversation_orchestrator.py
│   ├── pipeline_executor.py
│   ├── query_planner.py
│   ├── skill_registry.py
│   ├── skill_filter.py
│   ├── skill_verifier.py
│   ├── llm_backend.py
│   └── memory.py
└── functions/
    └── io.py              # Data loading
.claude/skills/            # Skill definitions (SKILL.md + helper libs)
web_interface.py           # Flask backend + API endpoints
static/                    # Frontend (JS + CSS)
templates/                 # HTML templates
```

## License

BSD-3-Clause

"""
Spatial Transcriptomics Agent Core.

Main agent class that orchestrates LLM interactions, memory, planning,
skill usage, and code execution for spatial transcriptomics analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncIterator

from .llm_backend import LLMBackend
from .memory import ConversationMemory
from .executor import CodeExecutor, ExecutionResult
from .skill_registry import SkillRegistry
from .tools import AgentTools
from .error_reflection import reflect_on_error_and_fix, should_attempt_fix
from .prompt_logger import PromptLogger
from .notebook_logger import NotebookLogger
from ..core.session import SimpleSession

# Import new pipeline components
from .query_planner import QueryPlanner, PlanStep, PlanResult
from .skill_filter import SkillFilter
from .skill_verifier import SkillVerifier, VerificationResult
from .clarification_context import ClarificationContext
from .pipeline_executor import PipelineExecutor, PipelineResult
from .conversation_orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert spatial transcriptomics analysis assistant. You help researchers analyze spatial transcriptomics data using Python.

**Your capabilities:**
1. **Data Analysis**: Load, process, and analyze spatial transcriptomics data (AnnData format)
2. **ROI Analysis**: Define and analyze regions of interest
3. **Cell Type Analysis**: Study cell type distributions and spatial patterns
4. **Visualization**: Create spatial plots and visualizations
5. **Statistical Analysis**: Perform spatial statistics and neighborhood analysis

**Response Format:**
- Briefly explain what you will do (1-2 sentences). Do NOT reference code, say "this code", or describe implementation details — the user cannot see the code. Instead, describe the analysis action (e.g., "I'll identify spatially variable genes using SpatialDE" not "This code runs SpatialDE").
- Write Python code in ```python``` blocks
- If code output needs biological interpretation, add on the LAST line: __INTERPRET__: <what to focus on>
- If output is self-explanatory (e.g., simple counts, plots), do NOT add __INTERPRET__

## Data Access API (CRITICAL - NEW SIMPLIFIED ARCHITECTURE)

**Core Principle:** Each slice is an independent data unit with unique `slice_id`.

**Backend API (ALWAYS use explicit slice IDs in analysis code):**
```python
# Get specific slice (explicit - USE THIS)  (Always call to get the needed slice data)
slice_0 = session.get_slice(0)
adata = slice_0.adata
image = slice_0.primary_image

# Check slice properties
if slice_0.is_spot_level:
    # Use spot-level methods (deconvolution)
    print("This is spot-level data (Visium)")
elif slice_0.is_cell_level:
    # Use cell-level methods (clustering, niche)
    print("This is cell-level data")

# Check modality
if slice_0.is_gene:
    # Gene expression data
elif slice_0.is_protein:
    # Protein abundance data

# Iterate over all slices
for slice in session.iter_slices():
    analyze(slice.adata)

# Filter by modality
for slice in session.iter_slices(modality='gene'):
    gene_analysis(slice.adata)

# Filter by data level
for slice in session.iter_slices(data_level='cell'):
    clustering(slice.adata)

# Session info
session.slice_ids        # [0, 1, 2]
session.modalities       # ['gene', 'protein']
session.data_levels      # ['cell', 'spot']
```

**CRITICAL RULES:**
1. **ALWAYS specify slice_id explicitly** when accessing data
2. **NEVER use current_slice() in analysis code** - user may ask about any slice
3. **Check data_level before analysis** - spot vs cell requires different methods
4. **ROIs are per-slice** - Use `session.get_roi(roi_name)` to access ROI object

## Data Structure

**Each DataSlice has:**
- `slice_id`: int - Unique identifier (0, 1, 2, ...)
- `modality`: str - 'gene' or 'protein'
- `data_level`: str - 'cell' (single-cell) or 'spot' (Visium/spatial transcriptomics)
- `adata`: AnnData - Expression matrix with coordinates
- `images`: Dict[str, ndarray] - Named images ({'he': image, 'dapi': image})
- `metadata`: Dict - Tissue name, etc.

**AnnData structure:**
- `adata.obs['x'], adata.obs['y']`: Spatial coordinates (REQUIRED)
- `adata.obs['celltype']`: Cell type annotations (OPTIONAL - may not exist!) (For spot data, it is the dominant celltype for each spot)
- `adata.X`: Expression matrix (cells/spots × genes/proteins) (May be nd.array or sparse matrix)
- `adata.var_names`: Gene/protein names

**IMPORTANT: Cell Type Annotations are OPTIONAL**
- Check before using: `if slice.has_celltype(): ...` or `if 'celltype' in adata.obs.columns:`
- Users can annotate later using reference datasets

## Data Levels: Cell vs Spot

**Cell-Level Data** (single-cell resolution):
- Each row = one cell
- High resolution (~10μm)
- Supports: clustering, cell type annotation, niche detection
- Detection: `slice.is_cell_level`

**Spot-Level Data** (Visium, spatial transcriptomics):
- Each row = one tissue spot (~50-100μm diameter)
- Contains MIXTURE of multiple cells
- Supports: deconvolution (celltype proportions), regional analysis
- Detection: `slice.is_spot_level`
- **Special fields:**
  - `adata.uns['spot_shape']`: 'circle' or 'square'
  - `adata.uns['spot_diameter']`: Diameter in micrometers
  - `adata.obsm['deconv_weights']`: DataFrame (spots × celltypes) with proportions (May be missing if not deconvolved)
- **Virtual celltype:** `adata.obs['celltype']` shows DOMINANT type (may be misleading)
- **CANNOT use cell-level clustering** - spots are not cells!

**Always check data level first:**
```python
slice_0 = session.get_slice(0)
if slice_0.is_spot_level:
    print("Spot-level data - use deconvolution methods")
    # Access celltype proportions
    if 'deconv_weights' in slice_0.adata.obsm:
        props = slice_0.adata.obsm['deconv_weights']
elif slice_0.is_cell_level:
    print("Cell-level data - can use clustering")
```

## ROI Management (Per-Slice)

**ROIs are tied to specific slices (consistent with slice access API):**
```python
# Create ROI on slice 0
roi_def = {'type': 'bbox', 'x_min': 100, 'x_max': 500, 'y_min': 100, 'y_max': 500}
session.create_roi('tumor_region', slice_id=0, roi_definition=roi_def)

# Get ROI object (like get_slice)
roi = session.get_roi('tumor_region')

# Access ROI properties (like DataSlice)
print(f"ROI: {roi.name}")
print(f"Slice: {roi.slice_id}")          # int (0, 1, 2, ...)
print(f"Modality: {roi.modality}")       # 'gene' or 'protein'
print(f"Cells: {roi.n_obs}")             # Number of cells/spots in ROI
print(f"Type: {roi.type}")               # 'bbox', 'circle', or 'polygon' (freehand)
print(f"Bounds: {roi.bounds}")           # (x_min, y_min, x_max, y_max)

# Access filtered data (like slice.adata)
roi_adata = roi.adata                    # AnnData filtered to ROI
celltype_counts = roi.adata.obs['celltype'].value_counts()

# Iterate all ROIs
for roi_name in session.roi_subsets.keys():
    roi = session.get_roi(roi_name)
    print(f"ROI '{roi.name}' on slice {roi.slice_id}: {roi.n_obs} cells")
```

**ROI Object Structure:**
- `roi.name`: str - ROI name
- `roi.slice_id`: int - Which slice (0, 1, 2, ...)
- `roi.modality`: str - 'gene' or 'protein'
- `roi.adata`: AnnData - Filtered data for this ROI
- `roi.geometry`: shapely geometry - Full geometry object
- `roi.type`: str - 'bbox', 'circle', or 'polygon' (freehand)
- `roi.bounds`: tuple - (x_min, y_min, x_max, y_max)
- `roi.area`: float - Area in coordinate units
- `roi.centroid`: tuple - (center_x, center_y)
- `roi.n_obs`: int - Number of cells/spots
- `roi.n_vars`: int - Number of genes/proteins
```

## Available in Namespace

```python
session      # SimpleSession object
```

Remenber to import any additional libraries you need in your code blocks, for example, 
```python
# You need to import libraries in your code blocks - they are not pre-imported
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```



## Instructions

1. **Check session info first** using `session.get_summary()`
2. **Always specify slice_id explicitly** - don't rely on "current"
3. **Check data_level** before choosing analysis method
4. **Check for celltype** before using: `if slice.has_celltype(): ...`
5. **Write clean Python code** with proper error handling
6. **Create visualizations** - matplotlib figures are auto-captured and displayed. Use `plt.show()` to finish plots. Do NOT use `plt.savefig()` — it is unnecessary and will clutter the working directory
7. **Format code blocks** with ```python fences

**When NOT to generate code:**
- Greetings/introductions
- "What can you do?" questions  - Clarification questions
- Describing capabilities

## Example Workflows

**Example 1: Analyze specific slice**
```python
# User: "Analyze slice 0"
slice_0 = session.get_slice(0)
print(f"Slice 0: {slice_0.modality}, {slice_0.data_level}")
print(f"Shape: {slice_0.n_obs} obs × {slice_0.n_vars} vars")

if slice_0.has_celltype():
    counts = slice_0.adata.obs['celltype'].value_counts()
    print(f"\nCell types:\n{counts}")
else:
    print("\nNo celltype annotations available")
```

**Example 2: Visualize gene expression**
```python
import matplotlib.pyplot as plt

# Get slice
slice_0 = session.get_slice(0)
gene = "ERBB2"

if gene not in slice_0.adata.var_names:
    print(f"Gene {gene} not found")
else:
    expr = slice_0.adata[:, gene].X.toarray().flatten()

    plt.figure(figsize=(10, 8))
    plt.scatter(slice_0.adata.obs['x'], slice_0.adata.obs['y'],
                c=expr, s=1, cmap='viridis', alpha=0.8)
    plt.colorbar(label=f'{gene} expression')
    plt.title(f'Spatial Expression of {gene}')
    plt.show()
```

**Example 3: Compare across all gene slices**
```python
# User: "Compare ERBB2 expression across all gene slices"
import pandas as pd

gene = "ERBB2"
results = []

for slice in session.iter_slices(modality='gene'):
    if gene in slice.adata.var_names:
        expr = slice.adata[:, gene].X.toarray().flatten()
        results.append({
            'slice_id': slice.slice_id,
            'tissue': slice.metadata.get('tissue_name', f'slice_{slice.slice_id}'),
            'mean_expr': expr.mean(),
            'max_expr': expr.max()
        })

df = pd.DataFrame(results)
print(df)
```

**Example 4: ROI analysis on specific slice**
```python
# Get ROI object (consistent with slice access)
roi = session.get_roi('tumor_region')

if roi is None:
    print("ROI 'tumor_region' not found")
elif "celltype" not in roi.adata.obs.columns:
    print(f"ROI has {roi.n_obs} cells but no celltype annotations")
else:
    counts = roi.adata.obs['celltype'].value_counts()
    print(f"ROI '{roi.name}' on slice {roi.slice_id}:")
    print(f"Total cells: {roi.n_obs:,}")
    print(f"\nCell type distribution:\n{counts}")
```

**Example 5: Check and handle spot data**
```python
slice_0 = session.get_slice(0)

if slice_0.is_spot_level:
    print("This is spot-level data (Visium)")
    print(f"Spot diameter: {slice_0.adata.uns.get('spot_diameter', 'unknown')}")

    if 'deconv_weights' in slice_0.adata.obsm:
        # Access celltype proportions
        props = slice_0.adata.obsm['deconv_weights']
        print(f"\nCelltype deconvolution available:")
        print(f"Celltypes: {props.columns.tolist()}")
        print(f"\nMean proportions:\n{props.mean()}")
    else:
        print("No deconvolution data available")
else:
    print("This is cell-level data - can use clustering methods")
```
"""


class SpatialAgent:
    """
    Comprehensive spatial transcriptomics analysis agent.

    Features:
    - Multi-provider LLM support (OpenAI, Anthropic, Google, etc.)
    - Conversation memory with context management
    - Task planning for complex workflows
    - Skill-based architecture for extensibility
    - Safe code execution with session management
    - Streaming responses for real-time feedback

    Parameters
    ----------
    model : str
        LLM model to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
    api_key : Optional[str]
        API key for the LLM provider
    session : Optional[SimpleSession]
        Spatial transcriptomics session
    skill_dir : Optional[Path]
        Directory containing skill definitions
    enable_planning : bool
        Enable task planning for complex queries (default: True)
    enable_skills : bool
        Enable skill-based routing (default: True)
    max_context_messages : int
        Maximum messages in LLM context window (default: 20)
    safe_mode : bool
        Enable code safety checks (default: True)
    enable_prompt_logging : bool
        Enable logging of LLM prompts and responses (default: True)
    prompt_log_dir : str
        Directory for log files (default: "logs")
    enable_notebook_logging : bool
        Enable logging of code execution as Jupyter notebooks (default: True)

    Examples
    --------
    >>> agent = SpatialAgent(
    ...     model="gpt-4o",
    ...     api_key=os.getenv("OPENAI_API_KEY")
    ... )
    >>> response = await agent.chat("Show me the cell type distribution")
    >>> print(response)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        session: Optional[SimpleSession] = None,
        skill_dir: Optional[Path] = None,
        enable_planning: bool = True,
        enable_skills: bool = True,
        max_context_messages: int = 20,
        safe_mode: bool = True,
        enable_prompt_logging: bool = True,
        prompt_log_dir: str = "logs",
        enable_notebook_logging: bool = True,
        **llm_kwargs
    ):
        # Core components
        self.llm = LLMBackend(
            system_prompt=SYSTEM_PROMPT,
            model=model,
            api_key=api_key,
            **llm_kwargs
        )
        self.memory = ConversationMemory(
            max_full_messages=8,  # Show last 8 messages (4 turns) in full
            message_summary_threshold=200,  # Summarize if message > 200 chars
            max_context_messages=max_context_messages,  # For conversation summarization trigger
            llm_backend=self.llm  # Pass LLM for summarization
        )
        self.executor = CodeExecutor(safe_mode=safe_mode)
        self.tools = AgentTools(session=session)  # Tool-based system introspection
        self.session = session
        self._last_plots = []  # Store plots from last execution

        # Prompt logging for debugging and optimization
        self.prompt_logger = PromptLogger(log_dir=prompt_log_dir, enabled=enable_prompt_logging)
        logger.info(f"PromptLogger initialized: enabled={self.prompt_logger.enabled}, dir={self.prompt_logger.log_dir.absolute()}")

        # Notebook logging for reproducible analyses
        self.notebook_logger = NotebookLogger(log_dir=prompt_log_dir, enabled=enable_notebook_logging)
        logger.info(f"NotebookLogger initialized: enabled={self.notebook_logger.enabled}, dir={self.notebook_logger.log_dir.absolute()}")

        # NEW: Track loaded skill paths for persistent loading across conversation
        self._loaded_skill_paths = set()  # Persist skill imports for multi-turn conversations

        # NEW: Track state changes for frontend synchronization
        self._state_changes = {
            'rois_added': [],       # List of ROI names added
            'rois_deleted': [],     # List of ROI names deleted
            'celltypes_updated': [], # List of slice_ids with celltype updates
            'celltype_colors_updated': [], # List of slice_ids with color updates
            'deconv_weights_updated': []  # List of slice_ids with deconv_weights updates
        }
        # Persisted before_state for multi-turn state detection
        self._turn_before_state = {}

        # Initialize pipeline components
        self.query_planner = QueryPlanner(llm_backend=self.llm, prompt_logger=self.prompt_logger) if enable_planning else None
        self.skill_filter = SkillFilter() if enable_skills else None
        self.skill_verifier = SkillVerifier(llm_backend=self.llm, prompt_logger=self.prompt_logger) if enable_skills else None
        self.skill_registry = None

        # Track clarification state for multi-turn conversations
        self.clarification_context = ClarificationContext()

        # Pipeline executor (will be initialized after skill_registry is set up)
        self.pipeline_executor = None
        self.conversation_orchestrator = None

        if enable_skills:
            # Auto-discover .claude/skills/ if skill_dir not provided
            if skill_dir is None:
                # Try to find .claude/skills/ relative to current working directory or package root
                project_root = Path.cwd()
                skill_dir = project_root / ".claude" / "skills"

                # If not found in cwd, try relative to this file (package installation)
                if not skill_dir.exists():
                    package_root = Path(__file__).parent.parent.parent
                    skill_dir = package_root / ".claude" / "skills"

            if skill_dir and skill_dir.exists():
                # Use progressive disclosure: load metadata only at startup for fast initialization
                self.skill_registry = SkillRegistry(skill_root=skill_dir, progressive_disclosure=True)
                self.skill_registry.load()
                logger.info(f"Loaded skills from {skill_dir}")
            else:
                logger.warning(f"Skill directory not found: {skill_dir}")

        # Initialize pipeline executor (after skill_registry is set up)
        if enable_planning and enable_skills and self.skill_registry:
            self.pipeline_executor = PipelineExecutor(
                query_planner=self.query_planner,
                skill_filter=self.skill_filter,
                skill_verifier=self.skill_verifier,
                skill_registry=self.skill_registry,
                semantic_matcher=self._select_skill_matches_llm_filtered,
                session=self.session
            )
            logger.info("PipelineExecutor initialized")

            # Initialize conversation orchestrator (after pipeline_executor)
            self.conversation_orchestrator = ConversationOrchestrator(
                pipeline_executor=self.pipeline_executor,
                code_executor=self._handle_with_llm_events,
                clarification_context=self.clarification_context,
                skill_registry=self.skill_registry,
                session=self.session,
                clear_clarification_fn=self._clear_clarification_context,
                memory=self.memory
            )
            logger.info("ConversationOrchestrator initialized")
        else:
            self.conversation_orchestrator = None

        # Configuration
        self.enable_planning = enable_planning
        self.enable_skills = enable_skills

        logger.info(
            f"Initialized SpatialAgent (model={model}, "
            f"planning={enable_planning}, skills={enable_skills})"
        )

    def set_session(self, session: SimpleSession) -> None:
        """Set or update the spatial transcriptomics session."""
        self.session = session
        self.tools.set_session(session)  # Update tools with new session

        # Update orchestrator session if it exists
        if self.conversation_orchestrator is not None:
            self.conversation_orchestrator.session = session

        # Update memory metadata
        if session and session.has_data:
            summary = session.get_summary()

            # Compute total cells and genes across all slices
            total_cells = sum(s['n_obs'] for s in summary['slices'])
            total_genes = max((s['n_vars'] for s in summary['slices']), default=0)

            # Collect all unique celltypes across all slices
            all_celltypes = set()
            for s in summary['slices']:
                all_celltypes.update(s.get('celltypes', []))

            self.memory.update_session_metadata(
                data_loaded=True,
                n_cells=total_cells,
                n_genes=total_genes,
                celltypes=sorted(list(all_celltypes)),
                n_rois=summary.get('n_rois', 0)
            )

        logger.info("Updated session in agent")

    def _clear_state_changes(self) -> None:
        """Clear state change tracker."""
        self._state_changes = {
            'rois_added': [],
            'rois_deleted': [],
            'celltypes_updated': [],
            'celltype_colors_updated': [],
            'deconv_weights_updated': []
        }

    def _detect_state_changes(self, before_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect state changes by comparing before and after session state.

        Args:
            before_state: Session state snapshot before execution

        Returns:
            Dictionary of detected state changes
        """
        if not self.session or not self.session.has_data:
            return self._state_changes

        # Get current state
        after_summary = self.session.get_summary()

        # Detect ROI changes - get ROI names directly from session
        before_rois = set(before_state.get('roi_names', []))
        after_rois = set(self.session.roi_subsets.keys())

        added_rois = list(after_rois - before_rois)
        deleted_rois = list(before_rois - after_rois)

        if added_rois:
            self._state_changes['rois_added'] = added_rois
        if deleted_rois:
            self._state_changes['rois_deleted'] = deleted_rois

        # Detect celltype updates (check if celltype was added OR values changed)
        if before_state.get('n_slices', 1) > 1 or after_summary.get('n_slices', 1) > 1:
            # Multi-slice: Check each slice for celltype changes
            for slice_id in range(after_summary.get('n_slices', 0)):
                try:
                    slice_data = self.session.slices.get(slice_id)
                    if not slice_data:
                        continue
                    has_celltype_now = 'celltype' in slice_data.adata.obs.columns

                    if has_celltype_now:
                        # Get current celltype values
                        current_celltypes = tuple(sorted(slice_data.adata.obs['celltype'].unique()))

                        # Check if celltype column was newly added
                        before_had_celltype = before_state.get(f'slice_{slice_id}_had_celltype', False)

                        if not before_had_celltype:
                            # Newly added celltype column
                            self._state_changes['celltypes_updated'].append(slice_id)
                        else:
                            # Column existed before - check if VALUES changed
                            before_celltypes = before_state.get(f'slice_{slice_id}_celltypes')
                            if before_celltypes != current_celltypes:
                                # Celltype values changed (re-annotation)
                                self._state_changes['celltypes_updated'].append(slice_id)
                                logger.info(
                                    f"Slice {slice_id} celltype re-annotated: "
                                    f"{len(before_celltypes) if before_celltypes else 0} types → "
                                    f"{len(current_celltypes)} types"
                                )

                except (IndexError, AttributeError):
                    pass

            # NEW: Detect celltype color changes (per-slice in multi-slice mode)
            for slice_id in after_summary.get('slice_ids', []):
                try:
                    slice_data = self.session.slices.get(slice_id)
                    if not slice_data or not slice_data.has_celltype():
                        continue

                    # Get current colors
                    current_colors = slice_data.get_celltype_colors()
                    before_colors = before_state.get(f'slice_{slice_id}_colors')

                    # Detect if colors changed (or were newly added)
                    if current_colors != before_colors:
                        self._state_changes['celltype_colors_updated'].append(slice_id)
                        logger.info(f"Slice {slice_id} celltype colors updated")

                except (IndexError, AttributeError):
                    pass
        else:
            # Single slice - check main adata
            slice_0 = self.session.get_slice(0)
            if slice_0 and 'celltype' in slice_0.adata.obs.columns:
                before_had_celltype = before_state.get('had_celltype', False)

                if not before_had_celltype:
                    # Newly added
                    self._state_changes['celltypes_updated'].append(0)
                else:
                    # Check if values changed
                    current_celltypes = tuple(sorted(slice_0.adata.obs['celltype'].unique()))
                    before_celltypes = before_state.get('celltypes')
                    if before_celltypes != current_celltypes:
                        self._state_changes['celltypes_updated'].append(0)

            # NEW: Detect celltype color changes (single-slice mode)
            if slice_0 and slice_0.has_celltype():
                current_colors = slice_0.get_celltype_colors()
                before_colors = before_state.get('slice_0_colors')

                if current_colors != before_colors:
                    self._state_changes['celltype_colors_updated'].append(0)
                    logger.info(f"Slice 0 celltype colors updated")

        # Detect deconv_weights updates (RCTD deconvolution results)
        if before_state.get('data_format') in ['multi_slice', 'multi_omics']:
            for slice_id in range(after_summary.get('n_slices', 0)):
                try:
                    slice_data = self.session.slices.get(slice_id)
                    if not slice_data:
                        continue
                    has_deconv_now = 'deconv_weights' in slice_data.adata.obsm

                    if has_deconv_now:
                        # Get current deconv_weights shape
                        current_deconv_shape = slice_data.adata.obsm['deconv_weights'].shape
                        before_had_deconv = before_state.get(f'slice_{slice_id}_had_deconv', False)

                        if not before_had_deconv:
                            # Newly added deconv_weights
                            self._state_changes['deconv_weights_updated'].append(slice_id)
                            logger.info(f"Slice {slice_id}: deconv_weights added ({current_deconv_shape[0]} spots, {current_deconv_shape[1]} celltypes)")
                        else:
                            # Check if shape or values changed
                            before_deconv_shape = before_state.get(f'slice_{slice_id}_deconv_shape')
                            if before_deconv_shape != current_deconv_shape:
                                self._state_changes['deconv_weights_updated'].append(slice_id)
                                logger.info(f"Slice {slice_id}: deconv_weights changed from {before_deconv_shape} to {current_deconv_shape}")

                except (IndexError, AttributeError, KeyError):
                    pass
        else:
            # Single slice - check main adata
            slice_0 = self.session.get_slice(0)
            if slice_0 and 'deconv_weights' in slice_0.adata.obsm:
                current_deconv_shape = slice_0.adata.obsm['deconv_weights'].shape
                before_had_deconv = before_state.get('had_deconv', False)

                if not before_had_deconv:
                    # Newly added
                    self._state_changes['deconv_weights_updated'].append(0)
                    logger.info(f"deconv_weights added ({current_deconv_shape[0]} spots, {current_deconv_shape[1]} celltypes)")
                else:
                    # Check if shape changed
                    before_deconv_shape = before_state.get('deconv_shape')
                    if before_deconv_shape != current_deconv_shape:
                        self._state_changes['deconv_weights_updated'].append(0)
                        logger.info(f"deconv_weights changed from {before_deconv_shape} to {current_deconv_shape}")

        return self._state_changes

    def _track_data_modifications_in_memory(self, before_state: Dict[str, Any]) -> None:
        """
        Track detected data modifications explicitly in memory.

        This ensures the agent knows exactly what columns/data exist.
        Converts _state_changes into DataModification entries.

        Args:
            before_state: State snapshot before execution (for getting metadata)
        """
        from .memory import DataModification
        from datetime import datetime

        current_turn = len(self.memory.turns)

        # Track celltype annotations
        if self._state_changes['celltypes_updated']:
            for slice_id in self._state_changes['celltypes_updated']:
                # Get celltype info (unified API)
                slice_data = self.session.slices.get(slice_id)
                n_types = len(slice_data.adata.obs['celltype'].unique()) if slice_data else 0
                target = f"session.get_slice({slice_id}).adata.obs"

                # Check if this is new addition or update
                # Use consistent key naming: always use slice_{id} format for multi-slice
                if before_state.get('n_slices', 1) > 1:
                    # Multi-slice mode: use slice_{id}_had_celltype
                    had_before = before_state.get(f'slice_{slice_id}_had_celltype', False)
                else:
                    # Single-slice mode: use had_celltype for slice 0
                    had_before = before_state.get('had_celltype', False)
                mod_type = "column_updated" if had_before else "column_added"

                self.memory.data_modifications.add_modification(
                    DataModification(
                        modification_type=mod_type,
                        target=target,
                        details={
                            "column_name": "celltype",
                            "n_values": n_types,
                            "dtype": "category",
                            "description": f"Cell type annotations ({n_types} types)"
                        },
                        timestamp=datetime.now().isoformat(),
                        source_turn=current_turn
                    )
                )

        # Track deconv_weights additions/updates
        if self._state_changes['deconv_weights_updated']:
            for slice_id in self._state_changes['deconv_weights_updated']:
                # Get deconv_weights info
                slice_data = self.session.slices.get(slice_id)
                if slice_data and 'deconv_weights' in slice_data.adata.obsm:
                    deconv_df = slice_data.adata.obsm['deconv_weights']
                    n_spots, n_celltypes = deconv_df.shape
                    target = f"session.get_slice({slice_id}).adata.obsm"

                    # Check if this is new or update
                    # Use consistent key naming: always use slice_{id} format for multi-slice
                    if before_state.get('n_slices', 1) > 1:
                        # Multi-slice mode: use slice_{id}_had_deconv
                        had_before = before_state.get(f'slice_{slice_id}_had_deconv', False)
                    else:
                        # Single-slice mode: use had_deconv for slice 0
                        had_before = before_state.get('had_deconv', False)
                    mod_type = "obsm_updated" if had_before else "obsm_added"

                    self.memory.data_modifications.add_modification(
                        DataModification(
                            modification_type=mod_type,
                            target=target,
                            details={
                                "key": "deconv_weights",
                                "shape": f"({n_spots}, {n_celltypes})",
                                "n_celltypes": n_celltypes,
                                "celltypes": list(deconv_df.columns) if hasattr(deconv_df, 'columns') else [],
                                "description": f"Deconvolution weights ({n_celltypes} cell types)"
                            },
                            timestamp=datetime.now().isoformat(),
                            source_turn=current_turn
                        )
                    )

        # Track celltype_colors additions/updates
        if self._state_changes['celltype_colors_updated']:
            for slice_id in self._state_changes['celltype_colors_updated']:
                slice_data = self.session.slices.get(slice_id)
                if slice_data:
                    colors = slice_data.get_celltype_colors()
                    target = f"session.get_slice({slice_id}).adata.uns"

                    if colors:
                        self.memory.data_modifications.add_modification(
                            DataModification(
                                modification_type="uns_updated",
                                target=target,
                                details={
                                    "key": "celltype_colors",
                                    "n_colors": len(colors),
                                    "celltypes": list(colors.keys()),
                                    "description": f"Cell type color mapping ({len(colors)} types)"
                                },
                                timestamp=datetime.now().isoformat(),
                                source_turn=current_turn
                            )
                        )

        # Track ROI creations
        if self._state_changes['rois_added']:
            for roi_name in self._state_changes['rois_added']:
                # Get ROI object
                roi = self.session.get_roi(roi_name)
                if roi:
                    self.memory.data_modifications.add_modification(
                        DataModification(
                            modification_type="roi_created",
                            target="session.roi_subsets",
                            details={
                                "roi_name": roi_name,
                                "n_cells": roi.n_obs,
                                "slice_id": roi.slice_id,
                                "modality": roi.modality
                            },
                            timestamp=datetime.now().isoformat(),
                            source_turn=current_turn
                        )
                    )

        # Track ROI deletions
        if self._state_changes['rois_deleted']:
            for roi_name in self._state_changes['rois_deleted']:
                self.memory.data_modifications.add_modification(
                    DataModification(
                        modification_type="roi_deleted",
                        target="session.roi_subsets",
                        details={"roi_name": roi_name},
                        timestamp=datetime.now().isoformat(),
                        source_turn=current_turn
                    )
                )

    def _get_state_snapshot(self) -> Dict[str, Any]:
        """Get current session state snapshot for change detection.

        IMPORTANT: Stores celltype VALUES, not just presence, to detect re-annotation.
        """
        if not self.session or not self.session.has_data:
            return {}

        summary = self.session.get_summary()

        # Access session directly for state information
        snapshot = {
            'roi_names': list(self.session.roi_subsets.keys()),
            'n_slices': summary['n_slices'],
            'data_format': 'multi_slice' if summary['n_slices'] > 1 else 'single_slice',
        }

        # For multi-slice, record per-slice state
        if summary['n_slices'] > 1:
            for slice_id in summary['slice_ids']:
                try:
                    slice_data = self.session.slices.get(slice_id)
                    if not slice_data:
                        continue
                    has_celltype = 'celltype' in slice_data.adata.obs.columns
                    snapshot[f'slice_{slice_id}_had_celltype'] = has_celltype

                    # CRITICAL FIX: Store celltype VALUES to detect re-annotation
                    if has_celltype:
                        # Store sorted list of unique celltypes as a tuple (hashable)
                        celltypes = tuple(sorted(slice_data.adata.obs['celltype'].unique()))
                        snapshot[f'slice_{slice_id}_celltypes'] = celltypes

                        # NEW: Store celltype colors to detect color changes
                        colors = slice_data.get_celltype_colors()
                        snapshot[f'slice_{slice_id}_colors'] = colors
                    else:
                        snapshot[f'slice_{slice_id}_celltypes'] = None
                        snapshot[f'slice_{slice_id}_colors'] = None

                    # Store deconv_weights shape
                    has_deconv = 'deconv_weights' in slice_data.adata.obsm
                    snapshot[f'slice_{slice_id}_had_deconv'] = has_deconv
                    if has_deconv:
                        deconv_shape = slice_data.adata.obsm['deconv_weights'].shape
                        snapshot[f'slice_{slice_id}_deconv_shape'] = deconv_shape
                    else:
                        snapshot[f'slice_{slice_id}_deconv_shape'] = None

                except (IndexError, AttributeError):
                    pass
        else:
            # Single slice - store celltype values
            slice_0 = self.session.get_slice(0)
            has_celltype = 'celltype' in slice_0.adata.obs.columns if slice_0 else False
            snapshot['had_celltype'] = has_celltype

            if has_celltype and slice_0:
                celltypes = tuple(sorted(slice_0.adata.obs['celltype'].unique()))
                snapshot['celltypes'] = celltypes

                # NEW: Store celltype colors to detect color changes
                colors = slice_0.get_celltype_colors()
                snapshot['slice_0_colors'] = colors
            else:
                snapshot['celltypes'] = None
                snapshot['slice_0_colors'] = None

            # Store deconv_weights shape
            has_deconv = 'deconv_weights' in slice_0.adata.obsm if slice_0 else False
            snapshot['had_deconv'] = has_deconv
            if has_deconv and slice_0:
                snapshot['deconv_shape'] = slice_0.adata.obsm['deconv_weights'].shape
            else:
                snapshot['deconv_shape'] = None

        return snapshot

    async def chat(
        self,
        user_message: str,
        execute_code: bool = True,
        stream: bool = False
    ) -> str:
        """
        Process a user message and return a response.

        Parameters
        ----------
        user_message : str
            User's question or request
        execute_code : bool
            Whether to execute generated code (default: True)
        stream : bool
            Whether to stream the response (default: False)

        Returns
        -------
        str
            Agent's response

        Examples
        --------
        >>> response = await agent.chat("What data is loaded?")
        >>> print(response)
        """
        logger.info(f"Processing user message: {user_message[:100]}...")

        # Check if this is a clarification response BEFORE starting new turn
        is_clarification_response = self.clarification_context.is_waiting_for_clarification()

        if is_clarification_response:
            # This is a clarification response - don't start new turn
            logger.info("Detected clarification response - continuing existing turn")
            # Log the clarification Q&A to existing turn
            self._log_clarification_response(user_message)
        else:
            # New turn - start logging
            logger.info("Starting new conversation turn")
            self.prompt_logger.start_turn(user_message)

        # Track timing and tokens for logging
        turn_start = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_llm_calls = 0

        # State change tracking: take before_state ONLY on new turns
        if not is_clarification_response:
            self._clear_state_changes()
            self._turn_before_state = self._get_state_snapshot()
        before_state = self._turn_before_state

        # Add user message to memory
        self.memory.add_user_message(user_message)

        # Use robust pipeline (QueryPlanner → SkillFilter → Matcher → Verifier → Execute)
        logger.info("Processing with pipeline: QueryPlanner → SkillFilter → Verifier")

        # Check if this is a clarification response
        clarification_response = None
        if self.clarification_context.is_waiting_for_clarification():
            # User is responding to a previous question
            clarification_response = user_message
            logger.debug(f"Detected clarification response to previous question")

        result = await self._handle_with_pipeline(
            user_message,
            execute_code=execute_code,
            clarification_response=clarification_response
        )

        # Handle different result types
        if result["type"] == "clarification_needed":
            # Planner needs clarification
            logger.info(f"Planner requesting clarification")
            response = f"❓ {result['question']}"

        elif result["type"] == "skill_selection":
            # Multiple skills matched - need user selection
            logger.info(f"Multiple skills matched, requesting user selection")
            response = f"🎯 {result['message']}"

        elif result["type"] == "prerequisites_needed":
            # Verifier needs info
            logger.info(f"Verifier requesting prerequisites for skill: {result.get('skill', 'unknown')}")
            questions_text = "\n".join(f"  {i}. {q}" for i, q in enumerate(result['questions'], 1))
            response = f"📋 To proceed, I need some information:\n{questions_text}"

        elif result["type"] == "advice":
            # Can't proceed - needs prior work
            logger.info(f"Verifier advising prior work needed")
            response = f"ℹ️  {result['message']}"

        elif result["type"] == "response":
            # Normal response
            logger.debug(f"Pipeline execution completed successfully")
            response = result['message']

        else:
            # Error or unknown type
            logger.error(f"Unknown result type from pipeline: {result.get('type')}")
            response = f"Error: Unexpected pipeline result"

        # Detect state changes and track modifications
        self._detect_state_changes(before_state)
        self._track_data_modifications_in_memory(before_state)

        # End prompt logging only if we're not waiting for clarification
        is_waiting_for_clarification = result["type"] in (
            "clarification_needed",
            "prerequisites_needed",
            "skill_selection"
        )

        if not is_waiting_for_clarification:
            # Final response - end the turn
            turn_duration = time.time() - turn_start
            self.prompt_logger.end_turn({
                'total_llm_calls': total_llm_calls,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_duration': turn_duration
            })
            logger.info(f"Turn completed (final response)")
        else:
            # Waiting for clarification - keep turn open
            logger.info(f"Turn paused - waiting for {result['type']} response")

        return response

    async def chat_stream(self, user_message: str, execute_code: bool = True) -> AsyncIterator[str]:
        """
        Stream agent response in real-time.

        Parameters
        ----------
        user_message : str
            User's question or request
        execute_code : bool
            Whether to execute generated code (default: True)

        Yields
        ------
        str
            Response chunks as they are generated

        Examples
        --------
        >>> async for chunk in agent.chat_stream("Analyze cell types"):
        ...     print(chunk, end="", flush=True)
        """
        logger.info(f"Streaming response for: {user_message[:100]}...")

        # Add user message to memory
        self.memory.add_user_message(user_message)

        # Build context
        context = self._build_context()
        prompt = self._build_prompt(user_message, context)

        # Stream LLM response
        full_response = ""
        async for chunk in self.llm.stream(prompt):
            full_response += chunk
            yield chunk

        # Extract and execute code if present
        if execute_code:
            code_blocks = self._extract_code_blocks(full_response)
            if code_blocks:
                yield "\n\n**Executing code...**\n\n"
                for code in code_blocks:
                    result = await self._execute_code(code)
                    if result.success:
                        yield f"```\n{result.stdout}\n```\n"
                    else:
                        yield f"**Error:** {result.error}\n"

        # Add to memory
        self.memory.add_assistant_message(full_response)

    def _log_clarification_response(self, user_message: str) -> None:
        """
        Log clarification response to current turn.

        This helper method handles logging for all 3 clarification types:
        - Planner clarifications
        - Verifier clarifications
        - Skill selection

        Parameters
        ----------
        user_message : str
            User's clarification response
        """
        clarification_type = self.clarification_context.get_pending_type()

        if clarification_type == 'planner':
            question = self.clarification_context._last_planner_question or ""
            self.prompt_logger.log_clarification(
                question=question,
                response=user_message,
                context_type="planner"
            )
        elif clarification_type == 'verifier':
            questions = self.clarification_context.get_last_verifier_questions()
            question = questions[0] if questions else "Prerequisites needed"
            self.prompt_logger.log_clarification(
                question=question,
                response=user_message,
                context_type="verifier"
            )
        elif clarification_type == 'skill_selection':
            pending = self.clarification_context.get_pending_skill_selection()
            if pending:
                options = pending.get('skill_options', [])
                question = f"Select skill from: {', '.join(options)}"
                self.prompt_logger.log_clarification(
                    question=question,
                    response=user_message,
                    context_type="skill_selection"
                )

    async def chat_with_events(self, user_message: str, execute_code: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """
        Process user message and yield events for real-time progress updates.

        This method provides visibility into pipeline execution:
        - Pipeline stage transitions
        - Clarification requests
        - Execution progress
        - Step completion with plots

        Parameters
        ----------
        user_message : str
            User's question or request
        execute_code : bool
            Whether to execute generated code (default: True)

        Yields
        ------
        Dict[str, Any]
            Events with type and data:
            - {'type': 'pipeline_start', 'query': str}
            - {'type': 'clarification_needed', 'question': str}
            - {'type': 'skill_selection', 'options': list}
            - {'type': 'prerequisites_needed', 'questions': list}
            - {'type': 'execution_complete', 'message': str}
            - {'type': 'warning', 'message': str}
            - {'type': 'state_changes', 'changes': dict} - ALWAYS emitted after execution completes
            - {'type': 'pipeline_complete', 'final_response': str, 'plots': list}

        Examples
        --------
        >>> async for event in agent.chat_with_events("Annotate both slices"):
        ...     if event['type'] == 'execution_complete':
        ...         print("Execution finished...")
        ...     elif event['type'] == 'final_response':
        ...         print(f"Done: {event['message']}")
        """
        logger.info(f"chat_with_events: Processing query: {user_message[:100]}...")

        # NEW: Delegate to ConversationOrchestrator if available
        if self.conversation_orchestrator is not None:
            async for event in self._chat_with_events_via_orchestrator(user_message, execute_code):
                yield event
            return

        # OLD: Fall back to legacy pipeline implementation

        # Check if this is a clarification response BEFORE starting new turn
        is_clarification_response = self.clarification_context.is_waiting_for_clarification()

        if is_clarification_response:
            # This is a clarification response - don't start new turn
            logger.info("Detected clarification response - continuing existing turn")
            # Log the clarification Q&A to existing turn
            self._log_clarification_response(user_message)
        else:
            # New turn - start logging
            logger.info("Starting new conversation turn")
            self.prompt_logger.start_turn(user_message)

        # Track timing for logging
        turn_start = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_llm_calls = 0

        # State change tracking: take before_state ONLY on new turns
        if not is_clarification_response:
            self._clear_state_changes()
            self._turn_before_state = self._get_state_snapshot()
        before_state = self._turn_before_state

        # Add user message to memory
        self.memory.add_user_message(user_message)

        # Emit start event
        yield {
            'type': 'pipeline_start',
            'query': user_message
        }

        # Check if this is a clarification response
        clarification_response = None
        if self.clarification_context.is_waiting_for_clarification():
            # User is responding to a previous question
            clarification_response = user_message
            logger.debug(f"Detected clarification response to previous question")

        # Process through pipeline with event streaming
        last_event_type = None
        async for event in self._handle_with_pipeline_events(
            user_message,
            execute_code=execute_code,
            clarification_response=clarification_response,
            before_state=before_state  # Pass before_state for change detection
        ):
            # Track last event type to determine if waiting for clarification
            if 'type' in event:
                last_event_type = event['type']
            # Forward all events to caller
            yield event

        # Check if we're waiting for clarification
        is_waiting_for_clarification = last_event_type in (
            "clarification_needed",
            "prerequisites_needed",
            "skill_selection"
        )

        # CRITICAL: Only detect state changes if execution actually completed
        # Don't detect when waiting for clarification (no code executed yet)
        if not is_waiting_for_clarification:
            # Execution completed - detect state changes
            self._detect_state_changes(before_state)
            self._track_data_modifications_in_memory(before_state)
            logger.info(f"State changes detected: {self._state_changes}")

            # Yield state changes event for frontend synchronization
            yield {
                'type': 'state_changes',
                'changes': self._state_changes.copy()
            }

            # Final response - end the turn
            turn_duration = time.time() - turn_start
            self.prompt_logger.end_turn({
                'total_llm_calls': total_llm_calls,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_duration': turn_duration
            })
            logger.info("chat_with_events: Turn completed (final response)")
        else:
            # Waiting for clarification - keep turn open, don't detect state changes yet
            logger.info(f"chat_with_events: Turn paused - waiting for {last_event_type} response")

        logger.info("chat_with_events: Completed successfully")

    async def _chat_with_events_via_orchestrator(
        self,
        user_message: str,
        execute_code: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        NEW: Delegate to ConversationOrchestrator with logging and state detection.

        This method wraps the orchestrator to preserve:
        - Prompt logging
        - Memory management
        - State change detection
        - Event streaming

        Parameters
        ----------
        user_message : str
            User's message
        execute_code : bool
            Whether to execute code

        Yields
        ------
        Dict[str, Any]
            Events from orchestrator
        """
        logger.info(f"_chat_with_events_via_orchestrator: Using new architecture")

        # Check if this is a clarification response BEFORE starting new turn
        is_clarification_response = self.clarification_context.is_waiting_for_clarification()

        if is_clarification_response:
            # This is a clarification response - don't start new turn
            logger.info("Detected clarification response - continuing existing turn")
            # Log the clarification Q&A to existing turn
            self._log_clarification_response(user_message)
        else:
            # New turn - start logging
            logger.info("Starting new conversation turn")
            self.prompt_logger.start_turn(user_message)

        # Track timing for logging
        turn_start = time.time()

        # State change tracking: take before_state ONLY on new turns.
        # For clarification responses, reuse the before_state from the original turn
        # so that changes from ALL steps (across multiple calls) are detected.
        if not is_clarification_response:
            self._clear_state_changes()
            self._turn_before_state = self._get_state_snapshot()
        # else: keep existing self._turn_before_state from the original call

        before_state = self._turn_before_state

        # Add user message to memory
        self.memory.add_user_message(user_message)

        # Emit start event
        yield {
            'type': 'pipeline_start',
            'query': user_message
        }

        # Delegate to orchestrator
        last_event_type = None
        async for event in self.conversation_orchestrator.handle_turn_with_events(
            user_message,
            execute_code=execute_code
        ):
            # Track last event type
            if 'type' in event:
                last_event_type = event['type']
            # Forward all events
            yield event

        # Check if we're waiting for clarification
        is_waiting_for_clarification = last_event_type in (
            "clarification_needed",
            "prerequisites_needed",
            "skill_selection"
        )

        # CRITICAL: Only detect state changes if execution actually completed
        if not is_waiting_for_clarification:
            # Execution completed - detect state changes
            self._detect_state_changes(before_state)
            self._track_data_modifications_in_memory(before_state)
            logger.info(f"State changes detected: {self._state_changes}")

            # Yield state changes event for frontend synchronization
            yield {
                'type': 'state_changes',
                'changes': self._state_changes.copy()
            }

            # Final response - end the turn
            turn_duration = time.time() - turn_start
            self.prompt_logger.end_turn({
                'total_llm_calls': 0,  # Note: Token tracking happens in LLM layer
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_duration': turn_duration
            })
            logger.info("_chat_with_events_via_orchestrator: Turn completed")
        else:
            # Waiting for clarification - keep turn open
            logger.info(f"_chat_with_events_via_orchestrator: Waiting for {last_event_type}")

        logger.info("_chat_with_events_via_orchestrator: Completed")

    async def _handle_with_llm(
        self,
        user_message: str,
        execute_code: bool = True,
        stream: bool = False,
        allow_planning: bool = True,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle user message with direct LLM interaction.

        Args:
            user_message: User's question or request
            execute_code: Whether to execute generated code
            stream: Whether to stream the response
            allow_planning: Whether to allow planning decision (prevents recursion)
            plan_context: Optional plan context for multi-step execution
                         {'is_plan_step': True, 'step_number': 1, 'total_steps': 3, 'original_query': '...'}

        Returns:
            Agent's response

        Note:
            This function does NOT add user message to memory - caller is responsible.
            It DOES add assistant response to memory automatically.
        """
        # LLM-based skill matching (if skills enabled)
        skill_guidance = ""
        matched_skill_slugs = []  # Track matched skills for execution context
        if self.enable_skills and self.skill_registry:
            logger.info("Matching relevant skills using LLM...")
            matched_skill_slugs = await self._select_skill_matches_llm(user_message, top_k=2)

            if matched_skill_slugs:
                logger.info(f"LLM matched skills: {matched_skill_slugs}")

                # Load full content for matched skills (progressive disclosure)
                skill_definitions = []
                for slug in matched_skill_slugs:
                    full_skill = self.skill_registry.load_full_skill(slug)
                    if full_skill:
                        skill_definitions.append(full_skill)
                        logger.info(f"  - Loaded skill: {full_skill.name}")

                # Format skill guidance for prompt injection
                skill_guidance = self._format_skill_guidance(skill_definitions)

        # Build context and prompt (with skill guidance and plan context if available)
        context = self._build_context()
        prompt = self._build_prompt(
            user_message,
            context,
            skill_guidance=skill_guidance,
            plan_context=plan_context
        )

        # Get LLM response
        call_start = time.time()
        if stream:
            # For streaming, collect full response
            response = ""
            async for chunk in self.llm.stream(prompt):
                response += chunk
        else:
            response = await self.llm.run(prompt)
        call_duration = time.time() - call_start

        # Log this LLM call
        self.prompt_logger.log_llm_call(
            call_type="main_agent",
            full_prompt=f"{SYSTEM_PROMPT}\n\n{prompt}",
            response=response,
            metadata={
                'model': self.llm.config.model,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': call_duration,
                'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                'output_tokens': getattr(self.llm, 'last_output_tokens', None),
                'matched_skills': matched_skill_slugs if matched_skill_slugs else None,
            }
        )

        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)

        # Execute code if requested (skipped for meta-queries)
        execution_results = []
        if execute_code and code_blocks:
            for i, code in enumerate(code_blocks):
                # Execute with retry and error reflection
                # Pass matched skills for script path resolution
                result = await self._execute_code_with_retry(
                    code, user_message, max_retries=2, matched_skill_slugs=matched_skill_slugs
                )
                execution_results.append(result)

                # Log to notebook
                if self.notebook_logger:
                    self.notebook_logger.append_code_execution(
                        code=code,
                        result=result,
                        user_query=user_message
                    )

        # Format final response
        final_response = response

        # Collect all plots from execution results
        all_plots = []
        if execution_results:
            final_response += "\n\n**Execution Results:**\n\n"
            for i, result in enumerate(execution_results, 1):
                if result.success:
                    if result.stdout:
                        final_response += f"```\n{result.stdout}\n```\n"
                    # Collect plots
                    if result.plots:
                        all_plots.extend(result.plots)
                        final_response += f"\n*({len(result.plots)} plot(s) generated)*\n"
                else:
                    final_response += f"**Error in code block {i}:** {result.error}\n"

            # Unified analysis: interpret ALL results together (not per-block)
            combined_code = "\n\n".join(code_blocks)
            combined_output = "\n".join(r.stdout for r in execution_results if r.success and r.stdout)
            if combined_output:
                combined_result = ExecutionResult(success=True, stdout=combined_output)
                interpretation = await self._interpret_results(
                    user_message, combined_code, combined_result
                )
                if interpretation:
                    final_response += f"\n**Analysis Interpretation:**\n{interpretation}\n\n"

                    finding_keywords = ['more', 'less', 'enriched', 'higher', 'lower', 'fold', 'times', 'significant']
                    if any(keyword in interpretation.lower() for keyword in finding_keywords):
                        self.memory.track_finding(interpretation[:200])

        # Store plots for retrieval (used by web interface)
        self._last_plots = all_plots

        # Generate summaries for memory (with smart summarization)
        code = "\n\n".join(code_blocks) if code_blocks else None
        exec_output = "\n".join(r.get_display_output() for r in execution_results) if execution_results else None
        metadata = {'plots': all_plots} if all_plots else {}

        # Generate message summary if response is long
        message_summary = None
        if len(final_response) > self.memory.message_summary_threshold:
            message_summary = await self.memory.create_message_summary(final_response)

        # Generate execution summary (critical for remembering data modifications)
        execution_summary = None
        if code and exec_output:
            execution_summary = await self.memory.create_execution_summary(
                user_message, code, exec_output
            )

        # Add to memory with summaries
        self.memory.add_assistant_message_with_summary(
            content=final_response,
            code=code,
            execution_result=exec_output,
            summary=message_summary,
            execution_summary=execution_summary,
            metadata=metadata
        )

        return final_response

    async def _handle_with_llm_events(
        self,
        user_message: str,
        execute_code: bool = True,
        allow_planning: bool = True,
        plan_context: Optional[Dict[str, Any]] = None,
        matched_skill_slugs: Optional[List[str]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle user message with direct LLM interaction, yielding events for progress.

        Yields execution events including real-time stdout/stderr output.

        Args:
            user_message: User's question or request
            execute_code: Whether to execute generated code
            allow_planning: Whether to allow planning decision
            plan_context: Optional plan context for multi-step execution
            matched_skill_slugs: Pre-matched skill slugs from pipeline (if provided, skip matching)

        Yields:
            Dict events with type and data
        """
        # LLM-based skill matching (if skills enabled and not already provided)
        if matched_skill_slugs is None:
            matched_skill_slugs = []
            if self.enable_skills and self.skill_registry:
                logger.info("Matching relevant skills using LLM...")
                matched_skill_slugs = await self._select_skill_matches_llm(user_message, top_k=2)

                if matched_skill_slugs:
                    logger.info(f"LLM matched skills: {matched_skill_slugs}")
        else:
            # Skills already matched by pipeline - use them
            if matched_skill_slugs:
                logger.info(f"Using pre-matched skills from pipeline: {matched_skill_slugs}")

        # Build context and prompt
        context = self._build_context()

        # Load skill guidance if skills matched
        skill_guidance = ""
        if matched_skill_slugs and self.skill_registry:
            skill_definitions = []
            for slug in matched_skill_slugs:
                full_skill = self.skill_registry.load_full_skill(slug)
                if full_skill:
                    skill_definitions.append(full_skill)
            skill_guidance = self._format_skill_guidance(skill_definitions)

        prompt = self._build_prompt(
            user_message,
            context,
            skill_guidance=skill_guidance,
            plan_context=plan_context
        )

        # Get LLM response
        call_start = time.time()
        response = await self.llm.run(prompt)
        call_duration = time.time() - call_start

        # Log this LLM call
        self.prompt_logger.log_llm_call(
            call_type="main_agent",
            full_prompt=f"{SYSTEM_PROMPT}\n\n{prompt}",
            response=response,
            metadata={
                'model': self.llm.config.model,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': call_duration,
                'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                'output_tokens': getattr(self.llm, 'last_output_tokens', None),
            }
        )

        # Parse response into sequential segments
        segments, interpret_instruction = self._parse_response_segments(response)
        logger.info(f"Parsed {len(segments)} segments, interpret={bool(interpret_instruction)}")

        # Process segments sequentially
        execution_results = []
        all_plots = []
        explanation_text = ""

        for seg_idx, (seg_type, content) in enumerate(segments):
            if seg_type == 'text':
                # Yield text segment
                explanation_text += content + "\n"
                yield {
                    'type': 'agent_text',
                    'text': content
                }
            elif seg_type == 'code' and execute_code:
                # Execute code block with streaming
                logger.info(f"Executing code block...")
                async for event in self._execute_code_with_events(content, user_message, matched_skill_slugs):
                    if event['type'] == 'execution_output':
                        yield event
                    elif event['type'] == 'reflection_start':
                        yield event
                    elif event['type'] == 'reflection_complete':
                        # Update segment with fixed code so analysis/memory see corrected version
                        if event.get('fixed_code'):
                            segments[seg_idx] = ('code', event['fixed_code'])
                        yield event
                    elif event['type'] == 'execution_result':
                        result = event['result']
                        execution_results.append(result)
                        if result.plots:
                            all_plots.extend(result.plots)

                # Yield code block complete event
                if execution_results:
                    last_result = execution_results[-1]
                    yield {
                        'type': 'code_block_complete',
                        'success': last_result.success,
                        'plots': last_result.plots if last_result.plots else [],
                        'stdout': last_result.stdout if last_result.stdout else ""
                    }

        # Collect code blocks from segments for memory
        code_blocks = [content for seg_type, content in segments if seg_type == 'code']

        # Always analyze execution results after code execution (replaces old summarization)
        analysis = None
        if execution_results:
            logger.info("Analyzing execution results...")
            analysis = await self._analyze_execution_results(
                user_query=user_message,
                segments=segments,
                execution_results=execution_results,
                interpret_instruction=interpret_instruction
            )
            logger.info(f"Analysis complete: {analysis.get('response_summary', 'N/A')[:100]}")

            # Yield execution issue warning if detected
            execution_issues = analysis.get('execution_issues', {})
            if execution_issues.get('has_issues'):
                yield {
                    'type': 'execution_issue',
                    'issue_type': execution_issues.get('issue_type'),
                    'explanation': execution_issues.get('explanation')
                }

            # Yield interpretation to frontend if present
            if analysis.get('interpretation'):
                yield {
                    'type': 'agent_text',
                    'text': f"**Analysis:**\n{analysis['interpretation']}"
                }

            # Track findings if present
            if analysis.get('key_findings'):
                for finding in analysis['key_findings']:
                    if finding:
                        self.memory.track_finding(finding[:200])

        # Format final response for memory (with corrected code + truncated outputs)
        logger.info("Formatting final response for memory...")
        # Rebuild from segments (which have been updated with fixed code if reflection succeeded)
        final_response = ""
        result_idx = 0
        for seg_type, content in segments:
            if seg_type == 'text':
                final_response += content + "\n\n"
            elif seg_type == 'code':
                final_response += f"```python\n{content}\n```\n\n"
                # Add corresponding output
                if result_idx < len(execution_results):
                    result = execution_results[result_idx]
                    if result.success:
                        if result.stdout:
                            truncated = result.stdout[:1000]
                            if len(result.stdout) > 1000:
                                truncated += "\n... (output truncated)"
                            final_response += f"```\n{truncated}\n```\n"
                        if result.plots:
                            final_response += f"*({len(result.plots)} plot(s) generated)*\n"
                    else:
                        final_response += f"**Error:** {result.error}\n"
                    result_idx += 1

        # Append analysis interpretation if present
        if analysis and analysis.get('interpretation'):
            final_response += f"\n**Analysis:**\n{analysis['interpretation']}\n"

        # Store plots
        self._last_plots = all_plots
        logger.info(f"Stored {len(all_plots)} plots")

        # Build summaries from analysis (replaces old summarization system)
        logger.info("Building summaries from analysis...")
        code = "\n\n".join(code_blocks) if code_blocks else None
        exec_output = "\n".join(r.get_display_output() for r in execution_results) if execution_results else None
        metadata = {'plots': all_plots} if all_plots else {}

        # Use analysis results for summaries
        message_summary = None
        execution_summary = None

        if analysis:
            # Message summary from analysis
            message_summary = analysis.get('response_summary')

            # Execution summary: data_changes + findings
            execution_summary = analysis.get('data_changes', '')
            if analysis.get('key_findings'):
                findings_text = "; ".join(analysis['key_findings'])
                execution_summary += f"\n**Findings:** {findings_text}"

            logger.info(f"Using analysis-based summaries")

        # Add to memory
        logger.info("Adding to memory...")
        self.memory.add_assistant_message_with_summary(
            content=final_response,
            code=code,
            execution_result=exec_output,
            summary=message_summary,
            execution_summary=execution_summary,
            metadata=metadata,
            query=user_message
        )
        logger.info("Added to memory")

        # Yield completion event
        logger.info("Yielding execution_complete event...")
        # CRITICAL: Include execution success status for reliable error detection
        # Don't rely on text parsing - use actual execution results
        had_execution_error = any(not r.success for r in execution_results) if execution_results else False
        yield {
            'type': 'execution_complete',
            'response': final_response,
            'plots': all_plots,
            'success': not had_execution_error  # Explicit success flag (bool, JSON serializable)
        }
        logger.info("Yielded execution_complete event")

    def _build_context(self) -> Dict[str, Any]:
        """Build context dictionary for LLM."""
        context = {
            "has_session": self.session is not None,
            "has_data": self.session and self.session.has_data
        }

        if self.session and self.session.has_data:
            summary = self.session.get_summary()

            # Compute totals from slices
            total_cells = sum(s['n_obs'] for s in summary['slices'])
            total_genes = max((s['n_vars'] for s in summary['slices']), default=0)

            # Check if any slice has celltypes
            has_celltypes = any(s['has_celltype'] for s in summary['slices'])

            context.update({
                "n_cells": total_cells,
                "n_genes": total_genes,
                "has_celltypes": has_celltypes,
                "n_rois": summary.get('n_rois', 0)
            })

        return context

    def _build_prompt(
        self,
        user_message: str,
        context: Dict[str, Any],
        skill_guidance: str = "",
        plan_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for LLM with context and optional skill guidance.

        Args:
            user_message: Current user query or step description
            context: Session context dictionary
            skill_guidance: Skill instructions (if skill matched)
            plan_context: Optional plan context for multi-step execution
                         {'is_plan_step': True, 'step_number': 1, 'total_steps': 3, 'original_query': '...'}
        """
        # Add plan context at the top if this is a plan step
        prompt_parts = []

        if plan_context and plan_context.get('is_plan_step'):
            prompt_parts.append("=" * 80)
            prompt_parts.append("**MULTI-STEP PLAN EXECUTION**")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
            prompt_parts.append(f"**Original Task:** {plan_context.get('original_query', 'N/A')}")
            prompt_parts.append("")
            prompt_parts.append(
                f"**Current Step:** [Step {plan_context.get('step_number', '?')} of "
                f"{plan_context.get('total_steps', '?')}] {user_message}"
            )
            prompt_parts.append("")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")

        # Add session context

        if context.get("has_data"):
            # Get detailed summary
            summary = self.session.get_summary()

            # Compute data format from slices
            n_slices = summary.get('n_slices', 0)
            has_protein = 'protein' in summary.get('modalities', [])

            if n_slices > 1 and has_protein:
                data_format = 'multi_slice_multi_omics'
            elif n_slices > 1:
                data_format = 'multi_slice'
            elif has_protein:
                data_format = 'multi_omics'
            else:
                data_format = 'single_slice'

            prompt_parts.append("**Current Session:**")

            # Show data format prominently
            if data_format == 'single_slice':
                prompt_parts.append(f"- Data Format: Single slice")
            elif data_format == 'multi_slice':
                prompt_parts.append(f"- Data Format: Multi-slice ({n_slices} slices)")
            elif data_format == 'multi_omics':
                prompt_parts.append(f"- Data Format: Multi-omics (gene + protein)")

            prompt_parts.append(f"- Cells/spots: {context['n_cells']:,}")

            # Add actual ROI names (get from session directly)
            roi_names = list(self.session.roi_subsets.keys())
            if roi_names:
                prompt_parts.append(f"- Available ROIs: {', '.join(roi_names)}")
            else:
                prompt_parts.append(f"- ROIs: None defined yet")

            # Multi-slice information (NEW: uses summary['slices'])
            if data_format == 'multi_slice' and summary.get('slices'):
                prompt_parts.append("")
                prompt_parts.append("**Available Slices:**")
                for slice_info in summary['slices']:
                    slice_id = slice_info['slice_id']
                    tissue = slice_info['tissue_name']
                    n_obs = slice_info['n_obs']
                    modality = slice_info['modality']
                    data_level = slice_info['data_level']

                    prompt_parts.append(
                        f"- Slice {slice_id}: {tissue} "
                        f"({n_obs:,} observations, {modality}, {data_level}-level, {slice_info['n_vars']:,} genes)"
                    )
                    prompt_parts.append(
                        f"  Access: session.get_slice({slice_id}).adata"
                    )

                    # Show celltypes for this slice if available
                    celltypes = slice_info.get('celltypes', [])
                    if celltypes:
                        prompt_parts.append(f"  Cell types: {', '.join(celltypes)}")
            elif data_format == 'single_slice':
                # Single slice: show celltype info here
                slice_0 = self.session.get_slice(0) if summary.get('slices') else None
                if slice_0:
                    data_level = slice_0.data_level
                    prompt_parts.append(f"- Data level: {data_level}")
                    prompt_parts.append(f"- Genes/features: {slice_0.adata.n_vars:,}")

                    celltypes = slice_0.adata.obs.get('celltype')
                    if celltypes is not None and celltypes.notna().any():
                        unique_celltypes = sorted(celltypes.unique().tolist())
                        prompt_parts.append(f"- Cell types: {', '.join(unique_celltypes)}")

            # Multi-omics information (simplified)
            if data_format == 'multi_omics':
                prompt_parts.append("")
                prompt_parts.append("**Multi-Omics Data:**")
                # Get gene and protein slices
                gene_slices = [s for s in summary.get('slices', []) if s['modality'] == 'gene']
                protein_slices = [s for s in summary.get('slices', []) if s['modality'] == 'protein']

                if gene_slices:
                    gene_slice = gene_slices[0]
                    prompt_parts.append(f"- Gene data: {gene_slice['n_vars']} features, {gene_slice['n_obs']} cells")
                    prompt_parts.append(f"  Access: session.get_slice({gene_slice['slice_id']}).adata")

                if protein_slices:
                    protein_slice = protein_slices[0]
                    prompt_parts.append(f"- Protein data: {protein_slice['n_vars']} features, {protein_slice['n_obs']} cells")
                    prompt_parts.append(f"  Access: session.get_slice({protein_slice['slice_id']}).adata")


            prompt_parts.append("")  # Empty line after session info

        # Add entity context (ROIs created, recent findings)
        entity_context = self.memory.get_entity_context_string()
        if entity_context:
            prompt_parts.append(entity_context)
            prompt_parts.append("")  # Empty line after entity context

        if not context.get("has_data"):
            prompt_parts.append("**Note:** No data loaded yet. Guide the user to load data first.\n")

        # Add conversation history using new smart summarization
        conversation_history = self.memory.get_history_string()
        if conversation_history:
            prompt_parts.append("**Conversation History:**")
            prompt_parts.append(conversation_history)
            prompt_parts.append("**[End of Conversation History]**")
            prompt_parts.append("")
        if skill_guidance:
            prompt_parts.append("=" * 80)
            prompt_parts.append("**IMPORTANT: SKILL GUIDANCE PROVIDED**")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
            prompt_parts.append("You have been provided with specialized skill guidance below.")
            prompt_parts.append("Follow these instructions EXACTLY to complete the user's request.")
            prompt_parts.append("The skill provides a proven workflow - use it step-by-step.")
            prompt_parts.append("")
            prompt_parts.append(skill_guidance)
            prompt_parts.append("")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")

        # Add current query
        prompt_parts.append(f"**User Query:** {user_message}\n")
        prompt_parts.append(
            "Please provide a helpful response. If code is needed, "
            "wrap it in ```python code fences."
        )

        return "\n".join(prompt_parts)

    async def _execute_code(self, code: str, matched_skill_slugs: List[str] = None) -> ExecutionResult:
        """Execute code with session in namespace and persistent skill imports.

        IMPORTANT: Skill paths are added persistently for entire conversation to support multi-turn.
        Skills use unique package names (celltype_fast, celltype_scvi, niche_analysis_lib) to avoid conflicts.
        """
        import sys
        import os
        from pathlib import Path

        namespace = {}

        if self.session:
            namespace['session'] = self.session
            if self.session.has_data:
                # Convenient shortcut for single-slice access
                slice_0 = self.session.get_slice(0)
                if slice_0:
                    namespace['adata'] = slice_0.adata

        # Add common imports
        namespace['np'] = __import__('numpy')
        namespace['pd'] = __import__('pandas')

        # PERSISTENT skill loading: Add to sys.path and PYTHONPATH for entire conversation
        if matched_skill_slugs and self.skill_registry:
            skill_paths_to_log = {}
            for slug in matched_skill_slugs:
                skill_metadata = self.skill_registry.skill_metadata.get(slug.lower())
                if skill_metadata:
                    # Add skill directory to sys.path so "from celltype_fast import X" works
                    skill_dir = skill_metadata.path
                    skill_path = str(skill_dir)

                    # Only add if not already loaded
                    if skill_path not in self._loaded_skill_paths:
                        sys.path.insert(0, skill_path)

                        # CRITICAL: Also add to PYTHONPATH for Ray/multiprocessing workers
                        current_pythonpath = os.environ.get('PYTHONPATH', '')
                        if skill_path not in current_pythonpath:
                            if current_pythonpath:
                                os.environ['PYTHONPATH'] = skill_path + ':' + current_pythonpath
                            else:
                                os.environ['PYTHONPATH'] = skill_path
                            logger.info(f"Added '{slug}' to PYTHONPATH for Ray workers: {skill_path}")

                        self._loaded_skill_paths.add(skill_path)
                        skill_paths_to_log[slug] = skill_path
                        logger.info(f"Loaded skill '{slug}' persistently: {skill_path}")
                    else:
                        logger.debug(f"Skill '{slug}' already loaded (persistent): {skill_path}")

        # Execute (NO finally block - paths stay loaded for multi-turn!)
        result = await asyncio.to_thread(
            self.executor.execute,
            code,
            namespace=namespace
        )

        return result

    async def _execute_code_with_events(
        self,
        code: str,
        user_query: str,
        matched_skill_slugs: List[str] = None,
        max_retries: int = 2
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute code with real-time stdout/stderr streaming AND error reflection/retry.

        Yields events for:
        - execution_output: Real-time stdout/stderr lines
        - execution_result: Final execution result
        - reflection_start: Error reflection started
        - reflection_complete: Error reflection generated fix

        Args:
            code: Python code to execute
            user_query: Original user question (for reflection context)
            matched_skill_slugs: Skills to load for execution
            max_retries: Maximum number of retry attempts (default: 2)

        Yields:
            Dict events with type and data
        """
        import sys
        from pathlib import Path

        # Load skills ONCE before retry loop (skills are persistent via sys.path)
        if matched_skill_slugs and self.skill_registry:
            for slug in matched_skill_slugs:
                skill_metadata = self.skill_registry.skill_metadata.get(slug.lower())
                if skill_metadata:
                    skill_dir = skill_metadata.path
                    skill_path = str(skill_dir)

                    if skill_path not in self._loaded_skill_paths:
                        sys.path.insert(0, skill_path)

                        # CRITICAL: Also add to PYTHONPATH for Ray/multiprocessing workers
                        import os
                        current_pythonpath = os.environ.get('PYTHONPATH', '')
                        if skill_path not in current_pythonpath:
                            if current_pythonpath:
                                os.environ['PYTHONPATH'] = skill_path + ':' + current_pythonpath
                            else:
                                os.environ['PYTHONPATH'] = skill_path
                            logger.info(f"Added '{slug}' to PYTHONPATH for Ray workers: {skill_path}")

                        self._loaded_skill_paths.add(skill_path)
                        logger.info(f"Loaded skill '{slug}' persistently: {skill_path}")
                    else:
                        logger.debug(f"Skill '{slug}' already loaded (persistent): {skill_path}")

        attempt = 0
        current_code = code

        while attempt <= max_retries:
            attempt += 1

            # Build namespace for execution
            namespace = {}

            if self.session:
                namespace['session'] = self.session
                if self.session.has_data:
                    # Convenient shortcut for single-slice access
                    slice_0 = self.session.get_slice(0)
                    if slice_0:
                        namespace['adata'] = slice_0.adata

            # Add common imports
            namespace['np'] = __import__('numpy')
            namespace['pd'] = __import__('pandas')

            # Execute with streaming (yields output lines in real-time)
            execution_result = None
            async for event in self._execute_with_streaming(current_code, namespace):
                if event['type'] == 'execution_result':
                    execution_result = event['result']
                    # DON'T yield execution_result yet - only yield after final attempt
                    # This prevents caller from seeing failed attempts that get fixed by reflection
                else:
                    yield event  # Yield output events immediately

            # Check if execution succeeded
            if execution_result and execution_result.success:
                if attempt > 1:
                    logger.info(f"Code execution succeeded after {attempt - 1} fix(es)")

                # Log successful execution to notebook (only correct code, no failed attempts)
                context = None
                if hasattr(self, '_current_user_query'):
                    user_query = self._current_user_query
                else:
                    user_query = None

                self.notebook_logger.append_code_execution(
                    code=current_code,
                    result=execution_result,
                    user_query=user_query,
                    context=context
                )

                # Yield the successful result
                yield {
                    'type': 'execution_result',
                    'result': execution_result
                }
                return  # Success - stop retrying

            # Execution failed
            logger.warning(f"Code execution failed (attempt {attempt}/{max_retries + 1}): {execution_result.error if execution_result else 'Unknown error'}")

            # Clean up any matplotlib figures from failed execution
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                logger.debug("Closed all matplotlib figures from failed execution")
            except ImportError:
                pass

            # If this was our last attempt, return failure
            if attempt > max_retries:
                logger.info("Max retries reached, returning failure")

                # Yield the final failed result
                yield {
                    'type': 'execution_result',
                    'result': execution_result
                }
                return

            # Check if error is fixable
            if not should_attempt_fix(execution_result):
                logger.info("Error is unfixable, returning failure")

                # Yield the final failed result
                yield {
                    'type': 'execution_result',
                    'result': execution_result
                }
                return

            # Yield reflection start event
            yield {
                'type': 'reflection_start',
                'attempt': attempt,
                'error': execution_result.error if execution_result else 'Unknown error'
            }

            logger.info("Attempting error reflection to fix code...")

            # Gather context for reflection
            available_context = {
                'available_rois': self.tools.get_all_rois_summary(),  # Better: includes slice_id/modality
                'available_columns': self.tools.list_available_columns() if self.session and self.session.has_data else [],
                'has_celltype': self.tools.column_exists('celltype') if self.session and self.session.has_data else False
            }

            # Reflect on error and get fix
            reflection = await reflect_on_error_and_fix(
                llm_backend=self.llm,
                original_code=current_code,
                error_result=execution_result,
                user_query=user_query,
                available_context=available_context,
                max_attempts=max_retries
            )

            # Check if reflection suggests a fix
            change_magnitude = reflection.get('change_magnitude', 'unknown')
            if not reflection['should_retry'] or not reflection['fixed_code']:
                reason = "large change required" if change_magnitude == 'large' else "not retrying"
                logger.info(f"Error reflection suggests {reason}")

                # Yield the final failed result
                yield {
                    'type': 'execution_result',
                    'result': execution_result
                }
                return

            logger.info(
                f"Error reflection (magnitude={change_magnitude}, confidence={reflection['confidence']:.2f}): "
                f"{reflection['reasoning']}"
            )

            # Yield reflection complete event
            yield {
                'type': 'reflection_complete',
                'reasoning': reflection['reasoning'],
                'confidence': reflection['confidence'],
                'fixed_code': reflection['fixed_code'],
                'change_magnitude': change_magnitude
            }

            # Update code with fix and retry
            current_code = reflection['fixed_code']

    async def _execute_with_streaming(
        self,
        code: str,
        namespace: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute code and stream stdout/stderr in real-time.

        This is the core streaming execution method that captures output line-by-line.

        IMPORTANT: Uses executor's persistent namespace to maintain variables across
        multiple code blocks in a single response.

        Yields:
            execution_output events (one per line) and final execution_result event
        """
        import sys
        import io
        import threading
        from queue import Queue, Empty

        # Clean code (remove markdown fences if present)
        code = self.executor._clean_code(code)

        # CRITICAL: Merge with persistent namespace to maintain variables across blocks
        # Start with persistent vars, then overlay fresh vars (session, np, pd)
        exec_namespace = self.executor.namespace.copy()
        exec_namespace.update(namespace)

        # Create queues for streaming output
        stdout_queue = Queue()
        stderr_queue = Queue()

        # Custom stdout/stderr that writes to queues
        class StreamToQueue(io.StringIO):
            def __init__(self, queue, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.queue = queue
                self.buffer = ""

            def write(self, text):
                super().write(text)
                self.buffer += text
                # Handle \r (carriage return) used by tqdm/progress bars:
                # Keep only content after the last \r (mimics terminal overwrite)
                while '\r' in self.buffer:
                    # Don't treat \r\n as carriage return
                    cr_pos = self.buffer.rfind('\r')
                    if cr_pos + 1 < len(self.buffer) and self.buffer[cr_pos + 1] == '\n':
                        break
                    self.buffer = self.buffer[cr_pos + 1:]
                    break
                # Yield complete lines
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    self.queue.put(line + '\n')

        captured_stdout = StreamToQueue(stdout_queue)
        captured_stderr = StreamToQueue(stderr_queue)

        # Execution result container
        result_container = {'result': None, 'exception': None}

        # Execute in thread to avoid blocking
        def execute_thread():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = captured_stdout
                sys.stderr = captured_stderr

                # Execute code with persistent namespace
                exec(code, exec_namespace)

                # CRITICAL: Update persistent namespace with new variables (like CodeExecutor.execute())
                # This allows variables from this block to be available in next block
                for key, value in exec_namespace.items():
                    if not key.startswith('_') and key not in ['__builtins__']:
                        self.executor.namespace[key] = value

                # Success
                from .executor import ExecutionResult
                result_container['result'] = ExecutionResult(
                    success=True,
                    stdout=captured_stdout.getvalue(),
                    stderr=captured_stderr.getvalue()
                )

                # Capture plots
                result_container['result'].plots = self.executor._capture_plots()

            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()

                from .executor import ExecutionResult
                result_container['result'] = ExecutionResult(
                    success=False,
                    error=error_msg,
                    stdout=captured_stdout.getvalue(),
                    stderr=traceback_str
                )

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                # Signal completion
                stdout_queue.put(None)
                stderr_queue.put(None)

        # Start execution thread
        thread = threading.Thread(target=execute_thread, daemon=True)
        thread.start()

        # Stream output as it arrives
        # Track completion of both stdout and stderr separately
        stdout_done = False
        stderr_done = False

        while not (stdout_done and stderr_done):
            # Check stdout queue
            try:
                line = stdout_queue.get(timeout=0.01)
                if line is None:
                    stdout_done = True  # stdout completed
                else:
                    yield {
                        'type': 'execution_output',
                        'line': line,
                        'stream': 'stdout'
                    }
            except Empty:
                pass

            # Check stderr queue
            try:
                line = stderr_queue.get(timeout=0.01)
                if line is None:
                    stderr_done = True  # stderr completed
                elif line is not None:
                    yield {
                        'type': 'execution_output',
                        'line': line,
                        'stream': 'stderr'
                    }
            except Empty:
                pass

            # Check if thread is still alive
            if not thread.is_alive():
                # Drain remaining output from both queues
                while not stdout_queue.empty():
                    line = stdout_queue.get()
                    if line is None:
                        stdout_done = True
                    elif line is not None:
                        yield {
                            'type': 'execution_output',
                            'line': line,
                            'stream': 'stdout'
                        }
                while not stderr_queue.empty():
                    line = stderr_queue.get()
                    if line is None:
                        stderr_done = True
                    elif line is not None:
                        yield {
                            'type': 'execution_output',
                            'line': line,
                            'stream': 'stderr'
                        }
                break

        # Wait for thread to complete
        thread.join(timeout=1.0)

        # Yield final result
        yield {
            'type': 'execution_result',
            'result': result_container['result']
        }

    async def _execute_code_with_retry(
        self,
        code: str,
        user_query: str,
        max_retries: int = 2,
        matched_skill_slugs: List[str] = None
    ) -> ExecutionResult:
        """
        Execute code with automatic error reflection and retry.

        If execution fails, agent analyzes the error, attempts to fix it,
        and retries. This gives the agent resilience and self-correction ability.

        Args:
            code: Python code to execute
            user_query: Original user question (for context)
            max_retries: Maximum number of retry attempts

        Returns:
            ExecutionResult (success or final failure)
        """
        attempt = 0
        current_code = code
        reflection_log = []

        while attempt <= max_retries:
            attempt += 1

            # Execute code
            result = await self._execute_code(current_code, matched_skill_slugs=matched_skill_slugs or [])

            # Success - return immediately
            if result.success:
                if reflection_log:
                    logger.info(f"Code execution succeeded after {len(reflection_log)} fix(es)")
                return result

            # Failure - check if we should attempt to fix
            logger.warning(f"Code execution failed (attempt {attempt}/{max_retries + 1}): {result.error}")

            # Clean up any matplotlib figures from failed execution to prevent plot accumulation
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                logger.debug("Closed all matplotlib figures from failed execution")
            except ImportError:
                pass  # matplotlib not available

            # If this was our last attempt, return failure
            if attempt > max_retries:
                logger.info("Max retries reached, returning failure")
                return result

            # Check if error is fixable
            if not should_attempt_fix(result):
                logger.info("Error is unfixable, returning failure")
                return result

            # Attempt to fix via error reflection
            logger.info("Attempting error reflection to fix code...")

            # Gather context for reflection
            available_context = {
                'available_rois': self.tools.get_all_rois_summary(),  # Better: includes slice_id/modality
                'available_columns': self.tools.list_available_columns() if self.session and self.session.has_data else [],
                'has_celltype': self.tools.column_exists('celltype') if self.session and self.session.has_data else False
            }

            # Reflect on error and get fix
            reflection = await reflect_on_error_and_fix(
                llm_backend=self.llm,
                original_code=current_code,
                error_result=result,
                user_query=user_query,
                available_context=available_context,
                max_attempts=max_retries
            )

            # Check if reflection suggests a fix
            if not reflection['should_retry'] or not reflection['fixed_code']:
                logger.info("Error reflection suggests not retrying")
                return result

            # Log the fix attempt
            reflection_log.append({
                'attempt': attempt,
                'reasoning': reflection['reasoning'],
                'confidence': reflection['confidence']
            })

            logger.info(
                f"Error reflection (confidence={reflection['confidence']:.2f}): "
                f"{reflection['reasoning']}"
            )

            # Update code with fix
            current_code = reflection['fixed_code']

            # Continue to next iteration (retry with fixed code)

        # Should not reach here, but return last result just in case
        return result

    async def _interpret_results(
        self,
        user_query: str,
        code: str,
        result: ExecutionResult
    ) -> str:
        """
        Interpret execution results in biological context.

        This is KEY for intelligence: Don't just execute code - UNDERSTAND what results mean!

        Args:
            user_query: Original user question
            code: Code that was executed
            result: Execution result with stdout/stderr

        Returns:
            Natural language interpretation of results
        """
        if not result.stdout and not result.stderr:
            return ""

        interpret_prompt = f"""You are a spatial transcriptomics expert interpreting analysis results.

User Query: "{user_query}"

Code Executed:
```python
{code}
```

Execution Output:
{result.stdout if result.stdout else '(No output)'}

Your task:
1. Interpret what the results mean in biological context
2. Answer the user's original question based on the output
3. Highlight key findings (e.g., "ROI_1 has 3x more malignant cells")
4. Suggest follow-up analyses if relevant

Provide a clear, concise interpretation (2-4 sentences):
"""

        try:
            interpretation = await self.llm.run(interpret_prompt)
            return interpretation.strip()
        except Exception as e:
            logger.warning(f"Result interpretation failed: {e}")
            return ""  # Fallback: show raw output only

    def _parse_response_segments(self, response: str) -> tuple[list[tuple[str, str]], str]:
        """
        Parse LLM response into sequential segments of text and code.

        Returns:
            (segments, interpret_instruction) where:
            - segments: List of ('text', content) or ('code', content) tuples
            - interpret_instruction: Instruction from __INTERPRET__: marker, or empty string
        """
        import re

        # Extract __INTERPRET__: marker if present
        interpret_match = re.search(r'__INTERPRET__:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        interpret_instruction = interpret_match.group(1).strip() if interpret_match else ""

        # Remove __INTERPRET__: line from response
        if interpret_instruction:
            response = re.sub(r'__INTERPRET__:.+?(?:\n|$)', '', response, flags=re.IGNORECASE)

        # Split on code blocks
        segments = []
        pattern = r'```python\n(.*?)```'
        last_end = 0

        for match in re.finditer(pattern, response, re.DOTALL):
            # Add text before code block
            text_before = response[last_end:match.start()].strip()
            if text_before:
                segments.append(('text', text_before))

            # Add code block
            code = match.group(1).strip()
            if code:
                segments.append(('code', code))

            last_end = match.end()

        # Add remaining text after last code block
        text_after = response[last_end:].strip()
        if text_after:
            segments.append(('text', text_after))

        return segments, interpret_instruction

    async def _analyze_execution_results(
        self,
        user_query: str,
        segments: List[Tuple[str, str]],
        execution_results: List[ExecutionResult],
        interpret_instruction: str
    ) -> Dict[str, Any]:
        """
        Analyze execution results with full sequential context.

        Replaces old summarization system with unified analysis that provides:
        - Response summary (for conversation history compression)
        - Data changes (detailed description of what was added/modified)
        - Biological interpretation (if requested via __INTERPRET__)
        - Key findings (if interpretation requested)

        Args:
            user_query: Original user question
            segments: Sequential segments from LLM response [(type, content), ...]
            execution_results: Results from code execution
            interpret_instruction: What to focus on (from __INTERPRET__ marker, empty if not present)

        Returns:
            Dict with response_summary, data_changes, interpretation (optional), key_findings (optional)
        """
        # Build assembled view showing full sequential flow
        assembled_parts = []
        result_idx = 0

        for seg_type, content in segments:
            if seg_type == 'text':
                assembled_parts.append(f"**Your explanation:**\n{content}\n")
            elif seg_type == 'code':
                assembled_parts.append(f"**Code:**\n```python\n{content}\n```\n")
                # Add corresponding output
                if result_idx < len(execution_results):
                    result = execution_results[result_idx]
                    if result.stdout:
                        truncated = result.stdout[:3000]
                        if len(result.stdout) > 3000:
                            truncated += "\n... (output truncated)"
                        assembled_parts.append(f"**Output:**\n{truncated}\n")
                    if result.error:
                        assembled_parts.append(f"**Error:**\n{result.error}\n")
                    result_idx += 1

        assembled_context = "\n".join(assembled_parts)

        # Build session context (compact)
        session_info = ""
        if self.session and self.session.has_data:
            summary = self.session.get_summary()
            total_cells = sum(s['n_obs'] for s in summary['slices'])
            roi_names = list(self.session.roi_subsets.keys())
            session_info = f"Session: {summary['n_slices']} slice(s), {total_cells:,} cells"
            if roi_names:
                session_info += f", ROIs: {', '.join(roi_names)}"
            for s in summary['slices']:
                session_info += f"\nSlice {s['slice_id']} ({s['modality']}, {s['data_level']}): {s['n_vars']:,} genes"
                celltypes = s.get('celltypes', [])
                if celltypes:
                    session_info += f", celltypes = {', '.join(celltypes)}"

        # Build conversation history (compact)
        conversation_history = self.memory.get_history_string()

        # Build prompt
        has_interpret = bool(interpret_instruction)

        prompt = f"""Analyze spatial transcriptomics code execution results.

{f"**{session_info}**" + chr(10) if session_info else ""}{f"**Conversation History:**" + chr(10) + conversation_history + chr(10) + "**[End of Conversation History]**" + chr(10) if conversation_history else ""}
User asked: "{user_query}"

{assembled_context}

Provide JSON response:
{{
  "response_summary": "Brief summary of entire response in under 4 sentences (what was done and key results)",
  "data_changes": "Detailed description of what was added/changed in session data (columns, obsm keys, uns keys, ROIs). Be specific about names, types, and values. 2-3 sentences.",
  "interpretation": {"'" + interpret_instruction + "' - Provide biological interpretation in 2-4 sentences" if has_interpret else "null"},
  "key_findings": {["List of 1-3 specific biological findings with numbers"] if has_interpret else "[]"},
  "execution_issues": {{
    "has_issues": true/false,
    "issue_type": "error" | "validation_failed" | "partial_success" | "no_effect" | null,
    "explanation": "Clear explanation of what went wrong and why" (only if has_issues=true)
  }}
}}

Rules:
- response_summary: Always provide, covers entire response
- data_changes: Always provide, be detailed and specific
- interpretation: Only if requested above (otherwise null)
- key_findings: Only if interpretation requested (otherwise empty array)
- execution_issues: Detect if code ACTUALLY failed or did not achieve the user's goal
  - has_issues=true ONLY if: Python error occurred, validation failed, code had no effect, or the user's goal was NOT achieved
  - has_issues=false if: output was truncated but the analysis/data changes completed successfully. Truncated print output is NORMAL for large datasets and is NOT an issue.
  - issue_type: "error" (Python exception), "validation_failed" (data validation failed), "partial_success" (goal partially achieved — NOT for truncated output), "no_effect" (code ran but changed nothing)
  - explanation: User-friendly explanation of what went wrong and why

Output valid JSON only:"""

        try:
            response = await self.llm.run(prompt)

            # Log this LLM call
            self.prompt_logger.log_llm_call(
                call_type="analyze_execution_results",
                full_prompt=prompt,
                response=response,
                metadata={
                    'model': self.llm.config.model,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                    'output_tokens': getattr(self.llm, 'last_output_tokens', None),
                }
            )

            # Parse JSON response
            import json
            import re

            # Extract JSON from response (in case LLM adds markdown)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
            else:
                analysis = json.loads(response)

            return analysis
        except Exception as e:
            logger.warning(f"Execution analysis failed: {e}")
            # Fallback to basic structure
            return {
                "response_summary": "Executed code successfully",
                "data_changes": "Modified session data",
                "interpretation": None,
                "key_findings": [],
                "execution_issues": {
                    "has_issues": False,
                    "issue_type": None,
                    "explanation": None
                }
            }

    async def _select_skill_matches_llm(self, request: str, top_k: int = 2) -> List[str]:
        """Use LLM to select relevant skills based on the request.

        This is pure LLM reasoning - no algorithmic routing, embeddings, or pattern matching.
        The LLM reads skill descriptions and decides which skills match the user's intent.

        Parameters
        ----------
        request : str
            User's natural language request
        top_k : int
            Maximum number of skills to match (default: 2)

        Returns
        -------
        List[str]
            List of skill slugs matched by the LLM

        Examples
        --------
        >>> matched = await agent._select_skill_matches_llm("Perform niche detection")
        >>> print(matched)
        ['niche-detection']
        """
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return []

        # Format all available skills for LLM
        skills_list = []
        for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda s: s.name.lower()):
            skills_list.append(f"- **{skill.slug}**: {skill.description}")

        skills_catalog = "\n".join(skills_list)

        # Ask LLM to match skills
        matching_prompt = f"""You are a strict skill matching system. Match skills ONLY when the user's request SPECIFICALLY asks for what the skill provides.

User Request: "{request}"

Available Skills:
{skills_catalog}

MATCHING CRITERIA:
- ✅ Match: User's request DIRECTLY asks for the skill's specific task/output
- ❌ Don't match: Request is only loosely related or shares general themes
- ❌ Don't match: User can accomplish their goal WITHOUT this skill
- ✅ Be conservative: When in doubt, return empty array

EXAMPLES:
- "Perform niche detection" → ["niche-detection"] ✅ (directly asks for niche detection)
- "Find spatial niches in tumor" → ["niche-detection"] ✅ (specifically wants niches)
- "Is gene X correlated with cell type Y?" → [] ❌ (correlation analysis, not niche detection)
- "Show spatial distribution of ERBB2" → [] ❌ (visualization, not niche detection)
- "Compare malignant cells between ROIs" → [] ❌ (comparison, not niche detection)
- "Cluster cells by spatial patterns" → [] ❌ (general clustering, niche detection is specific hierarchical method)

Your task:
1. Identify the SPECIFIC task the user is asking for
2. Match ONLY if a skill provides EXACTLY that task
3. Return at most {top_k} skill slugs, or fewer if not specifically relevant
4. Respond with ONLY a JSON array: ["skill-slug"] or []

CRITICAL: Return [] (empty array) unless the skill is SPECIFICALLY needed for the user's request.

Response (JSON array only):"""

        try:
            call_start = time.time()
            response = await self.llm.run(matching_prompt)
            call_duration = time.time() - call_start

            # Log this LLM call
            self.prompt_logger.log_llm_call(
                call_type="skill_matching",
                full_prompt=matching_prompt,
                response=response,
                metadata={
                    'model': self.llm.config.model,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': call_duration,
                    'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                    'output_tokens': getattr(self.llm, 'last_output_tokens', None),
                }
            )

            logger.debug(f"Skill matching LLM response: {response}")

            # Extract JSON array from response
            import json
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                matched_slugs = json.loads(json_match.group(0))
                validated_slugs = [slug for slug in matched_slugs if slug in self.skill_registry.skill_metadata]

                if validated_slugs:
                    logger.info(f"Skill matching result: {validated_slugs} (matched for query: '{request[:50]}...')")
                else:
                    logger.info(f"Skill matching result: No skills matched (query: '{request[:50]}...')")

                return validated_slugs

            logger.info(f"Skill matching result: No JSON found in response (query: '{request[:50]}...')")
            return []

        except Exception as exc:
            logger.warning(f"LLM skill matching failed: {exc}")
            return []

    async def _handle_with_pipeline(
        self,
        user_message: str,
        execute_code: bool = True,
        clarification_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle user message with robust skill pipeline.

        Pipeline: QueryPlanner → SkillFilter → SemanticMatcher → SkillVerifier → Execute

        Parameters
        ----------
        user_message : str
            User's question or request
        execute_code : bool
            Whether to execute generated code
        clarification_response : Optional[str]
            User's response to previous clarification question

        Returns
        -------
        Dict[str, Any]
            Result with type:
            - {"type": "clarification_needed", "question": str, "context": str}
            - {"type": "skill_selection", "message": str, "options": List[str]}
            - {"type": "prerequisites_needed", "questions": List[str], "skill": str}
            - {"type": "advice", "message": str}
            - {"type": "response", "message": str, "plots": list}
        """
        logger.info("="*60)
        logger.info("PIPELINE START: Processing user query")
        logger.info(f"Query: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        logger.info("="*60)

        # Handle clarification response
        if clarification_response:
            # Check which type of clarification this is
            if self.clarification_context.get_pending_skill_selection():
                # This is a skill selection response
                pending_selection = self.clarification_context.get_pending_skill_selection()
                plan_step = pending_selection['plan_step']
                skill_options = pending_selection['skill_options']

                # Parse user's selection
                selected_skill_slug = None
                user_input = clarification_response.strip().lower()

                # Try to match by number (1, 2, 3, ...) or by skill slug
                if user_input.isdigit():
                    selection_idx = int(user_input) - 1
                    if 0 <= selection_idx < len(skill_options):
                        selected_skill_slug = skill_options[selection_idx]
                else:
                    # Try to match by slug (partial match)
                    for slug in skill_options:
                        if user_input in slug.lower():
                            selected_skill_slug = slug
                            break

                if not selected_skill_slug:
                    # Invalid selection, ask again
                    logger.warning(f"Invalid skill selection: {clarification_response}")
                    skill_options_with_names = []
                    for slug in skill_options:
                        skill = self.skill_registry.load_full_skill(slug)
                        skill_name = skill.name if skill else slug
                        skill_options_with_names.append({"slug": slug, "name": skill_name})
                    return {
                        "type": "skill_selection",
                        "message": "Invalid selection. Please choose one of:",
                        "options": skill_options_with_names
                    }

                logger.info(f"User selected skill: {selected_skill_slug}")
                selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)

                # Clear selection context
                self.clarification_context.clear_skill_selection()

                # Run verification for selected skill
                if selected_skill and self.skill_verifier:
                    logger.info(f"Running verification for selected skill: {selected_skill_slug}")
                    user_responses = dict(
                        self.clarification_context._verifier_clarifications.get(selected_skill_slug, [])
                    )

                    verification_result = await self.skill_verifier.verify(
                        plan_step=plan_step,
                        selected_skill=selected_skill,
                        session_summary=self.session.get_summary() if self.session else {},
                        user_responses=user_responses
                    )

                    if not verification_result.prerequisites_met:
                        if verification_result.can_obtain_by_chat:
                            # Need to ask user for prerequisites
                            logger.info(f"Prerequisites missing, asking user for info...")
                            self.clarification_context._pending_plan_step = plan_step
                            self.clarification_context._pending_skill = selected_skill_slug
                            self.clarification_context._last_verifier_questions = verification_result.clarification_questions

                            return {
                                "type": "prerequisites_needed",
                                "questions": verification_result.clarification_questions,
                                "skill": selected_skill_slug
                            }
                        else:
                            # Can't proceed - needs prior work
                            logger.info(f"Prerequisites missing, needs prior work")
                            self._clear_clarification_context()
                            return {
                                "type": "advice",
                                "message": verification_result.advice
                            }

                    # Prerequisites met - execute
                    final_query = verification_result.complete_query
                    logger.info(f"Prerequisites met, executing with complete query")

                    matched_skill_slugs = [selected_skill_slug]
                    response = await self._handle_with_llm(
                        final_query,
                        execute_code=execute_code,
                        allow_planning=False,
                        matched_skills=matched_skill_slugs
                    )

                    self._clear_clarification_context()
                    return {
                        "type": "response",
                        "message": response,
                        "plots": self._last_plots
                    }
                else:
                    # No verifier or no skill - just execute
                    final_query = plan_step.refined_query
                    response = await self._handle_with_llm(
                        final_query,
                        execute_code=execute_code,
                        allow_planning=False
                    )

                    self._clear_clarification_context()
                    return {
                        "type": "response",
                        "message": response,
                        "plots": self._last_plots
                    }

            elif self.clarification_context.get_pending_plan_step():
                # This is a verifier clarification response
                plan_step = self.clarification_context.get_pending_plan_step()
                skill_slug = self.clarification_context.get_pending_skill()

                # Get the last question that was asked
                last_verifier_questions = self.clarification_context.get_last_verifier_questions()
                if last_verifier_questions:
                    # Store the response
                    if skill_slug not in self.clarification_context._verifier_clarifications:
                        self.clarification_context._verifier_clarifications[skill_slug] = []

                    # Map the response to the question (assuming single Q&A for simplicity)
                    # In future: handle multiple questions
                    self.clarification_context._verifier_clarifications[skill_slug].append(
                        (last_verifier_questions[0], clarification_response)
                    )

                    logger.info(f"Received verifier clarification: {clarification_response[:50]}...")

                    # Continue with verification
                    skill = self.skill_registry.load_full_skill(skill_slug)
                    user_responses = dict(self.clarification_context._verifier_clarifications[skill_slug])

                    verification_result = await self.skill_verifier.verify(
                        plan_step=plan_step,
                        selected_skill=skill,
                        session_summary=self.session.get_summary() if self.session else {},
                        user_responses=user_responses
                    )

                    if not verification_result.prerequisites_met:
                        if verification_result.can_obtain_by_chat:
                            # Still need more info
                            self.clarification_context._last_verifier_questions = verification_result.clarification_questions
                            return {
                                "type": "prerequisites_needed",  # Match streaming version
                                "questions": verification_result.clarification_questions,
                                "skill": skill_slug
                            }
                        else:
                            # Can't proceed
                            self._clear_clarification_context()
                            return {
                                "type": "advice",
                                "message": verification_result.advice
                            }

                    # Prerequisites met! Execute
                    complete_query = verification_result.complete_query
                    response = await self._handle_with_llm(
                        complete_query,
                        execute_code=execute_code,
                        allow_planning=False
                    )

                    self._clear_clarification_context()
                    return {
                        "type": "response",
                        "message": response,
                        "plots": self._last_plots
                    }

            else:
                # This is a planner clarification response
                self.clarification_context._planner_clarifications.append(
                    (self.clarification_context._last_planner_question or '', clarification_response)
                )
                logger.info(f"Received planner clarification: {clarification_response[:50]}...")

                # CRITICAL FIX: Override user_message with original query
                # When user responds to planner clarification, we need to re-plan with
                # the ORIGINAL query (e.g., "Annotate celltype") not the clarification
                # response (e.g., "slice 0")
                if self.clarification_context._original_query:
                    user_message = self.clarification_context._original_query
                    logger.info(f"Using original query for re-planning: {user_message[:100]}...")

                # Re-plan with clarification
                # Fall through to normal planning below

        # Step 1: PLANNER - Determine target slices
        if not self.query_planner:
            logger.error("QueryPlanner not initialized but planning is enabled!")
            return {"type": "error", "message": "Planning not enabled"}

        logger.info("-" * 60)
        logger.info("STAGE 1: QUERY PLANNING")
        logger.info("-" * 60)

        plan_result = await self.query_planner.plan(
            user_query=user_message,
            session_summary=self.session.get_summary() if self.session else {},
            previous_clarifications=self.clarification_context.get_planner_history()
        )

        # Handle planner clarification
        if plan_result.needs_clarification:
            logger.info(f"❓ Planner requesting clarification")
            logger.info(f"  Question: {plan_result.clarification_question[:100]}...")

            # Store original query so we can use it when user responds
            if not self.clarification_context._original_query:
                self.clarification_context._original_query = user_message
                logger.debug(f"Stored original query for re-planning: {user_message[:100]}...")

            self.clarification_context._last_planner_question = plan_result.clarification_question
            return {
                "type": "clarification_needed",
                "question": plan_result.clarification_question,
                "context": "planner"
            }

        # Clear planner clarification context after successful planning
        self.clarification_context._planner_clarifications = []
        self.clarification_context._original_query = None

        logger.info(f"✓ Planner generated {len(plan_result.steps)} step(s)")
        for i, step in enumerate(plan_result.steps, 1):
            logger.debug(f"  Step {i}: {step.description} (slices: {step.target_slice_ids})")

        # Process each step
        all_responses = []
        no_skill_matched_steps = []  # Track steps with no matched skills
        for step in plan_result.steps:
            logger.info(f"Processing Step {step.step_number}: {step.description}")

            # Step 2: FILTER - Programmatic filtering
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | STAGE 2: FILTER")
            logger.info(f"Target slices: {step.target_slice_ids}")
            logger.info("-" * 60)

            if not self.skill_filter or not self.skill_registry:
                logger.warning("SkillFilter not initialized, skipping filtering")
                compatible_skills = list(self.skill_registry.skill_metadata.values()) if self.skill_registry else []
            else:
                compatible_skills = self.skill_filter.filter_skills(
                    target_slice_ids=step.target_slice_ids,
                    session=self.session,
                    all_skills=self.skill_registry.skill_metadata
                )
                logger.info(f"✓ Filter result: {len(compatible_skills)} compatible skills")
                if compatible_skills:
                    skill_names = [s.slug for s in compatible_skills[:3]]
                    logger.debug(f"  Compatible skills: {skill_names}{'...' if len(compatible_skills) > 3 else ''}")

            # Step 3: SEMANTIC MATCHING
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | STAGE 3: SEMANTIC MATCHING")
            logger.info(f"Query: {step.refined_query}")
            logger.info("-" * 60)
            compatible_skills_dict = {skill.slug: skill for skill in compatible_skills}

            # Pass filtered skills to semantic matcher
            matched_slugs = await self._select_skill_matches_llm_filtered(
                request=step.refined_query,
                available_skills=compatible_skills_dict,
                top_k=2
            )

            logger.info(f"✓ Semantic matcher found: {len(matched_slugs)} skill(s)")
            if matched_slugs:
                logger.info(f"  Matched skills: {matched_slugs}")

            # Handle matching results
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | MATCHING RESULT")
            logger.info("-" * 60)

            if len(matched_slugs) == 0:
                # No skill matched - proceed without skill guidance
                logger.warning(f"⚠️  NO SPECIALIZED SKILL MATCHED")
                logger.info(f"  Step: {step.description}")
                logger.info(f"  Will attempt with general LLM capabilities")
                selected_skill = None
                no_skill_matched_steps.append(step.step_number)

            elif len(matched_slugs) == 1:
                # Single skill matched - proceed to verification
                selected_skill_slug = matched_slugs[0]
                selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)
                logger.info(f"✓ Single skill matched: {selected_skill_slug}")
                logger.info(f"  Skill name: {selected_skill.name if selected_skill else 'N/A'}")

            else:
                # Multiple skills matched - need user selection
                logger.info(f"🎯 Multiple skills matched: {len(matched_slugs)}")
                logger.info(f"  Matched: {matched_slugs}")
                logger.info(f"  Requesting user selection...")

                # Load skill names for display
                skill_options_with_names = []
                for slug in matched_slugs:
                    skill = self.skill_registry.load_full_skill(slug)
                    skill_name = skill.name if skill else slug
                    skill_options_with_names.append({"slug": slug, "name": skill_name})

                # Store pending selection context
                self.clarification_context._pending_skill_selection = {
                    'plan_step': step,
                    'skill_options': matched_slugs
                }

                return {
                    "type": "skill_selection",
                    "message": "Multiple skills matched for this task. Please select one:",
                    "options": skill_options_with_names
                }

            # Step 4: VERIFY prerequisites
            if selected_skill and self.skill_verifier:
                logger.info("-" * 60)
                logger.info(f"STEP {step.step_number} | STAGE 4: VERIFICATION")
                logger.info(f"Skill: {selected_skill.slug}")
                logger.info("-" * 60)

                # Get any previous responses for this skill
                user_responses = dict(
                    self.clarification_context._verifier_clarifications.get(selected_skill.slug, [])
                )

                if user_responses:
                    logger.debug(f"  Using {len(user_responses)} previous user response(s)")

                verification_result = await self.skill_verifier.verify(
                    plan_step=step,
                    selected_skill=selected_skill,
                    session_summary=self.session.get_summary() if self.session else {},
                    user_responses=user_responses
                )

                if not verification_result.prerequisites_met:
                    if verification_result.can_obtain_by_chat:
                        # Need to ask user for info
                        logger.info(f"❓ Prerequisites missing - can obtain via chat")
                        logger.info(f"  Missing: {verification_result.missing_prerequisites}")
                        logger.info(f"  Asking {len(verification_result.clarification_questions)} question(s)")
                        # Store context for when user responds
                        self.clarification_context._pending_plan_step = step
                        self.clarification_context._pending_skill = selected_skill.slug
                        self.clarification_context._last_verifier_questions = verification_result.clarification_questions

                        return {
                            "type": "prerequisites_needed",
                            "questions": verification_result.clarification_questions,
                            "skill": selected_skill.slug
                        }
                    else:
                        # Can't proceed - needs prior work
                        logger.warning(f"⚠️  Prerequisites missing - needs prior work")
                        logger.info(f"  Missing: {verification_result.missing_prerequisites}")
                        logger.info(f"  Advice: {verification_result.advice[:100]}...")
                        self._clear_clarification_context()
                        return {
                            "type": "advice",
                            "message": verification_result.advice
                        }

                # Prerequisites met - use complete query
                final_query = verification_result.complete_query
                logger.info(f"✓ Prerequisites met")
                logger.info(f"  Complete query: {final_query[:80]}...")

            else:
                # No skill or no verifier - use original query
                final_query = step.refined_query
                if not selected_skill:
                    logger.info("  No skill selected - using original query")

            # Step 5: EXECUTE
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | STAGE 5: EXECUTE")
            logger.info(f"Query: {final_query[:100]}...")
            logger.info("-" * 60)
            matched_skill_slugs = [selected_skill.slug] if selected_skill else []

            response = await self._handle_with_llm(
                final_query,
                execute_code=execute_code,
                allow_planning=False
            )

            logger.info(f"✓ Step {step.step_number} execution completed")
            all_responses.append(response)

        # Combine responses if multiple steps
        final_response = "\n\n".join(all_responses) if len(all_responses) > 1 else all_responses[0]

        # Add warning if no specialized skills were matched
        if no_skill_matched_steps:
            if len(no_skill_matched_steps) == 1 and len(plan_result.steps) == 1:
                # Single step, no skill matched
                warning_prefix = "⚠️  Note: No specialized skill found for this task. I'll do my best to help with general capabilities.\n\n"
            else:
                # Multiple steps or partial matching
                steps_str = ", ".join(map(str, no_skill_matched_steps))
                warning_prefix = f"⚠️  Note: No specialized skills found for step(s) {steps_str}. Proceeding with general capabilities.\n\n"
            final_response = warning_prefix + final_response

        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total steps: {len(plan_result.steps)}")
        logger.info(f"Steps with no skill: {len(no_skill_matched_steps)}")
        logger.info(f"Response length: {len(final_response)} chars")
        logger.info("="*60)

        self._clear_clarification_context()

        return {
            "type": "response",
            "message": final_response,
            "plots": self._last_plots
        }

    async def _handle_with_pipeline_events(
        self,
        user_message: str,
        execute_code: bool = True,
        clarification_response: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle user message with robust skill pipeline - streaming version.

        This is the MAIN PIPELINE ORCHESTRATOR for the new architecture.

        Architecture:
        -------------
        1. Clarification Handling (if clarification_response provided):
           - Skill Selection: User chose from multiple matched skills
           - Verifier Prerequisites: User provided required information
           - Planner Clarification: User answered "which slice?" etc.

        2. Pipeline Stages (sequential execution):
           - QueryPlanner: Determines target slices, may ask clarifications
           - SkillFilter: Filters skills by data format/modality/level
           - SemanticMatcher: LLM-based skill matching, may return multiple
           - SkillVerifier: Checks prerequisites, may ask questions
           - Execute: Runs via _handle_with_llm_events (includes interpretation & memory)

        3. State Change Detection:
           - Detects celltype updates, ROI changes, etc.
           - Emits final_response event with state_changes for frontend auto-reload

        Pipeline: QueryPlanner → SkillFilter → SemanticMatcher → SkillVerifier → Execute

        Parameters
        ----------
        user_message : str
            User's question or request
        execute_code : bool
            Whether to execute generated code
        clarification_response : Optional[str]
            User's response to previous clarification question
        before_state : Optional[Dict[str, Any]]
            Session state snapshot before execution (for change detection)

        Yields
        ------
        Dict[str, Any]
            Events with type:
            - {'type': 'clarification_needed', 'question': str, 'context': str}
            - {'type': 'skill_selection', 'message': str, 'options': list}
            - {'type': 'prerequisites_needed', 'questions': list, 'skill': str}
            - {'type': 'advice', 'message': str}
            - {'type': 'execution_start', 'step': int}
            - {'type': 'execution_output', 'text': str}
            - {'type': 'execution_complete', 'response': str, 'plots': list}
            - {'type': 'final_response', 'message': str, 'plots': list, 'state_changes': dict}
        """
        logger.info("="*60)
        logger.info("PIPELINE START: Processing user query")
        logger.info(f"Query: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        logger.info("="*60)

        # Handle clarification response
        if clarification_response:
            # Check which type of clarification this is
            if self.clarification_context.get_pending_skill_selection():
                # This is a skill selection response
                pending_selection = self.clarification_context.get_pending_skill_selection()
                plan_step = pending_selection['plan_step']
                skill_options = pending_selection['skill_options']

                # Parse user's selection
                selected_skill_slug = None
                user_input = clarification_response.strip().lower()

                # Try to match by number (1, 2, 3, ...) or by skill slug
                if user_input.isdigit():
                    selection_idx = int(user_input) - 1
                    if 0 <= selection_idx < len(skill_options):
                        selected_skill_slug = skill_options[selection_idx]
                else:
                    # Try to match by slug (partial match)
                    for slug in skill_options:
                        if user_input in slug.lower():
                            selected_skill_slug = slug
                            break

                if not selected_skill_slug:
                    # Invalid selection, ask again
                    logger.warning(f"Invalid skill selection: {clarification_response}")
                    skill_options_with_names = []
                    for slug in skill_options:
                        skill = self.skill_registry.load_full_skill(slug)
                        skill_name = skill.name if skill else slug
                        skill_options_with_names.append({"slug": slug, "name": skill_name})
                    yield {
                        "type": "skill_selection",
                        "message": "Invalid selection. Please choose one of:",
                        "options": skill_options_with_names
                    }
                    return

                logger.info(f"User selected skill: {selected_skill_slug}")
                selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)

                # Clear selection context
                self.clarification_context.clear_skill_selection()

                # Run verification for selected skill
                if selected_skill and self.skill_verifier:
                    logger.info(f"Running verification for selected skill: {selected_skill_slug}")
                    yield {'type': 'verification_start', 'skill': selected_skill_slug}

                    user_responses = dict(
                        self.clarification_context._verifier_clarifications.get(selected_skill_slug, [])
                    )

                    verification_result = await self.skill_verifier.verify(
                        plan_step=plan_step,
                        selected_skill=selected_skill,
                        session_summary=self.session.get_summary() if self.session else {},
                        user_responses=user_responses
                    )

                    if not verification_result.prerequisites_met:
                        if verification_result.can_obtain_by_chat:
                            # Need to ask user for prerequisites
                            logger.info(f"Prerequisites missing, asking user for info...")
                            self.clarification_context._pending_plan_step = plan_step
                            self.clarification_context._pending_skill = selected_skill_slug
                            self.clarification_context._last_verifier_questions = verification_result.clarification_questions

                            yield {
                                "type": "prerequisites_needed",
                                "questions": verification_result.clarification_questions,
                                "skill": selected_skill_slug
                            }
                            return
                        else:
                            # Can't proceed - needs prior work
                            logger.info(f"Prerequisites missing, needs prior work")
                            self._clear_clarification_context()
                            yield {
                                "type": "advice",
                                "message": verification_result.advice
                            }
                            return

                    # Prerequisites met - execute
                    final_query = verification_result.complete_query
                    logger.info(f"Prerequisites met, executing with complete query")
                    yield {'type': 'verification_complete', 'status': 'met'}

                    matched_skill_slugs = [selected_skill_slug]
                    yield {'type': 'execution_start', 'query': final_query}

                    # Stream execution events
                    async for event in self._handle_with_llm_events(
                        final_query,
                        execute_code=execute_code,
                        allow_planning=False,
                        matched_skill_slugs=matched_skill_slugs
                    ):
                        yield event

                    self._clear_clarification_context()
                    return
                else:
                    # No verifier or no skill - just execute
                    final_query = plan_step.refined_query
                    yield {'type': 'execution_start', 'query': final_query}

                    async for event in self._handle_with_llm_events(
                        final_query,
                        execute_code=execute_code,
                        allow_planning=False,
                        matched_skill_slugs=[]
                    ):
                        yield event

                    self._clear_clarification_context()
                    return

            elif self.clarification_context.get_pending_plan_step():
                # This is a verifier clarification response
                plan_step = self.clarification_context.get_pending_plan_step()
                skill_slug = self.clarification_context.get_pending_skill()

                # Get the last question that was asked
                last_verifier_questions = self.clarification_context.get_last_verifier_questions()
                if last_verifier_questions:
                    # Store the response
                    if skill_slug not in self.clarification_context._verifier_clarifications:
                        self.clarification_context._verifier_clarifications[skill_slug] = []

                    # Map the response to the question
                    self.clarification_context._verifier_clarifications[skill_slug].append(
                        (last_verifier_questions[0], clarification_response)
                    )

                    logger.info(f"Received verifier clarification: {clarification_response[:50]}...")

                    # Continue with verification
                    skill = self.skill_registry.load_full_skill(skill_slug)
                    user_responses = dict(self.clarification_context._verifier_clarifications[skill_slug])

                    verification_result = await self.skill_verifier.verify(
                        plan_step=plan_step,
                        selected_skill=skill,
                        session_summary=self.session.get_summary() if self.session else {},
                        user_responses=user_responses
                    )

                    if not verification_result.prerequisites_met:
                        if verification_result.can_obtain_by_chat:
                            # Still need more info
                            self.clarification_context._last_verifier_questions = verification_result.clarification_questions
                            yield {
                                "type": "prerequisites_needed",
                                "questions": verification_result.clarification_questions,
                                "skill": skill_slug
                            }
                            return
                        else:
                            # Can't proceed
                            self._clear_clarification_context()
                            yield {
                                "type": "advice",
                                "message": verification_result.advice
                            }
                            return

                    # Prerequisites met! Execute
                    complete_query = verification_result.complete_query
                    yield {'type': 'verification_complete', 'status': 'met'}
                    yield {'type': 'execution_start', 'query': complete_query}

                    async for event in self._handle_with_llm_events(
                        complete_query,
                        execute_code=execute_code,
                        allow_planning=False,
                        matched_skill_slugs=[skill_slug]
                    ):
                        yield event

                    self._clear_clarification_context()
                    return

            else:
                # This is a planner clarification response
                self.clarification_context._planner_clarifications.append(
                    (self.clarification_context._last_planner_question or '', clarification_response)
                )
                logger.info(f"Received planner clarification: {clarification_response[:50]}...")

                # CRITICAL FIX: Override user_message with original query
                # When user responds to planner clarification, we need to re-plan with
                # the ORIGINAL query (e.g., "Annotate celltype") not the clarification
                # response (e.g., "slice 0")
                if self.clarification_context._original_query:
                    user_message = self.clarification_context._original_query
                    logger.info(f"Using original query for re-planning: {user_message[:100]}...")

                # Re-plan with clarification
                # Fall through to normal planning below

        # Step 1: PLANNER - Determine target slices
        if not self.query_planner:
            logger.error("QueryPlanner not initialized but planning is enabled!")
            yield {"type": "error", "message": "Planning not enabled"}
            return

        logger.info("-" * 60)
        logger.info("STAGE 1: QUERY PLANNING")
        logger.info("-" * 60)
        yield {'type': 'planning_start'}

        plan_result = await self.query_planner.plan(
            user_query=user_message,
            session_summary=self.session.get_summary() if self.session else {},
            previous_clarifications=self.clarification_context.get_planner_history()
        )

        # Handle planner clarification
        if plan_result.needs_clarification:
            logger.info(f"❓ Planner requesting clarification")
            logger.info(f"  Question: {plan_result.clarification_question[:100]}...")

            # Store original query so we can use it when user responds
            if not self.clarification_context._original_query:
                self.clarification_context._original_query = user_message
                logger.debug(f"Stored original query for re-planning: {user_message[:100]}...")

            self.clarification_context._last_planner_question = plan_result.clarification_question
            yield {
                "type": "clarification_needed",
                "question": plan_result.clarification_question,
                "context": "planner"
            }
            return

        # Clear planner clarification context after successful planning
        self.clarification_context._planner_clarifications = []
        self.clarification_context._original_query = None

        logger.info(f"✓ Planner generated {len(plan_result.steps)} step(s)")
        yield {
            'type': 'planning_complete',
            'steps': len(plan_result.steps),
            'plan': [
                {
                    'step_number': s.step_number,
                    'description': s.description,
                    'target_slice_ids': s.target_slice_ids,
                }
                for s in plan_result.steps
            ]
        }

        for i, step in enumerate(plan_result.steps, 1):
            logger.debug(f"  Step {i}: {step.description} (slices: {step.target_slice_ids})")

        # Process each step
        all_responses = []
        all_plots = []
        no_skill_matched_steps = []  # Track steps with no matched skills

        for step in plan_result.steps:
            logger.info(f"Processing Step {step.step_number}: {step.description}")
            yield {
                'type': 'step_start',
                'step_number': step.step_number,
                'step_description': step.description,
                'total_steps': len(plan_result.steps)
            }

            # Step 2: FILTER - Programmatic filtering
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | STAGE 2: FILTER")
            logger.info(f"Target slices: {step.target_slice_ids}")
            logger.info("-" * 60)

            if not self.skill_filter or not self.skill_registry:
                logger.warning("SkillFilter not initialized, skipping filtering")
                compatible_skills = list(self.skill_registry.skill_metadata.values()) if self.skill_registry else []
            else:
                compatible_skills = self.skill_filter.filter_skills(
                    target_slice_ids=step.target_slice_ids,
                    session=self.session,
                    all_skills=self.skill_registry.skill_metadata
                )
                logger.info(f"✓ Filter result: {len(compatible_skills)} compatible skills")

            # Step 3: SEMANTIC MATCHING
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | STAGE 3: SEMANTIC MATCHING")
            logger.info(f"Query: {step.refined_query}")
            logger.info("-" * 60)
            compatible_skills_dict = {skill.slug: skill for skill in compatible_skills}

            # Pass filtered skills to semantic matcher
            matched_slugs = await self._select_skill_matches_llm_filtered(
                request=step.refined_query,
                available_skills=compatible_skills_dict,
                top_k=2
            )

            logger.info(f"✓ Semantic matcher found: {len(matched_slugs)} skill(s)")

            # Handle matching results
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | MATCHING RESULT")
            logger.info("-" * 60)

            if len(matched_slugs) == 0:
                # No skill matched - proceed without skill guidance
                logger.warning(f"⚠️  NO SPECIALIZED SKILL MATCHED")
                logger.info(f"  Step: {step.description}")
                logger.info(f"  Will attempt with general LLM capabilities")
                selected_skill = None
                no_skill_matched_steps.append(step.step_number)
                yield {'type': 'no_skill_matched', 'step': step.step_number}

            elif len(matched_slugs) == 1:
                # Single skill matched - proceed to verification
                selected_skill_slug = matched_slugs[0]
                selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)
                logger.info(f"✓ Single skill matched: {selected_skill_slug}")
                yield {'type': 'skill_matched', 'skill': selected_skill_slug, 'step': step.step_number}

            else:
                # Multiple skills matched - need user selection
                logger.info(f"🎯 Multiple skills matched: {len(matched_slugs)}")
                logger.info(f"  Matched: {matched_slugs}")
                logger.info(f"  Requesting user selection...")

                # Load skill names for display
                skill_options_with_names = []
                for slug in matched_slugs:
                    skill = self.skill_registry.load_full_skill(slug)
                    skill_name = skill.name if skill else slug
                    skill_options_with_names.append({"slug": slug, "name": skill_name})

                # Store pending selection context
                self.clarification_context._pending_skill_selection = {
                    'plan_step': step,
                    'skill_options': matched_slugs
                }

                yield {
                    "type": "skill_selection",
                    "message": "Multiple skills matched for this task. Please select one:",
                    "options": skill_options_with_names
                }
                return

            # Step 4: VERIFY prerequisites
            if selected_skill and self.skill_verifier:
                logger.info("-" * 60)
                logger.info(f"STEP {step.step_number} | STAGE 4: VERIFICATION")
                logger.info(f"Skill: {selected_skill.slug}")
                logger.info("-" * 60)
                yield {'type': 'verification_start', 'skill': selected_skill.slug, 'step': step.step_number}

                # Get any previous responses for this skill
                user_responses = dict(
                    self.clarification_context._verifier_clarifications.get(selected_skill.slug, [])
                )

                verification_result = await self.skill_verifier.verify(
                    plan_step=step,
                    selected_skill=selected_skill,
                    session_summary=self.session.get_summary() if self.session else {},
                    user_responses=user_responses
                )

                if not verification_result.prerequisites_met:
                    if verification_result.can_obtain_by_chat:
                        # Need to ask user for info
                        logger.info(f"❓ Prerequisites missing - can obtain via chat")
                        logger.info(f"  Missing: {verification_result.missing_prerequisites}")
                        logger.info(f"  Asking {len(verification_result.clarification_questions)} question(s)")
                        # Store context for when user responds
                        self.clarification_context._pending_plan_step = step
                        self.clarification_context._pending_skill = selected_skill.slug
                        self.clarification_context._last_verifier_questions = verification_result.clarification_questions

                        yield {
                            "type": "prerequisites_needed",
                            "questions": verification_result.clarification_questions,
                            "skill": selected_skill.slug
                        }
                        return
                    else:
                        # Can't proceed - needs prior work
                        logger.warning(f"⚠️  Prerequisites missing - needs prior work")
                        logger.info(f"  Missing: {verification_result.missing_prerequisites}")
                        logger.info(f"  Advice: {verification_result.advice[:100]}...")
                        self._clear_clarification_context()
                        yield {
                            "type": "advice",
                            "message": verification_result.advice
                        }
                        return

                # Prerequisites met - use complete query
                final_query = verification_result.complete_query
                logger.info(f"✓ Prerequisites met")
                logger.info(f"  Complete query: {final_query[:80]}...")
                yield {'type': 'verification_complete', 'status': 'met', 'step': step.step_number}

            else:
                # No skill or no verifier - use original query
                final_query = step.refined_query
                if not selected_skill:
                    logger.info("  No skill selected - using original query")

            # Step 5: EXECUTE
            logger.info("-" * 60)
            logger.info(f"STEP {step.step_number} | STAGE 5: EXECUTE")
            logger.info(f"Query: {final_query[:100]}...")
            logger.info("-" * 60)
            matched_skill_slugs = [selected_skill.slug] if selected_skill else []

            yield {'type': 'execution_start', 'step': step.step_number, 'query': final_query}

            # Stream execution events from _handle_with_llm_events
            step_response = None
            step_plots = []
            async for event in self._handle_with_llm_events(
                final_query,
                execute_code=execute_code,
                allow_planning=False,
                matched_skill_slugs=matched_skill_slugs
            ):
                # Forward all events (execution_output, reflection_*, execution_complete)
                yield event

                # Capture final response and plots
                if event['type'] == 'execution_complete':
                    step_response = event['response']
                    step_plots = event.get('plots', [])

            logger.info(f"✓ Step {step.step_number} execution completed")
            if step_response:
                all_responses.append(step_response)
            if step_plots:
                all_plots.extend(step_plots)

        # Combine responses if multiple steps
        final_response = "\n\n".join(all_responses) if len(all_responses) > 1 else all_responses[0] if all_responses else ""

        # Emit warning if no specialized skills were matched (as separate event, not in response)
        if no_skill_matched_steps:
            if len(no_skill_matched_steps) == 1 and len(plan_result.steps) == 1:
                # Single step, no skill matched
                warning_message = "⚠️  Note: No specialized skill found for this task. I'll do my best to help with general capabilities."
            else:
                # Multiple steps or partial matching
                steps_str = ", ".join(map(str, no_skill_matched_steps))
                warning_message = f"⚠️  Note: No specialized skills found for step(s) {steps_str}. Proceeding with general capabilities."
            # Emit warning as separate event (will appear after execution_complete in UI)
            yield {'type': 'warning', 'message': warning_message}

        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total steps: {len(plan_result.steps)}")
        logger.info(f"Steps with no skill: {len(no_skill_matched_steps)}")
        logger.info(f"Response length: {len(final_response)} chars")
        logger.info("="*60)

        self._clear_clarification_context()

        # NOTE: State change detection moved to chat_with_events() to ensure
        # it happens for ALL execution paths (including early returns from clarifications)
        # This event is primarily for multi-step tracking
        yield {
            "type": "pipeline_complete",
            "total_steps": len(plan_result.steps),
            "final_response": final_response,
            "plots": all_plots
        }

    async def _select_skill_matches_llm_filtered(
        self,
        request: str,
        available_skills: Dict[str, Any],
        top_k: int = 2
    ) -> List[str]:
        """Semantic skill matching on pre-filtered skills.

        This is a variant of _select_skill_matches_llm that takes already-filtered skills.

        Parameters
        ----------
        request : str
            User's request
        available_skills : Dict[str, SkillMetadata]
            Pre-filtered compatible skills
        top_k : int
            Maximum number of skills to return

        Returns
        -------
        List[str]
            List of matched skill slugs
        """
        if not available_skills:
            return []

        # Format available skills for LLM
        skills_list = []
        for slug, skill in sorted(available_skills.items(), key=lambda x: x[1].name.lower()):
            skills_list.append(f"- **{slug}**: {skill.description}")

        skills_catalog = "\n".join(skills_list)

        # Use same matching prompt as original
        matching_prompt = f"""You are a strict skill matching system. Match skills ONLY when the user's request SPECIFICALLY asks for what the skill provides.

User Request: "{request}"

Available Skills:
{skills_catalog}

MATCHING CRITERIA:
- ✅ Match: User's request DIRECTLY asks for the skill's specific task/output
- ❌ Don't match: Request is only loosely related or shares general themes
- ❌ Don't match: User can accomplish their goal WITHOUT this skill
- ✅ Be conservative: When in doubt, return empty array

Your task:
1. Identify the SPECIFIC task the user is asking for
2. Match ONLY if a skill provides EXACTLY that task
3. Return at most {top_k} skill slugs, or fewer if not specifically relevant
4. Respond with ONLY a JSON array: ["skill-slug"] or []

CRITICAL: Return [] (empty array) unless the skill is SPECIFICALLY needed for the user's request.

Response (JSON array only):"""

        try:
            import time
            from datetime import datetime

            call_start = time.time()
            response = await self.llm.run(matching_prompt)
            call_duration = time.time() - call_start

            logger.debug(f"Skill matching LLM response: {response}")

            # Log to prompt_logger if available
            if self.prompt_logger:
                self.prompt_logger.log_llm_call(
                    call_type="semantic_matcher",
                    full_prompt=matching_prompt,
                    response=response,
                    metadata={
                        'model': getattr(self.llm.config, 'model', 'unknown'),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': call_duration,
                        'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                        'output_tokens': getattr(self.llm, 'last_output_tokens', None),
                        'available_skills': list(available_skills.keys()),
                        'top_k': top_k
                    }
                )

            # Extract JSON array
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                matched_slugs = json.loads(json_match.group(0))
                validated_slugs = [slug for slug in matched_slugs if slug in available_skills]

                if validated_slugs:
                    logger.info(f"Skill matching result: {validated_slugs}")
                else:
                    logger.info(f"No skills matched")

                return validated_slugs

            logger.info(f"No JSON found in response")
            return []

        except Exception as exc:
            logger.warning(f"LLM skill matching failed: {exc}")
            return []

    def _clear_clarification_context(self):
        """Clear clarification context after successful execution."""
        self.clarification_context.clear_all()

    def _format_skill_guidance(self, skill_definitions: List) -> str:
        """Format skill instructions for prompt injection.

        Parameters
        ----------
        skill_definitions : List[SkillDefinition]
            List of loaded skill definitions with full content

        Returns
        -------
        str
            Formatted skill guidance text for injection into LLM prompt
        """
        if not skill_definitions:
            return ""

        blocks = []
        for skill in skill_definitions:
            # CRITICAL: Skills can be 20K+ chars - use 25K to ensure no truncation
            # Truncation causes LLM to miss workflow steps and generate broken code
            instructions = skill.prompt_instructions(max_chars=25000, provider=self.llm.config.provider)
            blocks.append(f"**Skill: {skill.name}**\n{instructions}")

        return "\n\n".join(blocks)

    async def _validate_code(self, code: str, user_query: str) -> Dict[str, Any]:
        """
        Review generated code for correctness before execution.

        Catches common errors: wrong ROI names, missing columns, etc.

        Args:
            code: Generated Python code
            user_query: Original user query

        Returns:
            Dictionary with:
                - is_valid: bool (True if code looks good)
                - issues: List[str] (list of identified issues)
                - corrected_code: Optional[str] (fixed code if possible)
        """
        # Get current session state for validation
        roi_names = list(self.session.roi_subsets.keys()) if self.session else []
        slice_0 = self.session.get_slice(0) if self.session else None
        has_celltypes = (slice_0 and 'celltype' in slice_0.adata.obs.columns)

        validation_prompt = f"""Review this code for correctness before execution.

User Query: "{user_query}"

Generated Code:
```python
{code}
```

Available Context:
- ROIs defined: {roi_names if roi_names else 'None'}
- Has celltypes: {has_celltypes}

Check for:
1. ROI names exist (use actual names: {roi_names})
2. Column names correct ('celltype' not 'cell_type')
3. Logic matches user intent
4. No obvious syntax errors

Return JSON:
{{
  "is_valid": true/false,
  "issues": ["list of issues"],
  "corrected_code": "fixed code if needed (or null)"
}}
"""

        try:
            call_start = time.time()
            response = await self.llm.run(validation_prompt)
            call_duration = time.time() - call_start

            # Log this LLM call
            self.prompt_logger.log_llm_call(
                call_type="code_validation",
                full_prompt=validation_prompt,
                response=response,
                metadata={
                    'model': self.llm.config.model,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': call_duration,
                    'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                    'output_tokens': getattr(self.llm, 'last_output_tokens', None),
                }
            )

            # Extract JSON from response (handle if LLM returns wrapped response)
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                validation = json.loads(json_match.group())
                return {
                    "is_valid": validation.get("is_valid", True),
                    "issues": validation.get("issues", []),
                    "corrected_code": validation.get("corrected_code")
                }

            # If no JSON found, assume valid
            return {"is_valid": True, "issues": [], "corrected_code": None}

        except Exception as e:
            logger.warning(f"Code validation failed: {e}")
            # On validation failure, assume code is valid (fail open, not closed)
            return {"is_valid": True, "issues": [], "corrected_code": None}

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from markdown."""
        # Match ```python or ``` code blocks
        pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _extract_meaningful_summary(self, text: str, max_chars: int = 200) -> str:
        """
        Extract meaningful content from text, skipping decoration banners.

        Args:
            text: Full text to extract from
            max_chars: Maximum characters to return

        Returns:
            Extracted meaningful text without decoration lines
        """
        lines = text.split('\n')
        meaningful_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue

            # Skip pure decoration lines (===, ---, ***, ###)
            # Remove spaces and check if only decoration chars remain
            cleaned = stripped.replace(' ', '')
            if cleaned and all(c in '=-*#' for c in cleaned):
                continue

            meaningful_lines.append(line)

            # Stop after collecting enough
            if len('\n'.join(meaningful_lines)) >= max_chars:
                break

        result = '\n'.join(meaningful_lines)[:max_chars]
        if len(result) < len(text):
            result += "..."
        return result

    def _get_session_context(self) -> Dict[str, Any]:
        """
        Get enhanced session context for planning.

        Now includes data modifications so planner knows what columns/data already exist.
        This prevents redundant planning (e.g., creating niche_label when it already exists).
        """
        if not self.session or not self.session.has_data:
            return {"data_loaded": False}

        summary = self.session.get_summary()
        context = {
            "data_loaded": True,
            **summary
        }

        # NEW: Add data modifications so planner knows what exists
        recent_mods = self.memory.data_modifications.get_recent_modifications(n=10)
        if recent_mods:
            context["data_modifications"] = [
                {
                    "type": mod.modification_type,
                    "target": mod.target,
                    "details": mod.details,
                    "description": mod.get_description()
                }
                for mod in recent_mods
            ]

            # Explicitly list added columns/data (CRITICAL for planner!)
            # Use set to deduplicate (column might be both "added" and "updated")
            added_columns_set = set()
            added_obsm_set = set()
            added_uns_set = set()

            for mod in recent_mods:
                # .obs columns (celltype, etc.)
                if mod.modification_type in ["column_added", "column_updated"]:
                    col_name = mod.details.get("column_name")
                    target = mod.target
                    if col_name:
                        added_columns_set.add(f"{col_name} (in {target})")

                # .obsm matrices (deconv_weights, embeddings, etc.)
                elif mod.modification_type in ["obsm_added", "obsm_updated"]:
                    key = mod.details.get("key")
                    target = mod.target
                    if key:
                        # Include shape and description for context
                        shape = mod.details.get("shape", "")
                        desc = mod.details.get("description", "")
                        added_obsm_set.add(f"{key} (in {target}, {shape})")

                # .uns metadata (celltype_colors, etc.)
                elif mod.modification_type == "uns_updated":
                    key = mod.details.get("key")
                    target = mod.target
                    if key:
                        desc = mod.details.get("description", "")
                        added_uns_set.add(f"{key} (in {target})")

            # Build available data summary
            if added_columns_set:
                context["available_columns"] = sorted(list(added_columns_set))
            if added_obsm_set:
                context["available_obsm"] = sorted(list(added_obsm_set))
            if added_uns_set:
                context["available_uns"] = sorted(list(added_uns_set))

        # NEW: Add recent actions so planner knows what was done
        recent_turns = self.memory.turns[-3:] if len(self.memory.turns) >= 3 else self.memory.turns
        if recent_turns:
            context["recent_actions"] = []
            for turn in recent_turns:
                action_summary = {
                    "user_query": turn.user_message[:100],
                }
                # Include execution result summary if available
                # Priority 1: Use execution_summary (more concise)
                if hasattr(turn, 'execution_summary') and turn.execution_summary:
                    action_summary["result"] = turn.execution_summary[:200]
                # Priority 2: Smart truncation (skip decoration banners)
                elif turn.execution_result:
                    action_summary["result"] = self._extract_meaningful_summary(
                        turn.execution_result, max_chars=200
                    )
                context["recent_actions"].append(action_summary)

        return context

    def get_status(self) -> Dict[str, Any]:
        """Get agent status summary."""
        return {
            "model": self.llm.config.model,
            "session_active": self.session is not None,
            "data_loaded": self.session and self.session.has_data,
            "conversation_length": len(self.memory),
            "executor_variables": len(self.executor.namespace),
            "skills_loaded": len(self.skill_registry.skills) if self.skill_registry else 0
        }

    def get_last_state_changes(self) -> Dict[str, Any]:
        """Get state changes from last chat execution for frontend synchronization.

        Returns:
            Dictionary with state changes:
            - rois_added: List of ROI names added
            - rois_deleted: List of ROI names deleted
            - celltypes_updated: List of slice_ids with celltype updates
            - celltype_colors_updated: List of slice_ids with color updates
            - deconv_weights_updated: List of slice_ids with deconv_weights updates
        """
        return self._state_changes.copy()

    def reset(self) -> None:
        """Reset agent state (clear memory, executor, and skill imports)."""
        import sys

        self.memory.clear()
        self.executor.reset_namespace()

        # NEW: Clean up loaded skill paths
        for skill_path in self._loaded_skill_paths:
            if skill_path in sys.path:
                sys.path.remove(skill_path)
        self._loaded_skill_paths.clear()

        # NEW: Clean up skill modules from sys.modules to fully reset
        modules_to_remove = [
            name for name in sys.modules
            if name.startswith('celltype_fast') or
               name.startswith('celltype_scvi') or
               name.startswith('niche_analysis_lib')
        ]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

        logger.info("Reset agent state and cleaned up skill imports")

    def __repr__(self) -> str:
        return (
            f"SpatialAgent(model={self.llm.config.model}, "
            f"messages={len(self.memory)}, "
            f"session={'active' if self.session else 'none'})"
        )


__all__ = ["SpatialAgent", "SYSTEM_PROMPT"]

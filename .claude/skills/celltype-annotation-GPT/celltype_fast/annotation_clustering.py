"""
Fast cell type annotation using clustering + marker genes + LLM.

This module provides functionality to annotate spatial transcriptomics data
using an unsupervised clustering approach combined with LLM-based annotation.

Method:
1. Leiden clustering to identify cell populations
2. Differential expression to find marker genes per cluster
3. LLM annotation based on markers and tissue context

Advantages:
- No reference dataset required
- Fast (minutes vs hours for scANVI)
- Leverages LLM knowledge of tissue biology
- Good for exploratory analysis

Based on:
- Traag et al., Sci Rep 2019 (Leiden algorithm)
- Standard scanpy workflow
"""

from typing import Optional, List, Dict, Any
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def annotate_celltype_clustering(
    adata_spatial: ad.AnnData,
    tissue_type: str,
    llm_function: callable,
    resolution: float = 0.5,
    n_top_genes: int = 10,
    min_cluster_size: int = 10,
    preprocess: bool = True
) -> pd.Series:
    """
    Annotate cell types using clustering + marker genes + LLM.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial transcriptomics data to annotate.
        Must have gene expression in .X
    tissue_type : str
        Type of tissue (e.g., 'breast cancer', 'brain', 'liver')
        Used to provide context to LLM for annotation
    llm_function : callable
        Function to call LLM for annotation.
        Should accept (markers_dict, tissue_type) and return Dict[cluster, celltype]
    resolution : float, default=0.5
        Resolution parameter for Leiden clustering.
        Higher values = more clusters
    n_top_genes : int, default=10
        Number of top marker genes per cluster to use for annotation
    min_cluster_size : int, default=10
        Minimum cells per cluster. Smaller clusters marked as 'Unknown'
    preprocess : bool, default=True
        Whether to run preprocessing (normalize, log, HVG, PCA, neighbors)
        Set to False if data is already preprocessed

    Returns
    -------
    pd.Series
        Predicted cell type labels for spatial data.
        Index matches adata_spatial.obs_names

    Examples
    --------
    >>> def my_llm_annotator(markers_dict, tissue_type):
    ...     # Call your LLM here
    ...     return {'0': 'T cells', '1': 'Tumor cells', ...}
    >>>
    >>> st_data = sc.read_h5ad('spatial.h5ad')
    >>> celltype_labels = annotate_celltype_clustering(
    ...     st_data,
    ...     tissue_type='breast cancer',
    ...     llm_function=my_llm_annotator
    ... )
    >>> st_data.obs['celltype'] = celltype_labels
    """

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Copy data to avoid modifying original
    adata = adata_spatial.copy()

    print(f"\nInput data:")
    print(f"  Spatial: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  Tissue type: {tissue_type}")

    # Preprocessing
    if preprocess:
        print("\nPreprocessing...")

        # Normalize and log-transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Find highly variable genes
        print("  Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')

        # PCA
        print("  Computing PCA...")
        sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

        # Neighbors graph
        print("  Computing neighbor graph...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

        print("  ✓ Preprocessing complete")
    else:
        print("\nSkipping preprocessing (assuming data is preprocessed)")

    # Clustering
    print(f"\nClustering (resolution={resolution})...")
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')

    n_clusters = adata.obs['leiden'].nunique()
    print(f"  Found {n_clusters} clusters")

    # Check cluster sizes
    cluster_sizes = adata.obs['leiden'].value_counts()
    print(f"  Cluster sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"    Cluster {cluster}: {size:,} cells")

    # Find marker genes
    print(f"\nFinding marker genes (top {n_top_genes} per cluster)...")
    sc.tl.rank_genes_groups(
        adata,
        groupby='leiden',
        method='wilcoxon',
        n_genes=n_top_genes
    )

    # Extract marker genes per cluster
    markers_dict = {}
    for cluster in adata.obs['leiden'].cat.categories:
        # Get top genes for this cluster
        genes = sc.get.rank_genes_groups_df(adata, group=cluster, key='rank_genes_groups')
        top_genes = genes.head(n_top_genes)

        markers_dict[str(cluster)] = {
            'genes': top_genes['names'].tolist(),
            'scores': top_genes['scores'].tolist(),
            'pvals': top_genes['pvals_adj'].tolist(),
            'logfoldchanges': top_genes['logfoldchanges'].tolist(),
            'n_cells': int(cluster_sizes[cluster])
        }

        print(f"  Cluster {cluster}: {', '.join(top_genes['names'].head(5).tolist())}...")

    # Call LLM for annotation
    print(f"\nAnnotating clusters using LLM...")
    print(f"  Tissue context: {tissue_type}")

    try:
        cluster_annotations = llm_function(markers_dict, tissue_type)
        print(f"  ✓ LLM annotation complete")

        # Validate annotations
        if not isinstance(cluster_annotations, dict):
            raise ValueError(f"LLM function must return dict, got {type(cluster_annotations)}")

        # Check all clusters are annotated
        for cluster in adata.obs['leiden'].cat.categories:
            if str(cluster) not in cluster_annotations:
                print(f"  ⚠️  Warning: Cluster {cluster} not annotated, marking as 'Unknown'")
                cluster_annotations[str(cluster)] = 'Unknown'

    except Exception as e:
        print(f"  ❌ LLM annotation failed: {e}")
        print(f"  Falling back to generic cluster labels")
        cluster_annotations = {str(c): f'Cluster_{c}' for c in adata.obs['leiden'].cat.categories}

    # Handle small clusters
    for cluster, size in cluster_sizes.items():
        if size < min_cluster_size:
            print(f"  ⚠️  Cluster {cluster} has only {size} cells (< {min_cluster_size}), marking as 'Unknown'")
            cluster_annotations[str(cluster)] = 'Unknown'

    # Map cluster IDs to cell type labels
    print("\nMapping clusters to cell types...")
    celltype_predictions = adata.obs['leiden'].map(cluster_annotations)

    # Restore original index (preserve alignment with input)
    celltype_predictions.index = adata_spatial.obs_names

    print(f"\n✓ Annotation complete!")
    print(f"  Predicted {celltype_predictions.nunique()} cell types")
    print(f"  Distribution:")
    for celltype, count in celltype_predictions.value_counts().items():
        print(f"    {celltype}: {count:,} cells ({count/len(celltype_predictions):.1%})")

    return celltype_predictions


def create_llm_annotation_prompt(markers_dict: Dict[str, Dict], tissue_type: str) -> str:
    """
    Create a prompt for LLM to annotate clusters based on marker genes.

    Parameters
    ----------
    markers_dict : dict
        Dictionary with cluster IDs as keys, marker info as values
    tissue_type : str
        Type of tissue for context

    Returns
    -------
    str
        Formatted prompt for LLM
    """

    prompt = f"""You are a cell biology expert specializing in {tissue_type} tissue.

I have performed Leiden clustering on spatial transcriptomics data from {tissue_type} tissue and found {len(markers_dict)} clusters. For each cluster, I've identified the top marker genes using differential expression analysis.

Please annotate each cluster with the most likely cell type based on the marker genes. Provide specific cell type names (e.g., "CD8+ T cells", "Tumor cells", "Endothelial cells") rather than generic terms.

Cluster marker genes:

"""

    # Add marker genes for each cluster
    for cluster_id, info in markers_dict.items():
        genes = info['genes'][:10]  # Top 10 genes
        n_cells = info['n_cells']

        prompt += f"\nCluster {cluster_id} ({n_cells} cells):\n"
        prompt += f"  Top marker genes: {', '.join(genes)}\n"

    prompt += """\n
Please respond in the following JSON format:
{
    "0": "Cell type name for cluster 0",
    "1": "Cell type name for cluster 1",
    ...
}

Only include the JSON object in your response, no additional text.
"""

    return prompt


def annotate_with_openai(
    markers_dict: Dict[str, Dict],
    tissue_type: str,
    model: str = "gpt-4",
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Annotate clusters using OpenAI API.

    Parameters
    ----------
    markers_dict : dict
        Cluster marker information
    tissue_type : str
        Tissue type context
    model : str, default="gpt-4"
        OpenAI model to use
    api_key : str, optional
        OpenAI API key. If None, reads from environment

    Returns
    -------
    dict
        Mapping of cluster IDs to cell type names
    """

    try:
        import openai
        import json
        import os

        # Set API key
        if api_key:
            openai.api_key = api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not in environment")

        # Create prompt
        prompt = create_llm_annotation_prompt(markers_dict, tissue_type)

        print(f"  Calling OpenAI {model}...")

        # Call API
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a cell biology expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Parse response
        result = response.choices[0].message.content.strip()

        # Extract JSON (handle code blocks)
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()

        annotations = json.loads(result)

        return annotations

    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    except Exception as e:
        raise RuntimeError(f"OpenAI annotation failed: {e}")


def annotate_with_session_llm(
    markers_dict: Dict[str, Dict],
    tissue_type: str,
    session
) -> Dict[str, str]:
    """
    Annotate clusters using the LLM configured during session initialization.

    This function reuses the API key and model that the user provided when
    initializing the session, so it works with any LLM provider (OpenAI,
    Anthropic, Google, POE, etc.).

    Parameters
    ----------
    markers_dict : dict
        Cluster marker information
    tissue_type : str
        Tissue type context
    session : SimpleSession
        Session object with llm_config attribute

    Returns
    -------
    dict
        Mapping of cluster IDs to cell type names

    Raises
    ------
    ValueError
        If no LLM config is available in session
    """
    import json

    # Check if session has LLM config
    if not hasattr(session, 'llm_config') or session.llm_config is None:
        raise ValueError(
            "No LLM configuration found in session. "
            "Please provide API credentials when initializing the session."
        )

    try:
        from stat_agent.agent.llm_backend import LLMBackend
        import asyncio

        llm_config = session.llm_config
        api_key = llm_config.get('api_key')
        model = llm_config.get('model', 'gpt-4o')
        base_url = llm_config.get('base_url')

        print(f"  Using session LLM: {model}")

        # Create prompt
        prompt = create_llm_annotation_prompt(markers_dict, tissue_type)

        # Initialize LLM backend with session config
        llm_kwargs = {
            'system_prompt': 'You are a cell biology expert. Respond only with valid JSON.',
            'model': model,
            'api_key': api_key
        }
        if base_url:
            llm_kwargs['endpoint'] = base_url

        llm = LLMBackend(**llm_kwargs)

        # Call LLM (handle async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(llm.run(prompt))
        loop.close()

        # Extract JSON (handle code blocks)
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()

        annotations = json.loads(result)

        return annotations

    except ImportError as e:
        raise ImportError(f"Required module not available: {e}")
    except Exception as e:
        raise RuntimeError(f"LLM annotation failed: {e}")


def validate_annotation_inputs(
    adata_spatial: ad.AnnData,
    tissue_type: str
) -> dict:
    """
    Validate inputs for clustering-based annotation.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial data
    tissue_type : str
        Tissue type

    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool
        - 'errors': list of error messages
        - 'warnings': list of warning messages
        - 'info': dict with dataset statistics
    """
    errors = []
    warnings_list = []
    info = {}

    # Check spatial data
    if adata_spatial.n_obs == 0:
        errors.append("Spatial data has 0 cells")
    if adata_spatial.n_vars == 0:
        errors.append("Spatial data has 0 genes")

    # Check tissue type
    if not tissue_type or not isinstance(tissue_type, str):
        errors.append("Tissue type must be a non-empty string")

    # Check data size
    if adata_spatial.n_obs < 50:
        warnings_list.append(f"Only {adata_spatial.n_obs} cells - may produce unreliable clusters")

    if adata_spatial.n_vars < 100:
        warnings_list.append(f"Only {adata_spatial.n_vars} genes - limited marker gene resolution")

    # Info
    info['spatial_cells'] = adata_spatial.n_obs
    info['spatial_genes'] = adata_spatial.n_vars
    info['tissue_type'] = tissue_type

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings_list,
        'info': info
    }

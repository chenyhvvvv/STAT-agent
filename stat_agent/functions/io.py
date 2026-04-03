"""
Unified I/O functions for spatial transcriptomics data.

Two-mode loading (auto-detected):
1. Strict mode: Parses slice IDs and modalities from filenames using
   naming conventions (e.g., tissue_slice_0.h5ad, tissue_protein.h5ad).
   Images matched by rigid patterns (he_slice_0.tif, dapi.tif, etc.).
2. Flexible fallback: For images not matched by strict rules, uses
   filename similarity (SequenceMatcher) to pair images with h5ad files.
   Works with any directory layout and arbitrary filenames.

The loader always tries strict matching first, then runs flexible
matching as a fallback for any slices still without images.

Strict filename patterns:
- tissue.h5ad → slice_id=0, modality='gene'
- tissue_protein.h5ad → slice_id=1, modality='protein'
- tissue_slice_0.h5ad → slice_id=0, modality='gene'
- tissue_slice_1.h5ad → slice_id=1, modality='gene'
- tissue_slice_2_protein.h5ad → slice_id=2, modality='protein'
- spot_visium.h5ad → slice_id=0, modality='gene', data_level='spot'
"""

import re
import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from PIL import Image
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, List
import logging

from stat_agent.core.data_slice import DataSlice

logger = logging.getLogger(__name__)


# ========================================
# Core Loading Functions
# ========================================

def _preprocess_anndata(adata: ad.AnnData, filename: str) -> ad.AnnData:
    """
    Preprocess AnnData at initialization time.

    Steps:
    1. Make var and obs names unique
    2. Check if adata.X contains raw counts (integers);
       if not, try adata.raw.X or adata.layers['counts']
    3. Filter genes with min_cells=5
    4. Filter cells/spots with min_counts=20
    5. Validate necessary components (x, y)

    Parameters
    ----------
    adata : AnnData
        Loaded AnnData object
    filename : str
        Filename for logging context

    Returns
    -------
    adata : AnnData
        Preprocessed AnnData object

    Raises
    ------
    ValueError
        If raw counts cannot be found or required columns are missing
    """
    logger.info(f"Preprocessing {filename}...")

    # --- Step 1: Make names unique ---
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    logger.info(f"  Made var/obs names unique")

    # --- Step 2: Check for raw counts (integers) ---
    X = adata.X
    if sp.issparse(X):
        sample = X.data[:min(1000, len(X.data))] if len(X.data) > 0 else np.array([0])
    else:
        flat = X.ravel()
        sample = flat[:min(1000, len(flat))]

    is_integer = np.allclose(sample, np.round(sample))

    if is_integer:
        logger.info(f"  Raw counts verified (integer values in adata.X)")
    else:
        logger.warning(f"  adata.X does not contain raw counts, searching alternatives...")
        recovered = False

        # Try adata.raw
        if adata.raw is not None:
            raw_X = adata.raw.X
            if sp.issparse(raw_X):
                raw_sample = raw_X.data[:min(1000, len(raw_X.data))] if len(raw_X.data) > 0 else np.array([0])
            else:
                raw_flat = raw_X.ravel()
                raw_sample = raw_flat[:min(1000, len(raw_flat))]

            if np.allclose(raw_sample, np.round(raw_sample)):
                logger.info(f"  Found raw counts in adata.raw.X, using it as adata.X")
                adata = ad.AnnData(
                    X=adata.raw.X,
                    obs=adata.obs,
                    var=adata.raw.var,
                    obsm=adata.obsm,
                    obsp=adata.obsp if hasattr(adata, 'obsp') else None,
                    uns=adata.uns,
                )
                recovered = True

        # Try adata.layers['counts']
        if not recovered and 'counts' in getattr(adata, 'layers', {}):
            counts_X = adata.layers['counts']
            if sp.issparse(counts_X):
                counts_sample = counts_X.data[:min(1000, len(counts_X.data))] if len(counts_X.data) > 0 else np.array([0])
            else:
                counts_flat = counts_X.ravel()
                counts_sample = counts_flat[:min(1000, len(counts_flat))]

            if np.allclose(counts_sample, np.round(counts_sample)):
                logger.info(f"  Found raw counts in adata.layers['counts'], using it as adata.X")
                adata.X = counts_X
                recovered = True

        if not recovered:
            raise ValueError(
                f"Cannot find raw counts for {filename}. "
                f"adata.X contains non-integer values and no raw counts found in "
                f"adata.raw or adata.layers['counts']. "
                f"Please provide data with raw UMI counts."
            )

    # --- Step 3: Filter genes (min_cells=5) and cells/spots (min_counts=20) ---
    # Save user-defined .uns colors before scanpy filtering (scanpy mangles {key}_colors entries)
    saved_uns_colors = {}
    for key in list(adata.uns.keys()):
        if key.endswith('_colors'):
            saved_uns_colors[key] = adata.uns[key]

    n_vars_before = adata.n_vars
    n_obs_before = adata.n_obs
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_counts=20)

    # Restore saved colors
    for key, val in saved_uns_colors.items():
        adata.uns[key] = val
    if adata.n_vars < n_vars_before:
        logger.info(f"  Filtered genes (min_cells=5): {n_vars_before} -> {adata.n_vars}")
    if adata.n_obs < n_obs_before:
        logger.info(f"  Filtered cells/spots (min_counts=20): {n_obs_before} -> {adata.n_obs}")

    # --- Step 4: Remove cells/spots with NaN coordinates ---
    required_cols = ['x', 'y']
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing and 'spatial' in getattr(adata, 'obsm', {}):
        spatial = adata.obsm['spatial']
        adata.obs['x'] = spatial[:, 0]
        adata.obs['y'] = spatial[:, 1]
        logger.info(f"  Auto-created x, y from adata.obsm['spatial']")
        missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"AnnData missing required obs columns: {missing}. "
                         f"Provide x, y in adata.obs or spatial coordinates in adata.obsm['spatial'].")

    coord_mask = ~adata.obs['x'].isna() & ~adata.obs['y'].isna()
    n_nan = int((~coord_mask).sum())
    if n_nan > 0:
        adata = adata[coord_mask].copy()
        logger.info(f"  Removed {n_nan} cells/spots with NaN coordinates: {n_nan + adata.n_obs} -> {adata.n_obs}")

    logger.info(f"  Preprocessing complete: {adata.n_obs} obs x {adata.n_vars} vars")
    return adata

def load_anndata(path: Union[str, Path]) -> ad.AnnData:
    """
    Load AnnData from h5ad file.

    Parameters
    ----------
    path : str or Path
        Path to h5ad file

    Returns
    -------
    adata : AnnData
        Loaded AnnData object

    Notes
    -----
    Required obs columns: x, y
    Optional: celltype (for cell-level) or deconv_weights (for spot-level)

    For spot data (filename starts with "spot_"):
    - Should have uns keys: spot_shape, spot_diameter
    - May have obsm key: deconv_weights
    """
    path = Path(path)
    logger.info(f"Loading AnnData from: {path}")

    adata = ad.read_h5ad(path)

    # Preprocess: validate raw counts, filter genes/cells, make names unique
    adata = _preprocess_anndata(adata, path.name)

    # Detect and handle spot data
    is_spot_data = path.stem.startswith('spot_')

    if is_spot_data:
        logger.info(f"Detected spot-level data: {path.name}")

        # Check for spot metadata (optional but recommended)
        if 'spot_shape' in adata.uns and 'spot_diameter' in adata.uns:
            logger.info(f"Spot shape: {adata.uns['spot_shape']}, "
                       f"diameter: {adata.uns['spot_diameter']} μm")

        # Handle deconvolution weights
        if 'deconv_weights' in adata.obsm:
            logger.info("Found deconv_weights - creating virtual celltype from dominant type")
            deconv_weights = adata.obsm['deconv_weights']
            adata.obs['celltype'] = deconv_weights.idxmax(axis=1)
            adata.uns['has_deconv_weights'] = True
        else:
            adata.uns['has_deconv_weights'] = False

        logger.info(f"Loaded {adata.n_obs} spots × {adata.n_vars} features")
    else:
        logger.info(f"Detected cell-level data: {path.name}")
        logger.info(f"Loaded {adata.n_obs} cells × {adata.n_vars} features")

    # Log celltype info if available
    if 'celltype' in adata.obs.columns:
        n_celltypes = adata.obs['celltype'].nunique()
        logger.info(f"Found {n_celltypes} cell types")

    return adata


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from tif or npy file.

    Parameters
    ----------
    path : str or Path
        Path to image file (.tif, .tiff, .npy)

    Returns
    -------
    image : ndarray
        Image array with shape (height, width) or (height, width, channels)
    """
    path = Path(path)
    logger.info(f"Loading image from: {path}")

    if path.suffix.lower() in ['.tif', '.tiff']:
        img = Image.open(path)
        image = np.array(img)
    elif path.suffix.lower() == '.npy':
        image = np.load(path)
    else:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    logger.info(f"Loaded image with shape: {image.shape}")
    return image


# ========================================
# Image Discovery Helpers
# ========================================

def _find_all_images(dataset_dir: Path) -> List[Path]:
    """Find all image files (.tif, .tiff, .npy) in a directory."""
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.npy']:
        image_files.extend(dataset_dir.glob(ext))
    return sorted(set(image_files))


def _detect_image_type(stem: str) -> str:
    """Detect image type (he/dapi/primary) from filename stem."""
    s = stem.lower()
    if 'he' in s or 'h_e' in s:
        return 'he'
    if 'dapi' in s:
        return 'dapi'
    return 'primary'


# ========================================
# Filename Parsing
# ========================================

def parse_slice_id_from_filename(filename: str) -> Tuple[Optional[int], str]:
    """
    Parse slice_id and modality from filename.

    Patterns:
    - tissue.h5ad → (None, 'gene')
    - tissue_protein.h5ad → (None, 'protein')
    - tissue_slice_0.h5ad → (0, 'gene')
    - tissue_slice_1.h5ad → (1, 'gene')
    - tissue_slice_2_protein.h5ad → (2, 'protein')
    - spot_visium_slice_5.h5ad → (5, 'gene')

    Parameters
    ----------
    filename : str
        Filename (without path, with or without extension)

    Returns
    -------
    slice_id : int or None
        Parsed slice ID, or None if not in filename
    modality : str
        'gene' or 'protein'
    """
    stem = Path(filename).stem

    # Detect modality
    modality = 'protein' if 'protein' in stem.lower() else 'gene'

    # Remove protein suffix for slice ID parsing
    stem_clean = stem.replace('_protein', '').replace('_Protein', '')

    # Pattern: *_slice_N or *_slice_N_*
    match = re.search(r'_slice_(\d+)', stem_clean)
    if match:
        slice_id = int(match.group(1))
        return slice_id, modality

    # No slice ID in filename
    return None, modality


def detect_data_level(adata: ad.AnnData, filename: str) -> Tuple[str, str]:
    """
    Detect if data is cell-level or spot-level.

    Rules:
    1. Filename starts with 'spot_' → spot
    2. Fewer than 10,000 observations → spot (Visium heuristic)
    3. Otherwise → cell

    Parameters
    ----------
    adata : AnnData
        Loaded AnnData (used for observation count heuristic)
    filename : str
        Filename

    Returns
    -------
    data_level : str
        'cell' or 'spot'
    reason : str
        Human-readable detection reason
    """
    stem = Path(filename).stem
    if stem.startswith('spot_'):
        return 'spot', 'filename prefix "spot_"'
    # Heuristic: Visium typically has <10k spots
    if adata.n_obs < 10000:
        reason = f'{adata.n_obs} observations < 10,000 threshold'
        logger.info(f"  Auto-detected spot-level data ({reason})")
        return 'spot', reason
    return 'cell', f'{adata.n_obs} observations (cell-level)'


def find_images_for_slice(dataset_dir: Path,
                          slice_id: int,
                          modality: str,
                          tissue_prefix: str = None) -> Tuple[Dict[str, np.ndarray], List[Path]]:
    """
    Find and load images matching slice_id and modality using strict patterns.

    Gene modality patterns:
    - he.tif, dapi.tif → slice 0
    - he_slice_0.tif → slice 0
    - he_slice_1.tif → slice 1

    Protein modality patterns:
    - protein_cd3.tif → first protein slice
    - protein_cd8_slice_1.tif → slice 1 protein

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory
    slice_id : int
        Slice ID to match
    modality : str
        'gene' or 'protein'
    tissue_prefix : str, optional
        Tissue name prefix from h5ad filename

    Returns
    -------
    images : Dict[str, ndarray]
        {image_name: image_array}
        Empty dict if no images found
    matched_paths : List[Path]
        Resolved paths of image files that were successfully loaded
    """
    images = {}
    matched_paths = []

    if modality == 'gene':
        # Look for he/dapi images
        patterns = [
            f'he_slice_{slice_id}.tif',
            f'he_slice_{slice_id}.tiff',
            f'dapi_slice_{slice_id}.tif',
            f'dapi_slice_{slice_id}.tiff',
        ]

        # For slice 0, also try unnumbered files
        if slice_id == 0:
            patterns.extend(['he.tif', 'he.tiff', 'dapi.tif', 'dapi.tiff'])

        for pattern in patterns:
            matching = list(dataset_dir.glob(pattern))
            if matching:
                img_type = 'he' if 'he' in pattern else 'dapi'
                try:
                    images[img_type] = load_image(matching[0])
                    matched_paths.append(matching[0].resolve())
                    logger.info(f"Loaded {img_type} image for slice {slice_id}")
                    break  # Only load one he or dapi
                except Exception as e:
                    logger.warning(f"Failed to load {img_type} image: {e}")

    elif modality == 'protein':
        # Look for protein_*.tif images
        # First try slice-specific
        for img_file in dataset_dir.glob(f'protein_*_slice_{slice_id}.tif*'):
            protein_name = img_file.stem.replace('protein_', '').replace(f'_slice_{slice_id}', '')
            try:
                images[protein_name] = load_image(img_file)
                matched_paths.append(img_file.resolve())
                logger.info(f"Loaded protein image: {protein_name} for slice {slice_id}")
            except Exception as e:
                logger.warning(f"Failed to load protein image {img_file}: {e}")

        # If no slice-specific images and this is slice 1 (first protein slice),
        # try unnumbered protein images
        if not images and slice_id == 1:
            for img_file in dataset_dir.glob('protein_*.tif*'):
                if '_slice_' not in img_file.stem:  # Skip slice-specific
                    protein_name = img_file.stem.replace('protein_', '')
                    try:
                        images[protein_name] = load_image(img_file)
                        matched_paths.append(img_file.resolve())
                        logger.info(f"Loaded protein image: {protein_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load protein image {img_file}: {e}")

    if not images:
        logger.info(f"No strict image match for slice {slice_id} ({modality})")

    return images, matched_paths


# ========================================
# Flexible Image Matching
# ========================================

def _assign_images_flexible(dataset_dir: Path,
                            slices: Dict[int, DataSlice],
                            stem_map: Dict[int, str],
                            used_image_paths: set):
    """
    Flexible image matching fallback for slices without images.

    Called after strict matching. Uses filename similarity (greedy best-match)
    to assign unmatched image files to slices that didn't get images.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory
    slices : Dict[int, DataSlice]
        Already-loaded slices (modified in-place)
    stem_map : Dict[int, str]
        Mapping of slice_id → h5ad filename stem
    used_image_paths : set
        Resolved paths of images already assigned by strict matching
    """
    from difflib import SequenceMatcher

    all_images = _find_all_images(dataset_dir)
    if not all_images:
        return

    available = [p for p in all_images if p.resolve() not in used_image_paths]
    if not available:
        return

    slices_needing = [(sid, stem_map[sid]) for sid in sorted(slices.keys())
                      if not slices[sid].images]
    if not slices_needing:
        return

    logger.info(f"Flexible image matching: {len(available)} unmatched image(s), "
                f"{len(slices_needing)} slice(s) need images")

    # Special case: single slice needs image + single image available → always pair
    if len(slices_needing) == 1 and len(available) == 1:
        sid = slices_needing[0][0]
        img_path = available[0]
        try:
            img = load_image(img_path)
            img_type = _detect_image_type(img_path.stem)
            slices[sid].images[img_type] = img
            slices[sid].metadata['image_match'] = 'flexible (single pair)'
            logger.info(f"  Flexible match: {img_path.name} → slice {sid}")
        except Exception as e:
            logger.warning(f"  Failed to load image {img_path}: {e}")
        return

    # Greedy matching: compute all similarities, assign best-first
    pairs = []
    for sid, stem in slices_needing:
        for img_path in available:
            score = SequenceMatcher(None, stem.lower(), img_path.stem.lower()).ratio()
            pairs.append((score, sid, img_path))

    pairs.sort(key=lambda x: -x[0])  # Best matches first

    assigned_slices = set()
    assigned_images = set()

    for score, sid, img_path in pairs:
        if sid in assigned_slices or img_path in assigned_images:
            continue
        if score < 0.3:
            continue  # Too dissimilar
        try:
            img = load_image(img_path)
            img_type = _detect_image_type(img_path.stem)
            slices[sid].images[img_type] = img
            slices[sid].metadata['image_match'] = f'flexible (similarity: {score:.2f})'
            assigned_slices.add(sid)
            assigned_images.add(img_path)
            logger.info(f"  Flexible match: {img_path.name} → slice {sid} "
                        f"(similarity: {score:.2f})")
        except Exception as e:
            logger.warning(f"  Failed to load image {img_path}: {e}")

    # Report unmatched
    unmatched_slices = [sid for sid, _ in slices_needing if sid not in assigned_slices]
    if unmatched_slices:
        logger.info(f"  No image match for slice(s): {unmatched_slices}")


# ========================================
# Main Dataset Loader
# ========================================

def load_dataset(dataset_dir: Union[str, Path]) -> Dict[int, DataSlice]:
    """
    Load spatial dataset into unified DataSlice structure.

    Two-mode loading:
    1. Strict: Parses slice IDs from filenames (_slice_N), matches images
       by rigid naming patterns (he_slice_0.tif, dapi.tif, etc.).
    2. Flexible fallback: For images not matched by strict rules, uses
       filename similarity to pair remaining images with slices.

    Auto-detects format and assigns slice IDs:
    - Single h5ad → slice_id=0
    - Multiple h5ad with _slice_N → use N as slice_id
    - Multiple h5ad without _slice_N → sequential IDs (0, 1, 2, ...)
    - Protein file → gets next available slice_id (usually 1)

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing h5ad and image files

    Returns
    -------
    slices : Dict[int, DataSlice]
        {slice_id: DataSlice}

    Examples
    --------
    >>> # Strict naming
    >>> slices = load_dataset("/path/to/dataset")  # tissue_slice_0.h5ad, he_slice_0.tif
    >>> # Flexible naming (any filenames work)
    >>> slices = load_dataset("/path/to/data")  # brain.h5ad, brain_image.tif

    Raises
    ------
    FileNotFoundError
        If no h5ad files found
    ValueError
        If slice IDs conflict
    """
    dataset_dir = Path(dataset_dir)
    logger.info(f"Loading dataset from: {dataset_dir}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Find all h5ad files
    h5ad_files = sorted(dataset_dir.glob("*.h5ad"))

    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {dataset_dir}")

    logger.info(f"Found {len(h5ad_files)} h5ad file(s)")

    # Track assigned IDs to avoid conflicts
    slices = {}
    next_auto_id = 0
    stem_map = {}  # slice_id → h5ad stem (for flexible image matching)
    used_image_paths = set()  # Track images matched by strict rules

    for h5ad_file in h5ad_files:
        stem = h5ad_file.stem

        # Parse slice_id and modality from filename
        slice_id, modality = parse_slice_id_from_filename(stem)

        # Auto-assign ID if not in filename
        if slice_id is None:
            # For protein files without slice ID, assign next ID
            # For gene files without slice ID, assign 0 if available
            if modality == 'protein':
                slice_id = max(1, next_auto_id)  # Protein starts at 1
            else:
                slice_id = 0 if 0 not in slices else next_auto_id

        # Check for ID conflict
        if slice_id in slices:
            raise ValueError(
                f"Slice ID conflict: {slice_id} already used. "
                f"Files: {h5ad_file.name} and slice {slice_id} already loaded"
            )

        # Update next available ID
        next_auto_id = max(next_auto_id, slice_id + 1)

        # Load AnnData
        adata = load_anndata(h5ad_file)

        # Detect data level (with reason for UI feedback)
        data_level, detection_reason = detect_data_level(adata, stem)

        # Strict image matching
        images, matched_paths = find_images_for_slice(
            dataset_dir, slice_id, modality, tissue_prefix=stem)
        used_image_paths.update(matched_paths)

        # Create DataSlice
        slice_data = DataSlice(
            slice_id=slice_id,
            modality=modality,
            data_level=data_level,
            adata=adata,
            images=images,
            metadata={
                'tissue_name': stem,
                'data_level_detection': detection_reason,
                'image_match': 'strict' if images else 'none',
            }
        )

        slices[slice_id] = slice_data
        stem_map[slice_id] = stem

        logger.info(f"Loaded slice {slice_id}: {modality}, {data_level} "
                     f"({detection_reason}), "
                     f"{slice_data.n_obs} obs × {slice_data.n_vars} vars")

    # Flexible image matching fallback for slices without images
    _assign_images_flexible(dataset_dir, slices, stem_map, used_image_paths)

    logger.info(f"Successfully loaded {len(slices)} slice(s): {sorted(slices.keys())}")

    return slices


# ========================================
# Legacy compatibility function
# ========================================

def load_data(adata_path: Union[str, Path],
              image_path: Union[str, Path]) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Load AnnData and image together (legacy function for backward compatibility).

    This is kept for old code that directly loads adata + image.
    New code should use load_dataset() instead.

    Parameters
    ----------
    adata_path : str or Path
        Path to h5ad file
    image_path : str or Path
        Path to image file

    Returns
    -------
    adata : AnnData
        Loaded AnnData object
    image : ndarray
        Loaded image array
    """
    adata = load_anndata(adata_path)
    image = load_image(image_path)

    # Validate coordinate ranges
    if len(image.shape) == 3:
        img_height, img_width, _ = image.shape
    else:
        img_height, img_width = image.shape

    max_x = adata.obs['x'].max()
    max_y = adata.obs['y'].max()

    if max_x > img_width or max_y > img_height:
        logger.warning(
            f"Cell coordinates exceed image bounds: "
            f"max ({max_x:.1f}, {max_y:.1f}) vs image size ({img_width}, {img_height})"
        )

    logger.info("Data loaded successfully")
    return adata, image

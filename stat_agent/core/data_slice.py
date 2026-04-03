"""
DataSlice: Core data structure for spatial transcriptomics.

Each DataSlice represents a single, independent unit of spatial data with:
- Unique slice_id (0, 1, 2, ...)
- Modality ('gene' or 'protein')
- Data level ('cell' or 'spot')
- AnnData with expression matrix
- Optional images (H&E, DAPI, protein channels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import anndata as ad
import numpy as np


@dataclass
class DataSlice:
    """
    A single slice of spatial transcriptomics data.

    Represents one independent data unit with unique slice_id.
    Each slice can be:
    - Gene expression (modality='gene', data_level='cell' or 'spot')
    - Protein expression (modality='protein', data_level='cell')

    Attributes
    ----------
    slice_id : int
        Unique identifier (typically parsed from filename)
    modality : str
        'gene' or 'protein'
    data_level : str
        'cell' (single-cell) or 'spot' (spatial transcriptomics spots)
    adata : AnnData
        Expression matrix (cells/spots × genes/proteins)
    images : Dict[str, ndarray]
        Named images, e.g., {'he': array, 'dapi': array}
        Empty dict if no images available
    metadata : Dict[str, Any]
        Additional information (tissue_name, etc.)

    Examples
    --------
    >>> # Cell-level gene expression
    >>> slice_0 = DataSlice(
    ...     slice_id=0,
    ...     modality='gene',
    ...     data_level='cell',
    ...     adata=adata_gene,
    ...     images={'he': he_image},
    ...     metadata={'tissue_name': 'breast_cancer_rep1'}
    ... )

    >>> # Spot-level gene expression (Visium)
    >>> slice_1 = DataSlice(
    ...     slice_id=1,
    ...     modality='gene',
    ...     data_level='spot',
    ...     adata=adata_visium,
    ...     images={'he': he_image},
    ...     metadata={'tissue_name': 'visium_sample'}
    ... )

    >>> # Protein expression
    >>> slice_2 = DataSlice(
    ...     slice_id=2,
    ...     modality='protein',
    ...     data_level='cell',
    ...     adata=adata_protein,
    ...     images={'cd3': cd3_img, 'cd8': cd8_img},
    ...     metadata={'tissue_name': 'protein_panel'}
    ... )
    """

    slice_id: int
    modality: str  # 'gene' or 'protein'
    data_level: str  # 'cell' or 'spot'
    adata: ad.AnnData
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data after initialization."""
        # Validate modality
        if self.modality not in ['gene', 'protein']:
            raise ValueError(f"Invalid modality: {self.modality}. Must be 'gene' or 'protein'")

        # Validate data_level
        if self.data_level not in ['cell', 'spot']:
            raise ValueError(f"Invalid data_level: {self.data_level}. Must be 'cell' or 'spot'")

        # Validate AnnData has required columns
        if 'x' not in self.adata.obs.columns or 'y' not in self.adata.obs.columns:
            raise ValueError("AnnData must have 'x' and 'y' columns in obs")

    # ========================================
    # Type checking properties
    # ========================================

    @property
    def is_gene(self) -> bool:
        """Check if this slice contains gene expression data."""
        return self.modality == 'gene'

    @property
    def is_protein(self) -> bool:
        """Check if this slice contains protein expression data."""
        return self.modality == 'protein'

    @property
    def is_cell_level(self) -> bool:
        """Check if this is cell-level data (single-cell resolution)."""
        return self.data_level == 'cell'

    @property
    def is_spot_level(self) -> bool:
        """Check if this is spot-level data (spatial transcriptomics spots)."""
        return self.data_level == 'spot'

    # ========================================
    # Data access properties
    # ========================================

    @property
    def n_obs(self) -> int:
        """Number of observations (cells or spots)."""
        return int(self.adata.n_obs)

    @property
    def n_vars(self) -> int:
        """Number of variables (genes or proteins)."""
        return int(self.adata.n_vars)

    @property
    def primary_image(self) -> Optional[np.ndarray]:
        """
        Get the first available image.

        Returns None if no images are available.
        Useful for quick access when only one image is expected.
        """
        return next(iter(self.images.values())) if self.images else None

    @property
    def coordinate_range(self) -> Dict[str, tuple]:
        """
        Get spatial coordinate range.

        Returns
        -------
        dict
            {'x': (min, max), 'y': (min, max)}
        """
        return {
            'x': (self.adata.obs['x'].min(), self.adata.obs['x'].max()),
            'y': (self.adata.obs['y'].min(), self.adata.obs['y'].max())
        }

    # ========================================
    # Annotation checking
    # ========================================

    def has_celltype(self) -> bool:
        """
        Check if celltype annotations exist.

        Returns True only if:
        - 'celltype' column exists in adata.obs
        - At least one non-null celltype value exists
        """
        if 'celltype' not in self.adata.obs.columns:
            return False
        return bool(self.adata.obs['celltype'].notna().any())

    def has_niche_labels(self) -> bool:
        """Check if niche labels exist."""
        if 'niche_label' not in self.adata.obs.columns:
            return False
        return bool(self.adata.obs['niche_label'].notna().any())

    def has_deconv_weights(self) -> bool:
        """Check if deconvolution weights exist (for spot data)."""
        return 'deconv_weights' in self.adata.obsm

    def has_celltype_colors(self) -> bool:
        """Check if celltype colors are defined."""
        return 'celltype_colors' in self.adata.uns

    def get_celltype_colors(self) -> Optional[Dict[str, str]]:
        """
        Get celltype colors from .uns['celltype_colors'].

        Returns
        -------
        dict or None
            Mapping of celltype name to color (hex format), or None if not set
        """
        return self.adata.uns.get('celltype_colors')

    def set_celltype_colors(self, colors: Dict[str, str]):
        """
        Set celltype colors in .uns['celltype_colors'].

        Parameters
        ----------
        colors : Dict[str, str]
            Mapping of celltype name to color (hex format like '#RRGGBB' or 'rgb(r,g,b)')
        """
        self.adata.uns['celltype_colors'] = colors

    def ensure_celltype_colors(self):
        """
        Auto-generate celltype colors if not present.

        Only generates colors if:
        1. Celltype annotations exist
        2. Colors are not already defined

        Uses HSL golden ratio spacing for visually distinct colors.
        """
        if not self.has_celltype():
            return  # No celltypes to color

        if self.has_celltype_colors():
            return  # Already has colors

        # Get unique celltypes
        celltypes = self.adata.obs['celltype'].unique()
        valid_celltypes = sorted([ct for ct in celltypes if isinstance(ct, str)])

        if not valid_celltypes:
            return  # No valid celltypes

        # Generate colors using HSL golden ratio
        colors = self._generate_celltype_colors(valid_celltypes)
        self.set_celltype_colors(colors)

    def _generate_celltype_colors(self, celltypes: list) -> Dict[str, str]:
        """
        Generate distinct colors for celltypes using HSL golden ratio spacing.

        Parameters
        ----------
        celltypes : list
            List of celltype names

        Returns
        -------
        dict
            Mapping of celltype to color (rgb format)
        """
        import numpy as np

        colors = {}
        golden_ratio = 0.618033988749895
        hue = np.random.random()  # Start with random hue

        for celltype in celltypes:
            hue += golden_ratio
            hue %= 1
            saturation = 0.6 + np.random.random() * 0.2  # 0.6-0.8
            lightness = 0.45 + np.random.random() * 0.15  # 0.45-0.6

            # Convert HSL to RGB
            r, g, b = self._hsl_to_rgb(hue, saturation, lightness)
            colors[celltype] = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

        return colors

    def _hsl_to_rgb(self, h: float, s: float, l: float) -> tuple:
        """Convert HSL to RGB."""
        def hue2rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        if s == 0:
            r = g = b = l
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue2rgb(p, q, h + 1/3)
            g = hue2rgb(p, q, h)
            b = hue2rgb(p, q, h - 1/3)

        return r, g, b

    # ========================================
    # Summary and display
    # ========================================

    def get_summary(self) -> Dict[str, Any]:
        """
        Get slice summary for display and logging.

        Returns
        -------
        dict
            Summary information about this slice
        """
        summary = {
            'slice_id': self.slice_id,
            'modality': self.modality,
            'data_level': self.data_level,
            'data_level_detection': self.metadata.get('data_level_detection', ''),
            'n_obs': self.n_obs,
            'n_vars': self.n_vars,
            'has_celltype': self.has_celltype(),
            'celltypes': sorted(self.adata.obs['celltype'].unique().tolist()) if self.has_celltype() else [],
            'has_celltype_colors': self.has_celltype_colors(),
            'image_count': len(self.images),
            'image_names': list(self.images.keys()),
            'image_match': self.metadata.get('image_match', ''),
            'tissue_name': self.metadata.get('tissue_name', f'slice_{self.slice_id}'),
        }

        # Add spot-specific info
        if self.is_spot_level:
            summary['has_deconv_weights'] = self.has_deconv_weights()
            if 'spot_shape' in self.adata.uns:
                summary['spot_shape'] = str(self.adata.uns['spot_shape'])
            if 'spot_diameter' in self.adata.uns:
                summary['spot_diameter'] = float(self.adata.uns['spot_diameter'])

        return summary

    def __repr__(self) -> str:
        """String representation for debugging."""
        tissue_name = self.metadata.get('tissue_name', 'unknown')
        return (f"DataSlice(slice_id={self.slice_id}, modality='{self.modality}', "
                f"data_level='{self.data_level}', n_obs={self.n_obs}, "
                f"n_vars={self.n_vars}, tissue='{tissue_name}')")

"""
Simplified session management for spatial transcriptomics data.

New unified architecture:
- session.slices: Dict[int, DataSlice] - all data storage
- session.current_slice_id: int - UI state (frontend only)
- Per-slice ROIs: {roi_name: {slice_id: filtered_adata}}
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Iterator
from pathlib import Path
from datetime import datetime

import anndata as ad
import numpy as np

from stat_agent.core.data_slice import DataSlice
from stat_agent.core.roi_manager import ROIManager, ROI
from stat_agent.functions.io import load_dataset, load_data

logger = logging.getLogger(__name__)


class SimpleSession:
    """
    Session for spatial transcriptomics analysis.

    Unified architecture with explicit slice access:
    - Backend: Always use get_slice(slice_id) for explicit access
    - Frontend: Use current_slice_id + current_slice() for UI state

    Attributes
    ----------
    name : str
        Session name
    slices : Dict[int, DataSlice]
        All loaded slices {slice_id: DataSlice}
    current_slice_id : int, optional
        Currently selected slice for UI display (frontend only)
    roi_manager : ROIManager
        ROI tracking and manipulation
    roi_subsets : Dict[str, ROI]
        ROI objects with filtered data: {roi_name: ROI}
    celltype_colors : Dict[str, str], optional
        Shared celltype colors across all slices
    metadata : Dict
        Session metadata

    Examples
    --------
    >>> # Backend usage (explicit)
    >>> session = SimpleSession()
    >>> session.load_dataset('/path/to/data')
    >>> slice_0 = session.get_slice(0)
    >>> print(slice_0.modality, slice_0.data_level)

    >>> # ROI usage (consistent with slice API)
    >>> roi = session.get_roi('tumor_region')
    >>> print(f"ROI on slice {roi.slice_id}: {roi.n_obs} cells")
    >>> analyze(roi.adata)

    >>> # Frontend usage (current selection)
    >>> session.current_slice_id = 1
    >>> current = session.current_slice()
    >>> image = current.primary_image
    """

    def __init__(self, name: str = "session"):
        """
        Initialize a new session.

        Parameters
        ----------
        name : str
            Session name
        """
        self.name = name

        # Core data storage (NEW: unified structure)
        self.slices: Dict[int, DataSlice] = {}

        # UI state (frontend only)
        self.current_slice_id: Optional[int] = None

        # ROI management (per-slice)
        self.roi_manager = ROIManager()
        self.roi_subsets: Dict[str, ROI] = {}  # {roi_name: ROI object with filtered data}

        # Shared celltype colors (consistent across all slices)
        self.celltype_colors: Optional[Dict[str, str]] = None

        # Metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "name": name,
        }

        logger.info(f"Created session: {name}")

    # ========================================
    # BACKEND API (Explicit slice access)
    # ========================================

    def get_slice(self, slice_id: int) -> DataSlice:
        """
        Get a specific slice by ID.

        BACKEND: Always use this method (explicit).
        FRONTEND: Don't use this (use current_slice() instead).

        Parameters
        ----------
        slice_id : int
            Slice ID

        Returns
        -------
        slice : DataSlice
            The requested slice

        Raises
        ------
        ValueError
            If slice_id not found

        Examples
        --------
        >>> slice_0 = session.get_slice(0)
        >>> adata = slice_0.adata
        >>> if slice_0.is_cell_level:
        ...     run_clustering(adata)
        """
        if slice_id not in self.slices:
            available = sorted(self.slices.keys())
            raise ValueError(f"Slice {slice_id} not found. Available: {available}")
        return self.slices[slice_id]

    def iter_slices(self,
                    modality: Optional[str] = None,
                    data_level: Optional[str] = None) -> Iterator[DataSlice]:
        """
        Iterate over slices with optional filtering.

        BACKEND: Use for multi-slice workflows.

        Parameters
        ----------
        modality : str, optional
            Filter by 'gene' or 'protein'
        data_level : str, optional
            Filter by 'cell' or 'spot'

        Yields
        ------
        slice : DataSlice
            Matching slices

        Examples
        --------
        >>> # All slices
        >>> for slice in session.iter_slices():
        ...     analyze(slice.adata)

        >>> # Only gene data
        >>> for slice in session.iter_slices(modality='gene'):
        ...     gene_analysis(slice.adata)

        >>> # Only cell-level data
        >>> for slice in session.iter_slices(data_level='cell'):
        ...     clustering(slice.adata)
        """
        for slice in self.slices.values():
            if modality is not None and slice.modality != modality:
                continue
            if data_level is not None and slice.data_level != data_level:
                continue
            yield slice

    def add_slice(self, slice: DataSlice):
        """
        Add a new slice to the session.

        Parameters
        ----------
        slice : DataSlice
            Slice to add

        Notes
        -----
        If slice_id already exists, it will be overwritten with a warning.
        """
        if slice.slice_id in self.slices:
            logger.warning(f"Overwriting existing slice {slice.slice_id}")

        self.slices[slice.slice_id] = slice

        # Set as current if first slice
        if self.current_slice_id is None:
            self.current_slice_id = slice.slice_id

        logger.info(f"Added slice {slice.slice_id} ({slice.modality}, {slice.data_level})")

    # ========================================
    # FRONTEND API (UI helpers - current selection)
    # ========================================

    def current_slice(self) -> Optional[DataSlice]:
        """
        Get currently selected slice for UI rendering.

        FRONTEND: Use this for canvas/ROI display.
        BACKEND: DO NOT USE - use get_slice(id) instead.

        Returns
        -------
        slice : DataSlice or None
            Current slice, or None if no slice selected

        Examples
        --------
        >>> # Frontend: Display current slice
        >>> current = session.current_slice()
        >>> if current:
        ...     image = current.primary_image
        ...     cells = current.adata.obs[['x', 'y']]
        """
        if self.current_slice_id is None:
            return None
        return self.slices.get(self.current_slice_id)

    # ========================================
    # Data Loading
    # ========================================

    def load_dataset(self, dataset_dir: str):
        """
        Load spatial dataset from directory.

        Auto-detects format and loads all slices.

        Parameters
        ----------
        dataset_dir : str
            Path to dataset directory

        Notes
        -----
        This is the recommended loading method for all data formats.
        """
        dataset_dir = Path(dataset_dir)

        # Load all slices
        loaded_slices = load_dataset(dataset_dir)

        # Add to session
        for slice_id, slice_data in loaded_slices.items():
            self.add_slice(slice_data)

        # Initialize shared celltype colors
        self._initialize_celltype_colors()

        # Update metadata
        self.metadata.update({
            'dataset_dir': str(dataset_dir),
            'n_slices': len(self.slices),
            'slice_ids': sorted(self.slices.keys()),
        })

        logger.info(f"Loaded dataset with {len(self.slices)} slice(s): {self.slice_ids}")

    def load_data_legacy(self, adata_path: str, image_path: str):
        """
        Load single AnnData and image (legacy method for backward compatibility).

        Creates a single slice with slice_id=0.

        Parameters
        ----------
        adata_path : str
            Path to h5ad file
        image_path : str
            Path to image file

        Notes
        -----
        This is the old loading method. Prefer load_dataset() for new code.
        """
        from stat.functions.io import parse_slice_id_from_filename, detect_data_level

        # Load using legacy function
        adata, image = load_data(adata_path, image_path)

        # Parse filename to determine modality
        _, modality = parse_slice_id_from_filename(adata_path)
        data_level = detect_data_level(adata, adata_path)

        # Create DataSlice
        slice_data = DataSlice(
            slice_id=0,
            modality=modality,
            data_level=data_level,
            adata=adata,
            images={'primary': image} if image is not None else {},
            metadata={'tissue_name': Path(adata_path).stem}
        )

        # Add to session
        self.add_slice(slice_data)

        # Initialize celltype colors
        self._initialize_celltype_colors()

        # Update metadata
        self.metadata.update({
            'adata_path': str(adata_path),
            'image_path': str(image_path),
        })

        logger.info("Loaded data using legacy method")

    # ========================================
    # ROI Management (per-slice)
    # ========================================

    def create_roi(self, roi_name: str, slice_id: int, roi_definition: Dict):
        """
        Create ROI for a specific slice.

        Parameters
        ----------
        roi_name : str
            Name of the ROI
        slice_id : int
            Which slice this ROI belongs to
        roi_definition : Dict
            ROI geometry definition with 'type' and coordinates:
            - bbox: {'type': 'bbox', 'x_min', 'x_max', 'y_min', 'y_max'}
            - circle: {'type': 'circle', 'center_x', 'center_y', 'radius'}
            - polygon: {'type': 'polygon', 'vertices': [(x,y), ...]}  (used by freehand)

        Raises
        ------
        ValueError
            If slice_id not found or invalid ROI type
        """
        # Validate slice exists
        slice = self.get_slice(slice_id)

        # Convert roi_definition dict to shapely geometry and add to ROI manager
        roi_type = roi_definition.get('type', 'bbox')

        if roi_type == 'bbox':
            roi = self.roi_manager.add_bbox_roi(
                name=roi_name,
                min_x=roi_definition['x_min'],
                min_y=roi_definition['y_min'],
                max_x=roi_definition['x_max'],
                max_y=roi_definition['y_max'],
                slice_id=slice_id,  # Use int directly
                modality=slice.modality
            )
        elif roi_type == 'circle':
            roi = self.roi_manager.add_circle_roi(
                name=roi_name,
                center_x=roi_definition['center_x'],
                center_y=roi_definition['center_y'],
                radius=roi_definition['radius'],
                slice_id=slice_id,  # Use int directly
                modality=slice.modality
            )
        elif roi_type == 'polygon':
            from shapely.geometry import Polygon
            roi = self.roi_manager.add_roi(
                name=roi_name,
                geometry=Polygon(roi_definition['vertices']),
                slice_id=slice_id,
                modality=slice.modality
            )
        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")

        # Filter data for this slice
        filtered_adata = self._filter_by_roi(slice.adata, roi_definition)

        # Store complete ROI object with filtered data
        roi.adata = filtered_adata
        self.roi_subsets[roi_name] = roi

        logger.info(f"Created ROI '{roi_name}' on slice {slice_id}: "
                   f"{filtered_adata.n_obs} cells/spots")

    def get_roi(self, roi_name: str) -> Optional[ROI]:
        """
        Get ROI object by name.

        BACKEND: Use this for ROI access (consistent with get_slice API).

        Parameters
        ----------
        roi_name : str
            ROI name

        Returns
        -------
        roi : ROI or None
            ROI object with filtered data, or None if not found

        Examples
        --------
        >>> roi = session.get_roi('tumor_region')
        >>> if roi:
        ...     print(f"ROI on slice {roi.slice_id}")
        ...     print(f"Cells: {roi.n_obs}")
        ...     analyze(roi.adata)
        """
        return self.roi_subsets.get(roi_name)

    def get_roi_data(self, roi_name: str, slice_id: int = None) -> Optional[ad.AnnData]:
        """
        Get ROI-filtered data (deprecated - use get_roi() instead).

        DEPRECATED: Use get_roi() for consistent API.

        Parameters
        ----------
        roi_name : str
            ROI name
        slice_id : int, optional
            Slice ID (for backward compatibility, not needed anymore)

        Returns
        -------
        adata : AnnData or None
            Filtered data, or None if ROI doesn't exist
        """
        roi = self.get_roi(roi_name)
        if roi is None:
            return None

        # If slice_id specified, verify it matches
        if slice_id is not None and roi.slice_id != slice_id:
            return None

        return roi.adata

    def get_roi_for_current_slice(self, roi_name: str) -> Optional[ad.AnnData]:
        """
        Get ROI data for current slice (frontend helper).

        FRONTEND: Use this for ROI display on current canvas.
        BACKEND: Use get_roi_data(roi_name, slice_id) instead.

        Parameters
        ----------
        roi_name : str
            ROI name

        Returns
        -------
        adata : AnnData or None
            Filtered data for current slice
        """
        if self.current_slice_id is None:
            return None
        return self.get_roi_data(roi_name, self.current_slice_id)

    def _filter_by_roi(self, adata: ad.AnnData, roi_def: Dict) -> ad.AnnData:
        """
        Filter AnnData by ROI geometry.

        Parameters
        ----------
        adata : AnnData
            Data to filter
        roi_def : Dict
            ROI definition with 'type' and geometry

        Returns
        -------
        filtered : AnnData
            Filtered copy of adata
        """
        roi_type = roi_def.get('type', 'bbox')

        if roi_type == 'bbox':
            # Bounding box filtering
            x_min = roi_def['x_min']
            x_max = roi_def['x_max']
            y_min = roi_def['y_min']
            y_max = roi_def['y_max']

            mask = (
                (adata.obs['x'] >= x_min) &
                (adata.obs['x'] <= x_max) &
                (adata.obs['y'] >= y_min) &
                (adata.obs['y'] <= y_max)
            )

        elif roi_type == 'circle':
            # Circle filtering
            cx = roi_def['center_x']
            cy = roi_def['center_y']
            radius = roi_def['radius']

            distances = np.sqrt(
                (adata.obs['x'] - cx) ** 2 +
                (adata.obs['y'] - cy) ** 2
            )
            mask = distances <= radius

        elif roi_type == 'polygon':
            # Polygon filtering (using matplotlib path)
            from matplotlib.path import Path as MplPath
            vertices = roi_def['vertices']  # List of (x, y) tuples
            path = MplPath(vertices)
            points = adata.obs[['x', 'y']].values
            mask = path.contains_points(points)

        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")

        return adata[mask].copy()

    # ========================================
    # Celltype Color Management
    # ========================================

    @staticmethod
    def _normalize_color_to_rgb(color_str: str) -> str:
        """Convert a color string (hex, named, or rgb) to 'rgb(r, g, b)' format."""
        if not isinstance(color_str, str):
            return None

        color_str = color_str.strip()

        # Already in rgb(...) format
        if color_str.startswith('rgb('):
            return color_str

        # Hex format: #RGB, #RRGGBB
        if color_str.startswith('#'):
            hex_str = color_str[1:]
            if len(hex_str) == 3:
                hex_str = ''.join(c * 2 for c in hex_str)
            if len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return f'rgb({r}, {g}, {b})'

        # Try matplotlib color name as fallback
        try:
            from matplotlib.colors import to_rgb
            r, g, b = to_rgb(color_str)
            return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
        except (ImportError, ValueError):
            pass

        return None

    def _extract_existing_colors(self, slice_obj) -> dict:
        """
        Extract existing celltype colors from a slice's adata.uns.

        Handles scanpy format (list/array of colors ordered by categories)
        and dict format (celltype -> color mapping).

        Returns dict of celltype -> 'rgb(r, g, b)' or empty dict.
        """
        if 'celltype_colors' not in slice_obj.adata.uns:
            return {}

        raw_colors = slice_obj.adata.uns['celltype_colors']

        # Case 1: Already a dict (STAT format)
        if isinstance(raw_colors, dict):
            converted = {}
            for ct, color in raw_colors.items():
                rgb = self._normalize_color_to_rgb(color)
                if rgb:
                    converted[ct] = rgb
            return converted

        # Case 2: List/array of colors (scanpy format) — ordered by categories
        if hasattr(raw_colors, '__len__') and not isinstance(raw_colors, str):
            color_list = list(raw_colors)
            # Match colors to category order
            obs_col = slice_obj.adata.obs.get('celltype')
            if obs_col is None:
                return {}

            if hasattr(obs_col, 'cat') and hasattr(obs_col.cat, 'categories'):
                categories = list(obs_col.cat.categories)
            else:
                categories = sorted([ct for ct in obs_col.unique() if isinstance(ct, str)])

            converted = {}
            for i, cat in enumerate(categories):
                if i < len(color_list):
                    rgb = self._normalize_color_to_rgb(str(color_list[i]))
                    if rgb:
                        converted[cat] = rgb
            return converted

        return {}

    def _initialize_celltype_colors(self):
        """
        Initialize celltype colors across all slices.

        Respects pre-existing colors in adata.uns['celltype_colors']:
        1. Extracts existing colors from each slice (scanpy list or dict format)
        2. Only generates new colors for celltypes without existing colors
        3. Stores normalized colors back in each slice's .uns['celltype_colors']
        4. Also keeps session.celltype_colors for backward compatibility
        """
        if not self.slices:
            return

        # Collect all unique celltypes and existing colors from all slices
        all_celltypes = set()
        existing_colors = {}  # celltype -> rgb string (from first slice that has it)

        for slice_obj in self.slices.values():
            if slice_obj.has_celltype():
                celltypes = slice_obj.adata.obs['celltype'].unique()
                valid_celltypes = [ct for ct in celltypes if isinstance(ct, str)]
                all_celltypes.update(valid_celltypes)

                # Extract pre-existing colors from this slice
                slice_existing = self._extract_existing_colors(slice_obj)
                for ct, color in slice_existing.items():
                    if ct not in existing_colors:
                        existing_colors[ct] = color

        if not all_celltypes:
            logger.info("No celltypes found in any slice")
            return

        all_celltypes = sorted(list(all_celltypes))

        if existing_colors:
            logger.info(f"Found pre-existing colors for {len(existing_colors)} celltypes: {list(existing_colors.keys())[:5]}...")

        # Start with existing colors, generate only for missing ones
        colors = dict(existing_colors)

        missing_celltypes = [ct for ct in all_celltypes if ct not in colors]
        if missing_celltypes:
            golden_ratio = 0.618033988749895
            hue = np.random.random()
            for celltype in missing_celltypes:
                hue += golden_ratio
                hue %= 1
                saturation = 0.6 + np.random.random() * 0.2
                lightness = 0.45 + np.random.random() * 0.15
                r, g, b = self._hsl_to_rgb(hue, saturation, lightness)
                colors[celltype] = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
            logger.info(f"Generated colors for {len(missing_celltypes)} celltypes without existing colors")

        # Store colors at session level
        self.celltype_colors = colors

        # Store normalized colors in each slice's .uns['celltype_colors']
        for slice_obj in self.slices.values():
            if slice_obj.has_celltype():
                slice_celltypes = slice_obj.adata.obs['celltype'].unique()
                valid_slice_celltypes = [ct for ct in slice_celltypes if isinstance(ct, str)]
                slice_colors = {ct: colors[ct] for ct in valid_slice_celltypes if ct in colors}
                slice_obj.set_celltype_colors(slice_colors)

        logger.info(f"Initialized colors for {len(colors)} cell types across {len(self.slices)} slices ({len(existing_colors)} from data, {len(missing_celltypes)} generated)")

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
    # Convenience Properties
    # ========================================

    @property
    def has_data(self) -> bool:
        """Check if session has any data."""
        return len(self.slices) > 0

    @property
    def n_slices(self) -> int:
        """Total number of slices."""
        return len(self.slices)

    @property
    def slice_ids(self) -> List[int]:
        """List of all slice IDs (sorted)."""
        return sorted(self.slices.keys())

    @property
    def modalities(self) -> List[str]:
        """List of unique modalities across all slices."""
        return sorted(set(s.modality for s in self.slices.values()))

    @property
    def data_levels(self) -> List[str]:
        """List of unique data levels across all slices."""
        return sorted(set(s.data_level for s in self.slices.values()))

    def get_slices_by_modality(self, modality: str) -> List[DataSlice]:
        """Get all slices of a specific modality."""
        return list(self.iter_slices(modality=modality))

    def get_slices_by_level(self, data_level: str) -> List[DataSlice]:
        """Get all slices of a specific data level."""
        return list(self.iter_slices(data_level=data_level))

    # ========================================
    # Summary & Info
    # ========================================

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive session summary for planning context.

        This method returns a CLEAN, non-redundant summary for the agent's
        planning prompt. All per-slice details are in the 'slices' field.

        Returns
        -------
        summary : dict
            Session information including all slices
            Simplified structure for planning context

        Examples
        --------
        >>> summary = session.get_summary()
        >>> print(f"Session has {summary['n_slices']} slices")
        >>> for slice_info in summary['slices']:
        ...     print(f"Slice {slice_info['slice_id']}: {slice_info['modality']}")
        """
        # Build simple, non-redundant summary
        summary = {
            # Core session info
            'name': self.name,
            'n_slices': self.n_slices,
            'slice_ids': self.slice_ids,
            'modalities': self.modalities,
            'data_levels': self.data_levels,

            # Per-slice details (all info is here!)
            'slices': [s.get_summary() for s in self.slices.values()],

            # ROI info
            'n_rois': len(self.roi_manager.rois),
            'rois': [
                {
                    'name': roi_name,
                    'slice_id': roi.slice_id,
                    'modality': roi.modality,
                    'n_obs': roi.adata.n_obs if roi.adata is not None else 0,
                }
                for roi_name, roi in self.roi_subsets.items()
            ],
        }

        return summary

    def get_frontend_summary(self) -> Dict[str, Any]:
        """
        Get session summary for frontend/web interface.

        This method includes legacy fields that the web interface expects
        for backward compatibility.

        Returns
        -------
        summary : dict
            Session information with frontend-specific fields
        """
        # Get current slice for legacy fields
        current = self.current_slice()

        # Collect all unique celltypes across all slices
        all_celltypes = set()
        for slice in self.slices.values():
            if slice.has_celltype():
                celltypes = slice.adata.obs['celltype'].unique()
                valid_celltypes = [ct for ct in celltypes if isinstance(ct, str)]
                all_celltypes.update(valid_celltypes)

        # Determine data format for frontend
        has_multiple_slices = self.n_slices > 1
        has_protein = 'protein' in self.modalities

        if has_multiple_slices and has_protein:
            data_format = 'multi_slice_multi_omics'
        elif has_multiple_slices:
            data_format = 'multi_slice'
        elif has_protein:
            data_format = 'multi_omics'
        else:
            data_format = 'single_slice'

        # Build available_slices list for frontend
        available_slices = []
        for slice_id in sorted(self.slices.keys()):
            slice_data = self.slices[slice_id]
            available_slices.append({
                'slice_id': slice_id,
                'tissue_name': slice_data.metadata.get('tissue_name', f'slice_{slice_id}'),
                'modality': slice_data.modality,
                'data_level': slice_data.data_level,
                'data_level_detection': slice_data.metadata.get('data_level_detection', ''),
                'image_match': slice_data.metadata.get('image_match', ''),
                'n_obs': slice_data.n_obs,
                'n_vars': slice_data.n_vars
            })

        # Ensure current_slice_id is always set
        effective_slice_id = self.current_slice_id
        if effective_slice_id is None and self.slice_ids:
            effective_slice_id = self.slice_ids[0]

        # Build summary with both new and legacy fields
        summary = {
            # New unified fields
            'name': self.name,
            'n_slices': self.n_slices,
            'slice_ids': self.slice_ids,
            'modalities': self.modalities,
            'data_levels': self.data_levels,
            'current_slice_id': effective_slice_id,
            'slices': [s.get_summary() for s in self.slices.values()],
            'n_rois': len(self.roi_manager.rois),
            'has_celltype_colors': self.celltype_colors is not None,

            # Legacy fields for frontend compatibility
            'data_format': data_format,
            'available_slices': available_slices,
            'has_protein': has_protein,
            'available_modalities': self.modalities,
        }

        # Add current slice info for legacy frontend
        if current:
            summary['n_cells'] = current.n_obs
            summary['n_genes'] = current.n_vars
            summary['current_modality'] = current.modality
            summary['celltypes'] = sorted(list(all_celltypes)) if all_celltypes else []
        else:
            # Fallback if no current slice
            summary['n_cells'] = 0
            summary['n_genes'] = 0
            summary['current_modality'] = 'gene'
            summary['celltypes'] = []

        return summary

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"SimpleSession(name='{self.name}', n_slices={self.n_slices}, "
                f"slice_ids={self.slice_ids}, current={self.current_slice_id})")

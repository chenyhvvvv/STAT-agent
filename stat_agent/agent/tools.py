"""
Agent Tools for System Introspection.

Provides functions the agent can call to inspect current state
instead of relying on static context injection.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from stat_agent.core.session import SimpleSession

logger = logging.getLogger(__name__)


class AgentTools:
    """
    Tool registry for agent to inspect system state.

    Instead of guessing or using fuzzy matching, agent can call these
    functions to get accurate, real-time information about what's available.
    """

    def __init__(self, session: Optional['SimpleSession'] = None):
        self.session = session

    def set_session(self, session: 'SimpleSession') -> None:
        """Update the session reference."""
        self.session = session

    # ========== ROI Introspection ==========

    def list_available_rois(self) -> List[str]:
        """
        Get list of ROI names currently defined in session.

        Returns:
            List of ROI names (e.g., ['ROI_1', 'ROI_2'])

        Example:
            >>> tools.list_available_rois()
            ['ROI_1', 'ROI_2', 'Region_1']
        """
        if not self.session or not self.session.has_data:
            return []
        return list(self.session.roi_subsets.keys())

    def get_roi_info(self, roi_name: str, slice_id: int = 0) -> Dict[str, Any]:
        """
        Get detailed information about a specific ROI.

        Args:
            roi_name: Name of the ROI
            slice_id: Slice ID to query (default: 0)

        Returns:
            Dict with n_cells, columns, bounds, slice_id, modality, etc.

        Raises:
            ValueError: If ROI doesn't exist

        Example:
            >>> tools.get_roi_info('ROI_1')
            {
                'roi_name': 'ROI_1',
                'slice_id': '0',
                'modality': 'gene',
                'n_cells': 1234,
                'columns': ['x', 'y', 'celltype'],
                'has_celltype': True,
                'celltypes': ['Malignant', 'T cell', ...]
            }
        """
        if not self.session or not self.session.has_data:
            raise ValueError("No data loaded in session")

        available = self.list_available_rois()
        if roi_name not in available:
            raise ValueError(
                f"ROI '{roi_name}' not found. Available ROIs: {available}"
            )

        # Get ROI object
        roi = self.session.get_roi(roi_name)
        if roi is None:
            raise ValueError(f"ROI '{roi_name}' not found")

        # Verify slice_id if specified
        if slice_id is not None and roi.slice_id != slice_id:
            raise ValueError(
                f"ROI '{roi_name}' belongs to slice {roi.slice_id}, not slice {slice_id}"
            )

        info = {
            'roi_name': roi_name,
            'slice_id': roi.slice_id,  # Which slice this ROI belongs to
            'modality': roi.modality,   # Which modality (gene/protein)
            'n_cells': roi.n_obs,
            'n_genes': roi.n_vars,
            'columns': list(roi.adata.obs.columns),
            'has_celltype': 'celltype' in roi.adata.obs.columns
        }

        # Add cell type info if available
        if info['has_celltype']:
            celltypes = roi.adata.obs['celltype'].value_counts()
            info['celltypes'] = list(celltypes.index)
            info['celltype_counts'] = {k: int(v) for k, v in celltypes.to_dict().items()}

        # Add bounds if available
        if 'x' in roi.adata.obs.columns and 'y' in roi.adata.obs.columns:
            info['bounds'] = {
                'min_x': float(roi.adata.obs['x'].min()),
                'max_x': float(roi.adata.obs['x'].max()),
                'min_y': float(roi.adata.obs['y'].min()),
                'max_y': float(roi.adata.obs['y'].max())
            }

        return info

    def roi_exists(self, roi_name: str) -> bool:
        """
        Check if a ROI exists.

        Args:
            roi_name: Name to check

        Returns:
            True if ROI exists, False otherwise
        """
        return roi_name in self.list_available_rois()

    def get_all_rois_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all ROIs with their slice and modality information.

        Returns:
            Dict mapping ROI name to {slice_id, modality, n_cells}

        Example:
            >>> tools.get_all_rois_summary()
            {
                'ROI_1': {'slice_id': '0', 'modality': 'gene', 'n_cells': 3686},
                'ROI_2': {'slice_id': '1', 'modality': 'gene', 'n_cells': 1234}
            }
        """
        if not self.session or not self.session.has_data:
            return {}

        summary = {}
        for roi_name in self.list_available_rois():
            try:
                roi = self.session.roi_manager.get_roi(roi_name)
                roi_adata = self.session.roi_subsets.get(roi_name)
                summary[roi_name] = {
                    'slice_id': roi.slice_id,
                    'modality': roi.modality,
                    'n_cells': int(roi_adata.n_obs) if roi_adata else 0
                }
            except Exception:
                continue  # Skip if ROI has issues
        return summary

    # ========== Data Introspection ==========

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data.

        Returns:
            Dict with n_cells, n_genes, celltypes, etc.
        """
        if not self.session or not self.session.has_data:
            return {'data_loaded': False}

        return self.session.get_summary()

    def list_available_genes(self, slice_id: int = 0, limit: Optional[int] = None) -> List[str]:
        """
        Get list of available gene names.

        Args:
            slice_id: Slice ID to query (default: 0)
            limit: Maximum number of genes to return (None for all)

        Returns:
            List of gene names
        """
        if not self.session or not self.session.has_data:
            return []

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return []

        genes = list(slice_obj.adata.var_names)
        if limit:
            return genes[:limit]
        return genes

    def gene_exists(self, gene_name: str, slice_id: int = 0, case_sensitive: bool = False) -> bool:
        """
        Check if a gene exists in the dataset.

        Args:
            gene_name: Gene name to check
            slice_id: Slice ID to query (default: 0)
            case_sensitive: Whether to match case-sensitively

        Returns:
            True if gene exists, False otherwise

        Example:
            >>> tools.gene_exists('CD3D')
            True
            >>> tools.gene_exists('cd3d', case_sensitive=False)
            True
        """
        if not self.session or not self.session.has_data:
            return False

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return False

        genes = slice_obj.adata.var_names
        if case_sensitive:
            return gene_name in genes
        else:
            genes_lower = [g.lower() for g in genes]
            return gene_name.lower() in genes_lower

    def find_genes_matching(self, pattern: str, slice_id: int = 0, limit: int = 10) -> List[str]:
        """
        Find genes matching a pattern (substring search).

        Args:
            pattern: Pattern to search for
            slice_id: Slice ID to query (default: 0)
            limit: Maximum number of matches to return

        Returns:
            List of matching gene names

        Example:
            >>> tools.find_genes_matching('CD3')
            ['CD3D', 'CD3E', 'CD3G']
        """
        if not self.session or not self.session.has_data:
            return []

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return []

        pattern_lower = pattern.lower()
        genes = slice_obj.adata.var_names
        matches = [g for g in genes if pattern_lower in g.lower()]
        return matches[:limit]

    def list_available_columns(self, slice_id: int = 0) -> List[str]:
        """
        List all columns in adata.obs.

        Args:
            slice_id: Slice ID to query (default: 0)

        Returns:
            List of column names
        """
        if not self.session or not self.session.has_data:
            return []

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return []

        return list(slice_obj.adata.obs.columns)

    def column_exists(self, column_name: str, slice_id: int = 0) -> bool:
        """
        Check if a column exists in adata.obs.

        Args:
            column_name: Column name to check
            slice_id: Slice ID to query (default: 0)

        Returns:
            True if column exists, False otherwise
        """
        if not self.session or not self.session.has_data:
            return False

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return False

        return column_name in slice_obj.adata.obs.columns

    def list_celltypes(self, slice_id: int = 0) -> List[str]:
        """
        Get list of unique cell types in the data.

        Args:
            slice_id: Slice ID to query (default: 0)

        Returns:
            List of cell type names, or empty list if no celltype column

        Example:
            >>> tools.list_celltypes()
            ['Malignant', 'T cell', 'B cell', 'Macrophage']
        """
        if not self.session or not self.session.has_data:
            return []

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return []

        if 'celltype' not in slice_obj.adata.obs.columns:
            return []

        return list(slice_obj.adata.obs['celltype'].unique())

    def get_celltype_counts(self, slice_id: int = 0) -> Dict[str, int]:
        """
        Get cell type counts.

        Args:
            slice_id: Slice ID to query (default: 0)

        Returns:
            Dict mapping cell type names to counts

        Example:
            >>> tools.get_celltype_counts()
            {'Malignant': 5000, 'T cell': 2000, 'B cell': 1500}
        """
        if not self.session or not self.session.has_data:
            return {}

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return {}

        if 'celltype' not in slice_obj.adata.obs.columns:
            return {}

        counts = slice_obj.adata.obs['celltype'].value_counts()
        return {k: int(v) for k, v in counts.to_dict().items()}

    # ========== Utility Methods ==========

    def get_current_working_roi(self) -> Optional[str]:
        """
        Get the most recently created ROI name (if any).

        This is useful when user says "in the current ROI" without specifying which one.

        Returns:
            ROI name or None if no ROIs exist
        """
        rois = self.list_available_rois()
        if not rois:
            return None
        # Return the last one (most recently created)
        return rois[-1]

    def validate_gene_list(self, gene_names: List[str], slice_id: int = 0) -> Dict[str, Any]:
        """
        Validate a list of gene names against available genes.

        Args:
            gene_names: List of gene names to validate
            slice_id: Slice ID to query (default: 0)

        Returns:
            Dict with 'valid', 'invalid', and 'suggestions' keys

        Example:
            >>> tools.validate_gene_list(['CD3D', 'InvalidGene', 'CD4'])
            {
                'valid': ['CD3D', 'CD4'],
                'invalid': ['InvalidGene'],
                'suggestions': {'InvalidGene': ['CD3E', 'CD3G']}
            }
        """
        if not self.session or not self.session.has_data:
            return {'valid': [], 'invalid': gene_names, 'suggestions': {}}

        valid = []
        invalid = []
        suggestions = {}

        for gene in gene_names:
            if self.gene_exists(gene, slice_id=slice_id, case_sensitive=False):
                valid.append(gene)
            else:
                invalid.append(gene)
                # Try to find similar genes
                matches = self.find_genes_matching(gene, slice_id=slice_id, limit=3)
                if matches:
                    suggestions[gene] = matches

        return {
            'valid': valid,
            'invalid': invalid,
            'suggestions': suggestions
        }

    # ========== Spot Data Introspection ==========

    def is_spot_data(self, slice_id: int = 0) -> bool:
        """
        Check if data is spot-level (e.g., Visium) rather than cell-level.

        Args:
            slice_id: Slice ID to query (default: 0)

        Returns:
            True if spot data, False if cell data

        Example:
            >>> tools.is_spot_data()
            True
        """
        if not self.session or not self.session.has_data:
            return False

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return False

        return 'spot_shape' in slice_obj.adata.uns and 'spot_diameter' in slice_obj.adata.uns

    def get_spot_properties(self, slice_id: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get spot properties if data is spot-level.

        Args:
            slice_id: Slice ID to query (default: 0)

        Returns:
            Dict with spot_shape, spot_diameter, has_deconv_weights, or None if not spot data

        Example:
            >>> tools.get_spot_properties()
            {
                'spot_shape': 'circle',
                'spot_diameter': 110,
                'has_deconv_weights': True,
                'n_celltypes_in_deconv': 17
            }
        """
        if not self.is_spot_data(slice_id):
            return None

        slice_obj = self.session.get_slice(slice_id)
        if not slice_obj:
            return None

        adata = slice_obj.adata
        props = {
            'spot_shape': adata.uns['spot_shape'],
            'spot_diameter': adata.uns['spot_diameter'],
            'has_deconv_weights': adata.uns.get('has_deconv_weights', False)
        }

        if props['has_deconv_weights'] and 'deconv_weights' in adata.obsm:
            props['n_celltypes_in_deconv'] = adata.obsm['deconv_weights'].shape[1]
            props['celltype_names'] = list(adata.obsm['deconv_weights'].columns)

        return props

    def __repr__(self) -> str:
        session_status = 'active' if self.session else 'none'
        data_status = 'loaded' if (self.session and self.session.has_data) else 'none'
        return f"AgentTools(session={session_status}, data={data_status})"


__all__ = ['AgentTools']

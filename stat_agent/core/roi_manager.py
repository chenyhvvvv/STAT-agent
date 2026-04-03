"""
Region of Interest (ROI) management for spatial data.

Handles creation, storage, and manipulation of ROIs including:
- Multiple ROI types (bounding box, circle, freehand polygon)
- ROI algebra (union, intersection, difference)
- Coordinate system awareness
- Serialization for reproducibility
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import anndata as ad
from shapely.geometry import Polygon, Point, box as Box
from shapely.ops import unary_union
from geopandas import GeoDataFrame
import geopandas as gpd

logger = logging.getLogger(__name__)


@dataclass
class ROI:
    """
    A region of interest in spatial coordinates.

    Parameters
    ----------
    name : str
        Unique name for the ROI
    geometry : shapely.geometry
        Geometric shape of the ROI
    slice_id : int, optional
        ID of the slice this ROI belongs to (for multi-slice data)
    modality : str, optional
        Modality this ROI belongs to ('gene' or 'protein', for multi-omics data)
    adata : AnnData, optional
        Filtered data for this ROI (cells/spots within ROI geometry)
    coordinate_system : str
        Coordinate system the ROI is defined in
    metadata : dict, optional
        Additional metadata
    created_at : str
        ISO format timestamp of creation
    """

    name: str
    geometry: Union[Polygon, Point, Box]
    slice_id: Optional[int] = None
    modality: Optional[str] = None
    adata: Optional[ad.AnnData] = None
    coordinate_system: str = "global"
    metadata: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ========================================
    # Data access properties (like DataSlice)
    # ========================================

    @property
    def n_obs(self) -> int:
        """Number of observations (cells or spots) in this ROI."""
        return int(self.adata.n_obs) if self.adata is not None else 0

    @property
    def n_vars(self) -> int:
        """Number of variables (genes or proteins) in this ROI."""
        return int(self.adata.n_vars) if self.adata is not None else 0

    @property
    def area(self) -> float:
        """Get the area of the ROI."""
        return self.geometry.area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (minx, miny, maxx, maxy)."""
        return self.geometry.bounds

    @property
    def centroid(self) -> Tuple[float, float]:
        """Get the centroid coordinates."""
        c = self.geometry.centroid
        return (c.x, c.y)

    @property
    def type(self) -> str:
        """Get the ROI type (from metadata or geometry type)."""
        return self.metadata.get('type', self.geometry.geom_type.lower())

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the ROI."""
        return self.geometry.contains(Point(x, y))

    def to_dict(self) -> Dict:
        """Serialize ROI to dictionary."""
        return {
            "name": self.name,
            "geometry": self.geometry.__geo_interface__,
            "slice_id": self.slice_id,
            "modality": self.modality,
            "coordinate_system": self.coordinate_system,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> ROI:
        """Deserialize ROI from dictionary."""
        from shapely.geometry import shape
        return cls(
            name=data["name"],
            geometry=shape(data["geometry"]),
            slice_id=data.get("slice_id"),
            modality=data.get("modality"),
            coordinate_system=data["coordinate_system"],
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
        )

    def __repr__(self) -> str:
        slice_info = f", slice_id='{self.slice_id}'" if self.slice_id else ""
        modality_info = f", modality='{self.modality}'" if self.modality else ""
        return (
            f"ROI(name='{self.name}', "
            f"type={self.geometry.geom_type}, "
            f"area={self.area:.2f}"
            f"{slice_info}{modality_info}, "
            f"coordinate_system='{self.coordinate_system}')"
        )


class ROIManager:
    """
    Manager for spatial regions of interest.

    Handles creation, storage, and manipulation of ROIs with support for:
    - Named ROI storage
    - ROI algebra (union, intersection, difference)
    - Coordinate system tracking
    - GeoDataFrame export

    Attributes
    ----------
    rois : Dict[str, ROI]
        Dictionary of named ROIs
    """

    def __init__(self):
        self.rois: Dict[str, ROI] = {}
        logger.debug("Initialized ROIManager")

    def add_roi(
        self,
        name: str,
        geometry: Union[Polygon, Point, Box, List[Tuple[float, float]]],
        slice_id: Optional[int] = None,
        modality: Optional[str] = None,
        coordinate_system: str = "global",
        metadata: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> ROI:
        """
        Add a new ROI.

        Parameters
        ----------
        name : str
            Unique name for the ROI
        geometry : shapely geometry or list of coordinates
            ROI geometry. If list, creates a Polygon from coordinates
        slice_id : int, optional
            ID of the slice this ROI belongs to (for multi-slice data)
        modality : str, optional
            Modality this ROI belongs to ('gene' or 'protein')
        coordinate_system : str
            Coordinate system name
        metadata : dict, optional
            Additional metadata
        overwrite : bool
            Whether to overwrite existing ROI with same name

        Returns
        -------
        ROI
            The created ROI object

        Raises
        ------
        ValueError
            If ROI name already exists and overwrite=False
        """
        if name in self.rois and not overwrite:
            raise ValueError(
                f"ROI '{name}' already exists. Use overwrite=True to replace."
            )

        # Convert list of coordinates to Polygon
        if isinstance(geometry, list):
            geometry = Polygon(geometry)

        roi = ROI(
            name=name,
            geometry=geometry,
            slice_id=slice_id,
            modality=modality,
            coordinate_system=coordinate_system,
            metadata=metadata or {},
        )

        self.rois[name] = roi
        logger.info(f"Added ROI: {roi}")
        return roi

    def add_circle_roi(
        self,
        name: str,
        center: Tuple[float, float],
        radius: float,
        coordinate_system: str = "global",
        n_points: int = 64,
        **kwargs
    ) -> ROI:
        """
        Create circular ROI.

        Parameters
        ----------
        name : str
            ROI name
        center : tuple
            (x, y) center coordinates
        radius : float
            Circle radius
        coordinate_system : str
            Coordinate system
        n_points : int
            Number of points to approximate circle
        """
        # Create circle as polygon approximation
        angles = np.linspace(0, 2 * np.pi, n_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        coords = list(zip(x, y))

        return self.add_roi(
            name=name,
            geometry=Polygon(coords),
            coordinate_system=coordinate_system,
            metadata={"type": "circle", "center": center, "radius": radius},
            **kwargs
        )

    def add_bbox_roi(
        self,
        name: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        coordinate_system: str = "global",
        **kwargs
    ) -> ROI:
        """Create ROI from bounding box."""
        return self.add_roi(
            name=name,
            geometry=Box(min_x, min_y, max_x, max_y),
            coordinate_system=coordinate_system,
            metadata={"type": "bbox"},
            **kwargs
        )

    def get_roi(self, name: str) -> ROI:
        """
        Get ROI by name.

        Parameters
        ----------
        name : str
            ROI name

        Returns
        -------
        ROI
            The requested ROI

        Raises
        ------
        KeyError
            If ROI name not found
        """
        if name not in self.rois:
            raise KeyError(f"ROI '{name}' not found. Available: {list(self.rois.keys())}")
        return self.rois[name]

    def remove_roi(self, name: str) -> None:
        """Remove an ROI by name."""
        if name in self.rois:
            del self.rois[name]
            logger.info(f"Removed ROI: {name}")
        else:
            logger.warning(f"ROI '{name}' not found")

    def get_rois_for_slice(
        self,
        slice_id: Optional[str] = None,
        modality: Optional[str] = None
    ) -> Dict[str, ROI]:
        """
        Get all ROIs belonging to a specific slice and/or modality.

        Parameters
        ----------
        slice_id : str, optional
            Slice ID to filter by. If None, returns ROIs for all slices.
        modality : str, optional
            Modality to filter by ('gene' or 'protein'). If None, returns ROIs for all modalities.

        Returns
        -------
        Dict[str, ROI]
            Dictionary of ROI name -> ROI object matching the criteria
        """
        filtered_rois = {}
        for name, roi in self.rois.items():
            # Check slice_id match (None means match all)
            slice_match = (slice_id is None) or (roi.slice_id == slice_id)
            # Check modality match (None means match all)
            modality_match = (modality is None) or (roi.modality == modality)

            if slice_match and modality_match:
                filtered_rois[name] = roi

        return filtered_rois

    def union(self, name1: str, name2: str, new_name: str) -> ROI:
        """
        Create union of two ROIs.

        Parameters
        ----------
        name1, name2 : str
            Names of ROIs to union
        new_name : str
            Name for the resulting ROI

        Returns
        -------
        ROI
            Union ROI
        """
        roi1 = self.get_roi(name1)
        roi2 = self.get_roi(name2)

        if roi1.coordinate_system != roi2.coordinate_system:
            raise ValueError(
                f"ROIs must be in same coordinate system. "
                f"Got {roi1.coordinate_system} and {roi2.coordinate_system}"
            )

        union_geom = unary_union([roi1.geometry, roi2.geometry])
        return self.add_roi(
            name=new_name,
            geometry=union_geom,
            coordinate_system=roi1.coordinate_system,
            metadata={"operation": "union", "sources": [name1, name2]},
        )

    def intersection(self, name1: str, name2: str, new_name: str) -> ROI:
        """Create intersection of two ROIs."""
        roi1 = self.get_roi(name1)
        roi2 = self.get_roi(name2)

        if roi1.coordinate_system != roi2.coordinate_system:
            raise ValueError("ROIs must be in same coordinate system")

        intersect_geom = roi1.geometry.intersection(roi2.geometry)
        return self.add_roi(
            name=new_name,
            geometry=intersect_geom,
            coordinate_system=roi1.coordinate_system,
            metadata={"operation": "intersection", "sources": [name1, name2]},
        )

    def difference(self, name1: str, name2: str, new_name: str) -> ROI:
        """Create difference of two ROIs (roi1 - roi2)."""
        roi1 = self.get_roi(name1)
        roi2 = self.get_roi(name2)

        if roi1.coordinate_system != roi2.coordinate_system:
            raise ValueError("ROIs must be in same coordinate system")

        diff_geom = roi1.geometry.difference(roi2.geometry)
        return self.add_roi(
            name=new_name,
            geometry=diff_geom,
            coordinate_system=roi1.coordinate_system,
            metadata={"operation": "difference", "sources": [name1, name2]},
        )

    def to_geodataframe(self, coordinate_system: Optional[str] = None) -> GeoDataFrame:
        """
        Export all ROIs to a GeoDataFrame.

        Parameters
        ----------
        coordinate_system : str, optional
            If specified, only export ROIs in this coordinate system

        Returns
        -------
        GeoDataFrame
            GeoDataFrame with all ROIs
        """
        rois_to_export = self.rois.values()
        if coordinate_system:
            rois_to_export = [r for r in rois_to_export if r.coordinate_system == coordinate_system]

        if not rois_to_export:
            return gpd.GeoDataFrame()

        data = {
            "name": [r.name for r in rois_to_export],
            "geometry": [r.geometry for r in rois_to_export],
            "coordinate_system": [r.coordinate_system for r in rois_to_export],
            "area": [r.area for r in rois_to_export],
            "created_at": [r.created_at for r in rois_to_export],
        }

        return gpd.GeoDataFrame(data, geometry="geometry")

    def clear(self) -> None:
        """Remove all ROIs."""
        self.rois.clear()
        logger.info("Cleared all ROIs")

    def __len__(self) -> int:
        return len(self.rois)

    def __repr__(self) -> str:
        return f"ROIManager(n_rois={len(self.rois)}, names={list(self.rois.keys())})"

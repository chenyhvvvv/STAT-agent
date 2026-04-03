"""Core components for simplified spatial data session management."""

from stat_agent.core.data_slice import DataSlice
from stat_agent.core.session import SimpleSession
from stat_agent.core.roi_manager import ROIManager, ROI

__all__ = [
    "DataSlice",
    "SimpleSession",
    "ROIManager",
    "ROI",
]

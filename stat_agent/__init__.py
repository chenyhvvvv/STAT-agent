"""
STAT

Spatial Transcriptomics Analytical agenT

Your AI Laboratory for Spatial Transcriptomics Analysis.
An AI-powered platform for spatial omics analysis with multi-format support,
interactive visualization, and intelligent code generation.
"""

__version__ = "0.1.1"

from stat_agent.core.session import SimpleSession
from stat_agent.core.roi_manager import ROIManager, ROI

# Import and expose IO functions
from stat_agent.functions.io import (
    load_anndata,
    load_image,
    load_data,
)

__all__ = [
    "SimpleSession",
    "ROIManager",
    "ROI",
    "load_anndata",
    "load_image",
    "load_data",
]


def create_session(name="session"):
    """
    Create a new simplified spatial analysis session.

    Parameters
    ----------
    name : str
        Session name

    Returns
    -------
    SimpleSession
        Initialized session object
    """
    return SimpleSession(name=name)

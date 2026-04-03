"""
__init__.py for celltype_annotation scripts.

Exports the main annotation function for easy import.
"""

from .annotation_scvi import annotate_celltype_scvi, validate_annotation_inputs

__all__ = ['annotate_celltype_scvi', 'validate_annotation_inputs']

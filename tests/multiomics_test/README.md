# Multi-Omics Test Suite

**Status**: Placeholder for Phase 2

## Purpose

This directory will contain integration tests for multi-omics spatial data analysis, including:

- Spatial transcriptomics + protein (e.g., CODEX, MIBI)
- Spatial transcriptomics + metabolomics
- Combined analysis workflows

## Planned Features

1. **Data Loading**
   - Multiple modalities in single session
   - Modality alignment and registration
   - Cross-modality queries

2. **Analysis**
   - Joint dimensionality reduction
   - Cross-modality cell type annotation
   - Modality-specific and joint spatial analysis

3. **Visualization**
   - Multi-modal spatial plots
   - Modality overlay visualization
   - Cross-modality correlation heatmaps

## Implementation

Will be implemented after:
- Single slice architecture is stable
- Multiple slice support is complete
- Data format standardization is finalized

## Test Data Requirements

- Xenium (RNA) + CODEX (protein) from same tissue
- Coordinated spatial coordinates
- Matching cell segmentation across modalities

# Single Slice Test Suite

**Status**: Placeholder for comprehensive integration tests

## Purpose

This directory will contain comprehensive integration tests for the current single-slice architecture:

- End-to-end workflow tests
- Real-world analysis scenarios
- Performance benchmarks

## Planned Test Scenarios

1. **Basic Workflows**
   - Load data → Create ROI → Analyze cell types → Export results
   - Load data → Visualize gene expression → Find markers
   - Load data → Niche detection → Interpret results

2. **Advanced Analysis**
   - Multi-ROI comparison
   - Differential expression between ROIs
   - Spatial correlation analysis
   - Cell-cell interaction analysis

3. **LLM Agent Tests**
   - Natural language query understanding
   - Code generation quality
   - Error recovery and reflection
   - Result interpretation accuracy

4. **Performance Tests**
   - Large dataset handling (1M+ cells)
   - Memory usage profiling
   - Response time benchmarks

## Test Data

### Small Dataset (Quick Tests)
- ~50K cells
- 500 genes
- H&E image 2000x2000

### Medium Dataset (Integration Tests)
- ~500K cells
- 2000 genes
- H&E image 10000x10000

### Large Dataset (Performance Tests)
- ~2M cells
- 5000 genes
- H&E image 30000x30000

## Implementation

Currently implemented basic tests:
- `test_basic.py` - Basic sanity checks
- `test_llm_skill_matching.py` - Skill matching
- `test_niche_detection_integration.py` - Niche analysis
- `test_xenium_integration.py` - Real data tests

**TODO**: Add comprehensive end-to-end workflow tests

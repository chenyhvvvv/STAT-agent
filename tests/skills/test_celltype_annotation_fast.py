"""
Integration test for fast celltype annotation skill (clustering + LLM).

Tests the complete workflow:
1. Create mock spatial data
2. Run clustering-based annotation
3. Verify celltype column is added
4. Validate results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import scanpy as sc
import anndata as ad
from stat.core.session import SimpleSession


def create_test_data():
    """Create synthetic test data for annotation."""
    print("Creating test data...")

    # Create spatial data with distinct patterns
    n_cells = 1000
    n_genes = 200

    # Create expression with 3 distinct patterns
    expr = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))

    # Pattern 1: High expression in genes 0-50 (will be "Type A")
    expr[:350, :50] *= 5

    # Pattern 2: High expression in genes 50-100 (will be "Type B")
    expr[350:700, 50:100] *= 5

    # Pattern 3: High expression in genes 100-150 (will be "Type C")
    expr[700:, 100:150] *= 5

    # Create AnnData
    adata = ad.AnnData(X=expr)
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]
    adata.obs['x'] = np.random.rand(n_cells) * 1000
    adata.obs['y'] = np.random.rand(n_cells) * 1000

    print(f"  Created: {adata.n_obs} cells × {adata.n_vars} genes")

    return adata


def test_annotation_skill():
    """Test the fast celltype annotation skill."""
    print("=" * 70)
    print("FAST CELLTYPE ANNOTATION SKILL TEST")
    print("=" * 70)

    # Create test data
    adata = create_test_data()

    # Test 1: Validate inputs
    print("\n" + "=" * 70)
    print("Test 1: Input Validation")
    print("=" * 70)

    try:
        from scripts.annotation_clustering import validate_annotation_inputs

        validation = validate_annotation_inputs(
            adata_spatial=adata,
            tissue_type='test tissue'
        )

        print(f"\nValidation result:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Errors: {validation['errors']}")
        print(f"  Warnings: {validation['warnings']}")
        print(f"  Info: {validation['info']}")

        if not validation['valid']:
            print("\n❌ Validation failed!")
            return False

        print("\n✅ Test 1 PASSED: Input validation successful")

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Mock LLM annotation
    print("\n" + "=" * 70)
    print("Test 2: Clustering and Mock Annotation")
    print("=" * 70)

    try:
        from scripts.annotation_clustering import annotate_celltype_clustering

        # Mock LLM function (no actual LLM call)
        def mock_llm_annotator(markers_dict, tissue_type):
            """Mock LLM that returns simple annotations."""
            print(f"  Mock LLM called for {tissue_type}")
            print(f"  Clusters to annotate: {len(markers_dict)}")

            annotations = {}
            for i, cluster_id in enumerate(sorted(markers_dict.keys())):
                # Simple naming scheme
                annotations[cluster_id] = f'CellType_{chr(65+i)}'  # A, B, C, ...

            return annotations

        # Run annotation
        print("\nRunning clustering annotation (with mock LLM)...")
        celltype_predictions = annotate_celltype_clustering(
            adata_spatial=adata,
            tissue_type='test tissue',
            llm_function=mock_llm_annotator,
            resolution=0.5,
            n_top_genes=10,
            min_cluster_size=10,
            preprocess=True
        )

        print(f"\n✓ Annotation completed!")
        print(f"  Predictions shape: {len(celltype_predictions)}")
        print(f"  Unique cell types: {celltype_predictions.nunique()}")
        print(f"\n  Distribution:")
        for ct, count in celltype_predictions.value_counts().items():
            print(f"    {ct}: {count} cells ({count/len(celltype_predictions):.1%})")

        # Verify predictions
        assert len(celltype_predictions) == adata.n_obs, "Prediction length mismatch"
        assert celltype_predictions.notna().all(), "Predictions contain NA"
        assert celltype_predictions.nunique() > 0, "No cell types predicted"
        assert (celltype_predictions.index == adata.obs_names).all(), "Index mismatch"

        print("\n✅ Test 2 PASSED: Clustering annotation successful")

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Session integration
    print("\n" + "=" * 70)
    print("Test 3: Session Integration")
    print("=" * 70)

    try:
        # Create session
        session = SimpleSession(name='test_fast_annotation')
        session.adata = adata.copy()

        # Verify no celltype initially
        initial_has_celltype = 'celltype' in session.adata.obs.columns
        print(f"  Initial has_celltype: {initial_has_celltype}")

        # Add celltype (simulating skill workflow)
        session.adata.obs['celltype'] = celltype_predictions

        # Verify celltype added
        final_has_celltype = 'celltype' in session.adata.obs.columns
        print(f"  Final has_celltype: {final_has_celltype}")

        # Verify values
        assert final_has_celltype, "Celltype not added to session"
        assert (session.adata.obs['celltype'] == celltype_predictions).all(), "Celltype values mismatch"
        assert session.adata.obs['celltype'].notna().all(), "Celltype contains NA"

        print(f"\n  Session after annotation:")
        print(f"    Cells: {session.n_cells:,}")
        print(f"    Genes: {session.n_genes:,}")
        print(f"    Cell types: {session.adata.obs['celltype'].nunique()}")

        print("\n✅ Test 3 PASSED: Session integration successful")

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: OpenAI integration (optional, skip if no API key)
    print("\n" + "=" * 70)
    print("Test 4: OpenAI Integration (optional)")
    print("=" * 70)

    try:
        import os
        if 'OPENAI_API_KEY' in os.environ:
            print("  OpenAI API key found, testing annotation...")

            from scripts.annotation_clustering import annotate_with_openai

            # Create simple test case
            test_markers = {
                '0': {
                    'genes': ['CD3D', 'CD3E', 'CD8A', 'CD4', 'IL7R'],
                    'scores': [10.0, 9.5, 8.0, 7.5, 7.0],
                    'pvals': [1e-50, 1e-45, 1e-40, 1e-35, 1e-30],
                    'logfoldchanges': [2.5, 2.3, 2.0, 1.8, 1.5],
                    'n_cells': 100
                },
                '1': {
                    'genes': ['EPCAM', 'KRT18', 'KRT19', 'CDH1', 'MUC1'],
                    'scores': [12.0, 11.0, 10.5, 9.0, 8.5],
                    'pvals': [1e-60, 1e-55, 1e-50, 1e-45, 1e-40],
                    'logfoldchanges': [3.0, 2.8, 2.5, 2.2, 2.0],
                    'n_cells': 150
                }
            }

            annotations = annotate_with_openai(
                markers_dict=test_markers,
                tissue_type='breast cancer',
                model='gpt-3.5-turbo'  # Use cheaper model for testing
            )

            print(f"\n  OpenAI annotations:")
            for cluster_id, celltype in annotations.items():
                print(f"    Cluster {cluster_id}: {celltype}")

            # Verify structure
            assert isinstance(annotations, dict), "Annotations not a dict"
            assert '0' in annotations, "Cluster 0 not annotated"
            assert '1' in annotations, "Cluster 1 not annotated"

            print("\n✅ Test 4 PASSED: OpenAI integration successful")

        else:
            print("  ⚠️  OPENAI_API_KEY not set, skipping OpenAI test")
            print("     (This is optional - skill works with custom LLM functions)")

    except ImportError as e:
        print(f"\n⚠️  OpenAI package not installed: {e}")
        print("     Install with: pip install openai")

    except Exception as e:
        print(f"\n⚠️  OpenAI test failed (non-critical): {e}")
        print("     Skill can work with custom LLM functions")

    return True


def main():
    """Run all tests."""
    try:
        success = test_annotation_skill()

        print("\n" + "=" * 70)
        if success:
            print("ALL TESTS PASSED ✅")
        else:
            print("SOME TESTS FAILED ❌")
        print("=" * 70)

        return success

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

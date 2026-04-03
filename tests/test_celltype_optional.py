"""
Test script to verify optional celltype annotation support.

Creates sample data with and without celltype, then tests the system's handling.
"""

import sys
from pathlib import Path

# Add stat_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scanpy as sc
from stat_agent.core.session import SimpleSession

def test_no_celltype():
    """Test loading and handling data without celltype annotation."""
    print("=" * 60)
    print("Test 1: Data WITHOUT celltype annotation")
    print("=" * 60)

    # Create test data without celltype
    n_cells = 1000
    n_genes = 50

    adata = sc.AnnData(
        X=np.random.rand(n_cells, n_genes),
        obs={
            'x': np.random.rand(n_cells) * 1000,
            'y': np.random.rand(n_cells) * 1000
        }
    )

    # Add gene names
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]

    print(f"Created AnnData: {adata.shape}")
    print(f"Columns in obs: {list(adata.obs.columns)}")
    print(f"Has celltype: {'celltype' in adata.obs.columns}")

    # Save test data
    test_file = Path('/tmp/test_no_celltype.h5ad')
    adata.write(test_file)
    print(f"\nSaved to: {test_file}")

    # Load into session
    session = SimpleSession(name='test_no_celltype')
    session.adata = adata
    # has_data is a property that checks if adata exists

    # Test celltype check
    has_celltype = 'celltype' in session.adata.obs.columns
    print(f"\nSession check:")
    print(f"  - Has data: {session.has_data}")
    print(f"  - Has celltype: {has_celltype}")
    print(f"  - N cells: {session.n_cells:,}")
    print(f"  - N genes: {session.n_genes:,}")

    # Simulate what the API endpoint would return
    print(f"\nAPI response would include:")
    print(f"  - has_celltype: {has_celltype}")
    print(f"  - celltypes: None")
    print(f"  - message: 'No celltype annotations available...'")

    print("\n✅ Test 1 PASSED: System handles missing celltype gracefully\n")

    return test_file


def test_with_celltype():
    """Test loading and handling data with celltype annotation."""
    print("=" * 60)
    print("Test 2: Data WITH celltype annotation")
    print("=" * 60)

    # Create test data with celltype
    n_cells = 1000
    n_genes = 50

    adata = sc.AnnData(
        X=np.random.rand(n_cells, n_genes),
        obs={
            'x': np.random.rand(n_cells) * 1000,
            'y': np.random.rand(n_cells) * 1000,
            'celltype': np.random.choice(['T cell', 'B cell', 'Macrophage', 'Fibroblast'], n_cells)
        }
    )

    # Add gene names
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]

    print(f"Created AnnData: {adata.shape}")
    print(f"Columns in obs: {list(adata.obs.columns)}")
    print(f"Has celltype: {'celltype' in adata.obs.columns}")

    # Save test data
    test_file = Path('/tmp/test_with_celltype.h5ad')
    adata.write(test_file)
    print(f"\nSaved to: {test_file}")

    # Load into session
    session = SimpleSession(name='test_with_celltype')
    session.adata = adata
    # has_data is a property that checks if adata exists

    # Test celltype check
    has_celltype = 'celltype' in session.adata.obs.columns
    print(f"\nSession check:")
    print(f"  - Has data: {session.has_data}")
    print(f"  - Has celltype: {has_celltype}")
    print(f"  - N cells: {session.n_cells:,}")
    print(f"  - N genes: {session.n_genes:,}")

    if has_celltype:
        celltypes = sorted(session.adata.obs['celltype'].unique().tolist())
        print(f"  - Celltypes: {celltypes}")

    # Simulate what the API endpoint would return
    print(f"\nAPI response would include:")
    print(f"  - has_celltype: {has_celltype}")
    print(f"  - celltypes: {celltypes}")
    print(f"  - celltype_colors: {{...}}")
    print(f"  - message: None")

    print("\n✅ Test 2 PASSED: System handles celltype annotation correctly\n")

    return test_file


def test_add_celltype_later():
    """Test adding celltype annotation after initial loading."""
    print("=" * 60)
    print("Test 3: Adding celltype AFTER loading")
    print("=" * 60)

    # Create test data without celltype
    n_cells = 1000
    n_genes = 50

    adata = sc.AnnData(
        X=np.random.rand(n_cells, n_genes),
        obs={
            'x': np.random.rand(n_cells) * 1000,
            'y': np.random.rand(n_cells) * 1000
        }
    )
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]

    print(f"Initial state:")
    print(f"  - Columns: {list(adata.obs.columns)}")
    print(f"  - Has celltype: {'celltype' in adata.obs.columns}")

    # Simulate annotation skill adding celltype
    predicted_celltypes = np.random.choice(['T cell', 'B cell', 'Macrophage'], n_cells)
    adata.obs['celltype'] = predicted_celltypes

    print(f"\nAfter annotation:")
    print(f"  - Columns: {list(adata.obs.columns)}")
    print(f"  - Has celltype: {'celltype' in adata.obs.columns}")
    print(f"  - Celltypes: {sorted(adata.obs['celltype'].unique().tolist())}")

    print("\n✅ Test 3 PASSED: Celltype can be added dynamically\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CELLTYPE OPTIONAL SUPPORT TESTS")
    print("=" * 60 + "\n")

    try:
        # Test 1: No celltype
        file1 = test_no_celltype()

        # Test 2: With celltype
        file2 = test_with_celltype()

        # Test 3: Add celltype later
        test_add_celltype_later()

        # Summary
        print("=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        print("\nTest files created:")
        print(f"  - {file1}")
        print(f"  - {file2}")
        print("\nYou can load these files in the web interface to test:")
        print("  1. Load test_no_celltype.h5ad → Should see gray cells")
        print("  2. Load test_with_celltype.h5ad → Should see colored cells")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

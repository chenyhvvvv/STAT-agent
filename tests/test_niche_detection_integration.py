"""Integration test: niche detection with real data."""

import pytest
from pathlib import Path
from stat_agent.agent.spatial_agent_core import SpatialAgent
from stat_agent.core.session import SimpleSession
import scanpy as sc


@pytest.mark.asyncio
async def test_niche_detection_skill_matching():
    """Test that 'Perform niche detection' triggers skill matching."""
    # Setup session with real data
    adata_path = "/import/home3/yhchenmath/Dataset/CellARTPaper/figure_4/adata_breast_cancer_rep1_x_y.h5ad"

    # Check if data file exists
    if not Path(adata_path).exists():
        pytest.skip(f"Data file not found: {adata_path}")

    session = SimpleSession()
    session.adata = sc.read_h5ad(adata_path)

    # Setup agent (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        session=session,
        enable_skills=True
    )

    # Test skill matching
    request = "Perform niche detection on this data"
    matched_slugs = await agent._select_skill_matches_llm(request, top_k=2)

    # Verify
    assert "niche-detection" in matched_slugs, f"Expected 'niche-detection' in {matched_slugs}"
    print(f"✅ Skill matching successful: {matched_slugs}")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_niche_detection_full_workflow():
    """Test that 'Perform niche detection' generates appropriate code (without execution)."""
    # Setup session with real data
    adata_path = "/import/home3/yhchenmath/Dataset/CellARTPaper/figure_4/adata_breast_cancer_rep1_x_y.h5ad"

    # Check if data file exists
    if not Path(adata_path).exists():
        pytest.skip(f"Data file not found: {adata_path}")

    session = SimpleSession()
    session.adata = sc.read_h5ad(adata_path)

    # Setup agent (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        session=session,
        enable_skills=True
    )

    # Test full workflow (without execution to avoid long runtime)
    request = "Perform niche detection on this data"
    response = await agent.chat(request, execute_code=False)

    # Verify response quality
    # 1. Response should mention niches
    assert "niche" in response.lower(), "Response should mention niches"

    # 2. Response should contain code block
    assert "```python" in response, "Response should contain code block"

    # 3. Check for specific niche detection concepts
    niche_keywords = ["harmonics", "niche_label", "neighborhood", "microenvironment", "spatial"]
    has_niche_keyword = any(keyword in response.lower() for keyword in niche_keywords)
    assert has_niche_keyword, f"Response should contain niche-related keywords: {niche_keywords}"

    print(f"✅ Niche detection workflow successful!")
    print(f"Response preview (first 500 chars):\n{response[:500]}...")


@pytest.mark.asyncio
async def test_skill_guidance_injection():
    """Test that skill guidance is properly injected into prompts."""
    # Setup session with minimal data
    import numpy as np
    import pandas as pd
    import anndata

    # Create minimal test data
    n_cells = 100
    n_genes = 50
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame({
        'x': np.random.rand(n_cells) * 1000,
        'y': np.random.rand(n_cells) * 1000,
        'celltype': np.random.choice(['A', 'B', 'C'], n_cells)
    })
    adata = anndata.AnnData(X=X, obs=obs)

    session = SimpleSession()
    session.adata = adata

    # Setup agent
    skill_dir = Path(__file__).parent.parent / "skills"
    agent = SpatialAgent(
        model="gpt-4o-mini",
        session=session,
        skill_dir=skill_dir,
        enable_skills=True
    )

    # Manually test skill guidance injection
    request = "Perform niche detection"
    matched_slugs = await agent._select_skill_matches_llm(request, top_k=1)

    if matched_slugs:
        # Load full skill
        skill_definitions = []
        for slug in matched_slugs:
            full_skill = agent.skill_registry.load_full_skill(slug)
            if full_skill:
                skill_definitions.append(full_skill)

        # Format skill guidance
        skill_guidance = agent._format_skill_guidance(skill_definitions)

        # Verify guidance has content
        assert skill_guidance, "Skill guidance should not be empty"
        assert len(skill_guidance) > 100, "Skill guidance should have substantial content"

        print(f"✅ Skill guidance injection working")
        print(f"Guidance length: {len(skill_guidance)} characters")
        print(f"Guidance preview (first 200 chars):\n{skill_guidance[:200]}...")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])

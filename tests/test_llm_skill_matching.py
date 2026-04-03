"""Test LLM-based skill matching."""

import pytest
from pathlib import Path
from stat.agent.spatial_agent_core import SpatialAgent


@pytest.mark.asyncio
async def test_select_skill_matches_llm_niche_detection():
    """Test that 'Perform niche detection' matches niche-detection skill."""
    # Setup (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        enable_skills=True
    )

    # Test
    request = "Perform niche detection on this data"
    matched_slugs = await agent._select_skill_matches_llm(request, top_k=2)

    # Verify
    assert "niche-detection" in matched_slugs, f"Expected 'niche-detection' in {matched_slugs}"
    print(f"✅ Matched skills: {matched_slugs}")


@pytest.mark.asyncio
async def test_select_skill_matches_llm_roi_analysis():
    """Test that 'Define an ROI' matches roi-analysis skill."""
    # Setup (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        enable_skills=True
    )

    # Test
    request = "Define an ROI for tumor region"
    matched_slugs = await agent._select_skill_matches_llm(request, top_k=2)

    # Verify
    assert "roi-analysis" in matched_slugs, f"Expected 'roi-analysis' in {matched_slugs}"
    print(f"✅ Matched skills: {matched_slugs}")


@pytest.mark.asyncio
async def test_select_skill_matches_llm_no_match():
    """Test that unrelated requests don't match skills."""
    # Setup (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        enable_skills=True
    )

    # Test
    request = "What is the weather today?"
    matched_slugs = await agent._select_skill_matches_llm(request, top_k=2)

    # Verify
    assert len(matched_slugs) == 0, f"Expected no matches for weather query, got {matched_slugs}"
    print(f"✅ Correctly returned no matches: {matched_slugs}")


@pytest.mark.asyncio
async def test_progressive_disclosure():
    """Test that progressive disclosure loads metadata first."""
    # Setup (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        enable_skills=True
    )

    # Verify metadata is loaded
    assert agent.skill_registry is not None, "Skill registry should be initialized"
    assert len(agent.skill_registry.skill_metadata) > 0, "Should have loaded skill metadata"

    # Verify full content is NOT loaded yet
    assert len(agent.skill_registry._full_skills_cache) == 0, "Should not have loaded full content at startup"

    print(f"✅ Progressive disclosure working: {len(agent.skill_registry.skill_metadata)} skills metadata loaded")


@pytest.mark.asyncio
async def test_lazy_loading_on_match():
    """Test that full skill content is loaded on-demand when matched."""
    # Setup (skill_dir will auto-discover .claude/skills/)
    agent = SpatialAgent(
        model="gpt-4o-mini",
        enable_skills=True
    )

    # Initially no full skills loaded
    assert len(agent.skill_registry._full_skills_cache) == 0, "Should start with no full skills loaded"

    # Match a skill
    request = "Perform niche detection"
    matched_slugs = await agent._select_skill_matches_llm(request, top_k=2)

    # Load full skill content
    for slug in matched_slugs:
        full_skill = agent.skill_registry.load_full_skill(slug)
        assert full_skill is not None, f"Should load full skill for {slug}"
        assert full_skill.body, f"Full skill should have body content"

    # Verify full skill is now cached
    assert len(agent.skill_registry._full_skills_cache) > 0, "Should have cached full skills"

    print(f"✅ Lazy loading working: {len(agent.skill_registry._full_skills_cache)} full skills loaded on-demand")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

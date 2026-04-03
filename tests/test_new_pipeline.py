"""Comprehensive tests for the new robust skill pipeline.

Tests the 4-stage pipeline:
1. QueryPlanner - Query clarification and slice inference
2. SkillFilter - Programmatic filtering by format/modality/data_level
3. Semantic Matching - LLM-based skill selection
4. SkillVerifier - Prerequisites checking and collection
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from stat_agent.agent.spatial_agent_core import SpatialAgent
from stat_agent.agent.query_planner import QueryPlanner, PlanResult, PlanStep
from stat_agent.agent.skill_filter import SkillFilter
from stat_agent.agent.skill_verifier import SkillVerifier, VerificationResult
from stat_agent.agent.skill_registry import SkillMetadata
from stat_agent.core.session_simple import SimpleSession


# ============================================================================
# QueryPlanner Tests
# ============================================================================

@pytest.mark.asyncio
async def test_query_planner_single_slice_no_clarification():
    """Test planner with unambiguous single-slice query."""
    # Create mock LLM backend
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value='''```json
{
  "needs_clarification": false,
  "steps": [
    {
      "step_number": 1,
      "description": "Annotate cell types in slice 0",
      "target_slice_ids": [0],
      "refined_query": "Annotate cell types in slice 0 using reference-based annotation"
    }
  ]
}
```''')

    planner = QueryPlanner(llm_backend=mock_llm)

    # Mock session summary
    session_summary = {
        'n_slices': 1,
        'slices': [
            {
                'slice_id': 0,
                'tissue_name': 'breast_cancer',
                'modality': 'gene',
                'data_level': 'cell',
                'n_obs': 50000
            }
        ]
    }

    # Test
    result = await planner.plan(
        user_query="Annotate cell types",
        session_summary=session_summary,
        previous_clarifications=[]
    )

    # Verify
    assert not result.needs_clarification
    assert len(result.steps) == 1
    assert result.steps[0].target_slice_ids == [0]
    print(f"✅ Planner correctly handled single-slice query without clarification")


@pytest.mark.asyncio
async def test_query_planner_multi_slice_needs_clarification():
    """Test planner asks for clarification with ambiguous multi-slice query."""
    # Create mock LLM backend
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value='''```json
{
  "needs_clarification": true,
  "clarification_question": "Which slice would you like to annotate? You have 2 slices: slice 0 (breast_cancer_rep1, gene, cell-level) and slice 1 (breast_cancer_rep2, gene, cell-level)."
}
```''')

    planner = QueryPlanner(llm_backend=mock_llm)

    # Mock session summary with multiple slices
    session_summary = {
        'n_slices': 2,
        'slices': [
            {'slice_id': 0, 'tissue_name': 'breast_cancer_rep1', 'modality': 'gene', 'data_level': 'cell', 'n_obs': 50000},
            {'slice_id': 1, 'tissue_name': 'breast_cancer_rep2', 'modality': 'gene', 'data_level': 'cell', 'n_obs': 45000}
        ]
    }

    # Test
    result = await planner.plan(
        user_query="Annotate cell types",
        session_summary=session_summary,
        previous_clarifications=[]
    )

    # Verify
    assert result.needs_clarification
    assert "which slice" in result.clarification_question.lower()
    print(f"✅ Planner correctly asked for clarification on ambiguous query")


@pytest.mark.asyncio
async def test_query_planner_with_clarification_response():
    """Test planner processes user clarification and generates plan."""
    # Create mock LLM backend
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value='''```json
{
  "needs_clarification": false,
  "steps": [
    {
      "step_number": 1,
      "description": "Annotate cell types in slice 0",
      "target_slice_ids": [0],
      "refined_query": "Annotate cell types in slice 0 (breast_cancer_rep1)"
    }
  ]
}
```''')

    planner = QueryPlanner(llm_backend=mock_llm)

    session_summary = {
        'n_slices': 2,
        'slices': [
            {'slice_id': 0, 'tissue_name': 'breast_cancer_rep1', 'modality': 'gene', 'data_level': 'cell', 'n_obs': 50000},
            {'slice_id': 1, 'tissue_name': 'breast_cancer_rep2', 'modality': 'gene', 'data_level': 'cell', 'n_obs': 45000}
        ]
    }

    # Test with previous clarification
    result = await planner.plan(
        user_query="Annotate cell types",
        session_summary=session_summary,
        previous_clarifications=[("Which slice?", "Slice 0")]
    )

    # Verify
    assert not result.needs_clarification
    assert len(result.steps) == 1
    assert result.steps[0].target_slice_ids == [0]
    print(f"✅ Planner correctly processed clarification and generated plan")


# ============================================================================
# SkillFilter Tests
# ============================================================================

def test_skill_filter_single_slice_cell_gene():
    """Test filter keeps only cell-level gene skills for single slice."""
    skill_filter = SkillFilter()

    # Mock skills
    all_skills = {
        'celltype-annotation-GPT': SkillMetadata(
            slug='celltype-annotation-GPT',
            name='Cell Type Annotation (GPT)',
            filter_requirements={'num_slices': 1, 'modalities': ['gene'], 'data_levels': ['cell']}
        ),
        'celltype-deconvolution-RCTD': SkillMetadata(
            slug='celltype-deconvolution-RCTD',
            name='Cell Type Deconvolution (RCTD)',
            filter_requirements={'num_slices': 1, 'modalities': ['gene'], 'data_levels': ['spot']}
        ),
        'niche-detection-Harmonics': SkillMetadata(
            slug='niche-detection-Harmonics',
            name='Niche Detection',
            filter_requirements={'num_slices': 1, 'modalities': ['gene'], 'data_levels': ['cell']}
        )
    }

    # Mock session
    mock_session = MagicMock(spec=SimpleSession)
    mock_session.get_slice_by_id.return_value = MagicMock(
        modality='gene',
        data_level='cell'
    )

    # Test
    compatible = skill_filter.filter_skills(
        target_slice_ids=[0],
        session=mock_session,
        all_skills=all_skills
    )

    # Verify - should filter out spot-level skill
    assert len(compatible) == 2
    compatible_slugs = [s.slug for s in compatible]
    assert 'celltype-annotation-GPT' in compatible_slugs
    assert 'niche-detection-Harmonics' in compatible_slugs
    assert 'celltype-deconvolution-RCTD' not in compatible_slugs
    print(f"✅ Filter correctly kept {len(compatible)} cell-level gene skills")


def test_skill_filter_spot_level_data():
    """Test filter keeps only spot-level skills for Visium data."""
    skill_filter = SkillFilter()

    # Mock skills
    all_skills = {
        'celltype-annotation-GPT': SkillMetadata(
            slug='celltype-annotation-GPT',
            name='Cell Type Annotation (GPT)',
            filter_requirements={'num_slices': 1, 'modalities': ['gene'], 'data_levels': ['cell']}
        ),
        'celltype-deconvolution-RCTD': SkillMetadata(
            slug='celltype-deconvolution-RCTD',
            name='Cell Type Deconvolution (RCTD)',
            filter_requirements={'num_slices': 1, 'modalities': ['gene'], 'data_levels': ['spot']}
        )
    }

    # Mock session with spot-level data
    mock_session = MagicMock(spec=SimpleSession)
    mock_session.get_slice_by_id.return_value = MagicMock(
        modality='gene',
        data_level='spot'
    )

    # Test
    compatible = skill_filter.filter_skills(
        target_slice_ids=[0],
        session=mock_session,
        all_skills=all_skills
    )

    # Verify - should keep only spot-level skill
    assert len(compatible) == 1
    assert compatible[0].slug == 'celltype-deconvolution-RCTD'
    print(f"✅ Filter correctly kept only spot-level skill for Visium data")


def test_skill_filter_no_requirements():
    """Test filter keeps skills with no requirements."""
    skill_filter = SkillFilter()

    # Mock skills
    all_skills = {
        'general-analysis': SkillMetadata(
            slug='general-analysis',
            name='General Analysis',
            filter_requirements=None  # No requirements
        )
    }

    # Mock session
    mock_session = MagicMock(spec=SimpleSession)
    mock_session.get_slice_by_id.return_value = MagicMock(
        modality='gene',
        data_level='cell'
    )

    # Test
    compatible = skill_filter.filter_skills(
        target_slice_ids=[0],
        session=mock_session,
        all_skills=all_skills
    )

    # Verify - should keep skill with no requirements
    assert len(compatible) == 1
    assert compatible[0].slug == 'general-analysis'
    print(f"✅ Filter correctly kept skill with no requirements")


# ============================================================================
# SkillVerifier Tests
# ============================================================================

@pytest.mark.asyncio
async def test_skill_verifier_prerequisites_met():
    """Test verifier when all prerequisites are met."""
    # Create mock LLM backend
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value='''```json
{
  "prerequisites_met": true,
  "complete_query": "Perform niche detection on slice 0"
}
```''')

    verifier = SkillVerifier(llm_backend=mock_llm)

    # Mock plan step
    plan_step = PlanStep(
        step_number=1,
        description="Detect niches in slice 0",
        target_slice_ids=[0],
        refined_query="Perform niche detection on slice 0"
    )

    # Mock skill with prerequisites
    mock_skill = MagicMock()
    mock_skill.slug = 'niche-detection-Harmonics'
    mock_skill.name = 'Niche Detection'
    mock_skill.prerequisites = ["Cell type annotations in target slice (adata.obs['celltype'])"]

    # Mock session summary with celltype available
    session_summary = {
        'n_slices': 1,
        'slices': [
            {
                'slice_id': 0,
                'tissue_name': 'breast_cancer',
                'modality': 'gene',
                'data_level': 'cell',
                'n_obs': 50000,
                'has_celltype': True,
                'celltypes': ['T cell', 'B cell', 'Tumor']
            }
        ]
    }

    # Test
    result = await verifier.verify(
        plan_step=plan_step,
        selected_skill=mock_skill,
        session_summary=session_summary,
        user_responses={}
    )

    # Verify
    assert result.prerequisites_met
    assert result.complete_query is not None
    print(f"✅ Verifier correctly determined prerequisites are met")


@pytest.mark.asyncio
async def test_skill_verifier_can_ask_user():
    """Test verifier asks user for missing info that can be obtained via chat."""
    # Create mock LLM backend
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value='''```json
{
  "prerequisites_met": false,
  "missing_prerequisites": ["Tissue type", "Reference dataset path"],
  "can_obtain_by_chat": true,
  "clarification_questions": [
    "What type of tissue is this? (e.g., breast cancer, brain, liver)",
    "Please provide the full path to your annotated reference dataset (.h5ad file)"
  ]
}
```''')

    verifier = SkillVerifier(llm_backend=mock_llm)

    # Mock plan step
    plan_step = PlanStep(
        step_number=1,
        description="Annotate cell types in slice 0",
        target_slice_ids=[0],
        refined_query="Annotate cell types in slice 0"
    )

    # Mock skill with prerequisites
    mock_skill = MagicMock()
    mock_skill.slug = 'celltype-annotation-GPT'
    mock_skill.name = 'Cell Type Annotation'
    mock_skill.prerequisites = [
        "Tissue type information (e.g., breast cancer, brain cortex, liver)",
        "Annotated reference dataset path (.h5ad file)"
    ]

    # Mock session summary
    session_summary = {
        'n_slices': 1,
        'slices': [{'slice_id': 0, 'tissue_name': 'unknown', 'modality': 'gene', 'data_level': 'cell', 'n_obs': 50000}]
    }

    # Test
    result = await verifier.verify(
        plan_step=plan_step,
        selected_skill=mock_skill,
        session_summary=session_summary,
        user_responses={}
    )

    # Verify
    assert not result.prerequisites_met
    assert result.can_obtain_by_chat
    assert len(result.clarification_questions) == 2
    print(f"✅ Verifier correctly asked user for missing information")


@pytest.mark.asyncio
async def test_skill_verifier_needs_prior_work():
    """Test verifier advises when prerequisites need prior analysis."""
    # Create mock LLM backend
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value='''```json
{
  "prerequisites_met": false,
  "missing_prerequisites": ["Cell type annotations"],
  "can_obtain_by_chat": false,
  "advice": "Niche detection requires cell type annotations in slice 0. Please first annotate cell types using one of the annotation methods (e.g., 'Annotate cell types in slice 0'), then retry niche detection."
}
```''')

    verifier = SkillVerifier(llm_backend=mock_llm)

    # Mock plan step
    plan_step = PlanStep(
        step_number=1,
        description="Detect niches in slice 0",
        target_slice_ids=[0],
        refined_query="Perform niche detection on slice 0"
    )

    # Mock skill
    mock_skill = MagicMock()
    mock_skill.slug = 'niche-detection-Harmonics'
    mock_skill.name = 'Niche Detection'
    mock_skill.prerequisites = ["Cell type annotations in target slice (adata.obs['celltype'])"]

    # Mock session summary WITHOUT celltype
    session_summary = {
        'n_slices': 1,
        'slices': [
            {
                'slice_id': 0,
                'tissue_name': 'breast_cancer',
                'modality': 'gene',
                'data_level': 'cell',
                'n_obs': 50000,
                'has_celltype': False
            }
        ]
    }

    # Test
    result = await verifier.verify(
        plan_step=plan_step,
        selected_skill=mock_skill,
        session_summary=session_summary,
        user_responses={}
    )

    # Verify
    assert not result.prerequisites_met
    assert not result.can_obtain_by_chat
    assert result.advice is not None
    assert "annotate cell types" in result.advice.lower()
    print(f"✅ Verifier correctly advised prior work needed")


@pytest.mark.asyncio
async def test_skill_verifier_no_prerequisites():
    """Test verifier with skill that has no prerequisites."""
    # Create mock LLM backend (won't be called)
    mock_llm = MagicMock()

    verifier = SkillVerifier(llm_backend=mock_llm)

    # Mock plan step
    plan_step = PlanStep(
        step_number=1,
        description="Some analysis",
        target_slice_ids=[0],
        refined_query="Perform some analysis"
    )

    # Mock skill with NO prerequisites
    mock_skill = MagicMock()
    mock_skill.slug = 'simple-analysis'
    mock_skill.prerequisites = []

    # Test
    result = await verifier.verify(
        plan_step=plan_step,
        selected_skill=mock_skill,
        session_summary={},
        user_responses={}
    )

    # Verify
    assert result.prerequisites_met
    assert mock_llm.run.call_count == 0  # LLM should not be called
    print(f"✅ Verifier correctly skipped verification for skill with no prerequisites")


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_single_slice_no_clarification():
    """Test complete pipeline flow with single slice and no clarifications needed."""
    # This would require a full SpatialAgent setup with mocked session
    # For now, we'll test the individual components work together
    pass  # TODO: Implement when integration testing infrastructure is ready


@pytest.mark.asyncio
async def test_full_pipeline_multi_turn_clarification():
    """Test pipeline handles multi-turn clarification (planner + verifier)."""
    # This would test:
    # 1. Planner asks for slice clarification
    # 2. User responds
    # 3. Skill is matched and filtered
    # 4. Verifier asks for prerequisites
    # 5. User responds
    # 6. Query is executed
    pass  # TODO: Implement when integration testing infrastructure is ready


@pytest.mark.asyncio
async def test_full_pipeline_multiple_skill_selection():
    """Test pipeline handles multiple matched skills and user selection."""
    # This would test:
    # 1. Planning succeeds
    # 2. Multiple skills match
    # 3. User is asked to select
    # 4. User selects skill
    # 5. Verification and execution proceed
    pass  # TODO: Implement when integration testing infrastructure is ready


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_no_skills_matched_warning():
    """Test warning is displayed when no skills match."""
    # Create agent with new pipeline enabled
    agent = SpatialAgent(
        model="gpt-4o-mini",
        enable_skills=True,
        use_new_pipeline=True
    )

    # This would test that when no skills match, a warning is shown
    # But still attempts to help with general capabilities
    pass  # TODO: Implement when agent mocking is ready


def test_skill_filter_with_missing_session_data():
    """Test filter handles gracefully when session data is incomplete."""
    skill_filter = SkillFilter()

    # Mock skills
    all_skills = {
        'test-skill': SkillMetadata(
            slug='test-skill',
            name='Test Skill',
            filter_requirements={'num_slices': 1, 'modalities': ['gene'], 'data_levels': ['cell']}
        )
    }

    # Mock session that returns None for slice
    mock_session = MagicMock(spec=SimpleSession)
    mock_session.get_slice_by_id.return_value = None

    # Test - should handle gracefully
    compatible = skill_filter.filter_skills(
        target_slice_ids=[999],  # Non-existent slice
        session=mock_session,
        all_skills=all_skills
    )

    # Verify - should filter out skill when slice data is missing
    assert len(compatible) == 0
    print(f"✅ Filter gracefully handled missing session data")


@pytest.mark.asyncio
async def test_verifier_with_malformed_llm_response():
    """Test verifier handles malformed LLM responses gracefully."""
    # Create mock LLM backend that returns invalid JSON
    mock_llm = MagicMock()
    mock_llm.run = AsyncMock(return_value="This is not valid JSON")

    verifier = SkillVerifier(llm_backend=mock_llm)

    # Mock plan step and skill
    plan_step = PlanStep(
        step_number=1,
        description="Test",
        target_slice_ids=[0],
        refined_query="Test query"
    )

    mock_skill = MagicMock()
    mock_skill.slug = 'test-skill'
    mock_skill.prerequisites = ["Some prerequisite"]

    # Test
    result = await verifier.verify(
        plan_step=plan_step,
        selected_skill=mock_skill,
        session_summary={},
        user_responses={}
    )

    # Verify - should fallback gracefully
    assert not result.prerequisites_met
    assert result.can_obtain_by_chat  # Fallback assumes we can ask user
    print(f"✅ Verifier gracefully handled malformed LLM response")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

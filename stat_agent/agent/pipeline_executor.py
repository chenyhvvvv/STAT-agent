"""
Pipeline Executor - Pure 5-stage skill selection pipeline.

Executes the skill selection pipeline WITHOUT handling clarifications.
Returns results indicating what the orchestrator should do next.

The 5 stages are:
1. Query Planning - Determine target slices and refine query
2. Skill Filtering - Programmatic filtering by format/modality/data_level
3. Semantic Matching - LLM-based skill selection
4. Skill Verification - Prerequisites checking
5. Result Assembly - Prepare for execution

Key principle: This is a PURE pipeline - no clarification handling,
just return results indicating what needs to happen next.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, AsyncIterator

from .query_planner import QueryPlanner, PlanStep, PlanResult
from .skill_filter import SkillFilter
from .skill_verifier import SkillVerifier, VerificationResult
from .skill_registry import SkillRegistry, SkillDefinition

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from pipeline execution.

    The pipeline can return different result types based on what happened:
    - 'success': Ready to execute with selected skill
    - 'planner_clarification': Planner needs user input
    - 'skill_selection': Multiple skills matched, user must choose
    - 'verifier_clarification': Skill needs prerequisite information
    - 'advice': Cannot proceed, needs prior work
    - 'no_skill': No skill matched, proceed with general LLM
    """

    # Result type
    type: str  # 'success', 'planner_clarification', 'skill_selection',
               # 'verifier_clarification', 'advice', 'no_skill'

    # For success / no_skill execution
    selected_skill: Optional[SkillDefinition] = None
    final_query: str = ""
    plan_step: Optional[PlanStep] = None
    plan_steps: List[PlanStep] = field(default_factory=list)  # All steps in plan

    # For clarifications
    clarification_question: Optional[str] = None
    clarification_type: str = ""  # 'planner', 'verifier', 'skill_selection'
    clarification_context: Dict[str, Any] = field(default_factory=dict)

    # For skill selection
    skill_options: List[str] = field(default_factory=list)  # List of skill slugs

    # For verifier clarification
    verifier_questions: List[str] = field(default_factory=list)

    # For advice
    advice_message: str = ""

    # Metadata
    step_number: int = 1
    total_steps: int = 1
    no_skill_matched: bool = False


class PipelineExecutor:
    """Executes 5-stage skill selection pipeline.

    Pure pipeline - NO clarification handling.
    Returns results indicating what orchestrator should do next.

    The pipeline:
    1. Query Planning - Break down user query into steps
    2. Skill Filtering - Filter by compatibility
    3. Semantic Matching - Select best matching skills
    4. Skill Verification - Check prerequisites
    5. Result Assembly - Prepare execution

    This class does NOT:
    - Handle clarification responses
    - Execute code
    - Manage multi-turn state
    - Detect state changes
    Those are orchestration layer responsibilities.
    """

    def __init__(
        self,
        query_planner: QueryPlanner,
        skill_filter: SkillFilter,
        skill_verifier: SkillVerifier,
        skill_registry: SkillRegistry,
        semantic_matcher,  # The _select_skill_matches_llm_filtered method from agent
        session=None  # SimpleSession for skill filtering
    ):
        """Initialize pipeline executor.

        Parameters
        ----------
        query_planner : QueryPlanner
            Query planning component
        skill_filter : SkillFilter
            Skill filtering component
        skill_verifier : SkillVerifier
            Skill verification component
        skill_registry : SkillRegistry
            Skill registry for loading skills
        semantic_matcher : callable
            Async function for semantic skill matching
        session : SimpleSession, optional
            Current session (needed for skill filtering by modality/data_level)
        """
        self.query_planner = query_planner
        self.skill_filter = skill_filter
        self.skill_verifier = skill_verifier
        self.skill_registry = skill_registry
        self.semantic_matcher = semantic_matcher
        self.session = session

    async def execute_pipeline(
        self,
        user_query: str,
        session_summary: Dict[str, Any],
        planner_history: Optional[List[tuple]] = None,
        verifier_responses: Optional[Dict[str, Dict[str, str]]] = None,
        conversation_history: str = ""
    ) -> PipelineResult:
        """Execute pipeline and return result (non-streaming).

        Parameters
        ----------
        user_query : str
            User's query
        session_summary : Dict[str, Any]
            Current session state
        planner_history : Optional[List[tuple]]
            Previous planner clarifications [(question, answer), ...]
        verifier_responses : Optional[Dict[str, Dict[str, str]]]
            Previous verifier responses {skill_slug: {question: answer}}
        conversation_history : str
            Formatted conversation history for planner/verifier context

        Returns
        -------
        PipelineResult
            Result indicating what to do next
        """
        if verifier_responses is None:
            verifier_responses = {}

        logger.info("="*60)
        logger.info("PIPELINE START: Processing user query")
        logger.info(f"Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
        logger.info("="*60)

        # Stage 1: QUERY PLANNING
        logger.info("-" * 60)
        logger.info("STAGE 1: QUERY PLANNING")
        logger.info("-" * 60)

        plan_result = await self.query_planner.plan(
            user_query=user_query,
            session_summary=session_summary,
            previous_clarifications=planner_history or [],
            conversation_history=conversation_history
        )

        # Handle planner clarification
        if plan_result.needs_clarification:
            logger.info(f"❓ Planner requesting clarification")
            logger.info(f"  Question: {plan_result.clarification_question[:100]}...")
            return PipelineResult(
                type='planner_clarification',
                clarification_question=plan_result.clarification_question,
                clarification_type='planner'
            )

        logger.info(f"✓ Planner generated {len(plan_result.steps)} step(s)")

        # For now, handle only single-step plans
        # TODO: Multi-step support requires orchestration layer changes
        if len(plan_result.steps) != 1:
            logger.warning(f"Multi-step plans not yet supported in Phase 2, using first step only")

        step = plan_result.steps[0]
        logger.info(f"Processing Step {step.step_number}: {step.description}")

        # Stage 2: FILTER
        logger.info("-" * 60)
        logger.info(f"STAGE 2: FILTER")
        logger.info(f"Target slices: {step.target_slice_ids}")
        logger.info("-" * 60)

        compatible_skills = self.skill_filter.filter_skills(
            target_slice_ids=step.target_slice_ids,
            session=self.session,
            all_skills=self.skill_registry.skill_metadata
        )
        logger.info(f"✓ Filter result: {len(compatible_skills)} compatible skills")

        # Stage 3: SEMANTIC MATCHING
        logger.info("-" * 60)
        logger.info(f"STAGE 3: SEMANTIC MATCHING")
        logger.info(f"Query: {step.refined_query}")
        logger.info("-" * 60)

        compatible_skills_dict = {skill.slug: skill for skill in compatible_skills}
        matched_slugs = await self.semantic_matcher(
            request=step.refined_query,
            available_skills=compatible_skills_dict,
            top_k=2
        )

        logger.info(f"✓ Semantic matcher found: {len(matched_slugs)} skill(s)")

        # Handle matching results
        logger.info("-" * 60)
        logger.info(f"MATCHING RESULT")
        logger.info("-" * 60)

        if len(matched_slugs) == 0:
            # No skill matched - proceed without skill guidance
            logger.warning(f"⚠️  NO SPECIALIZED SKILL MATCHED")
            logger.info(f"  Step: {step.description}")
            logger.info(f"  Will attempt with general LLM capabilities")
            return PipelineResult(
                type='no_skill',
                selected_skill=None,
                final_query=step.refined_query,
                plan_step=step,
                plan_steps=plan_result.steps,
                no_skill_matched=True
            )

        elif len(matched_slugs) == 1:
            # Single skill matched - proceed to verification
            selected_skill_slug = matched_slugs[0]
            selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)
            logger.info(f"✓ Single skill matched: {selected_skill_slug}")

        else:
            # Multiple skills matched - need user selection
            logger.info(f"🎯 Multiple skills matched: {len(matched_slugs)}")
            logger.info(f"  Matched: {matched_slugs}")
            logger.info(f"  Requesting user selection...")

            return PipelineResult(
                type='skill_selection',
                skill_options=matched_slugs,
                clarification_type='skill_selection',
                plan_step=step,
                plan_steps=plan_result.steps,
                clarification_context={
                    'plan_step': step,
                    'skill_options': matched_slugs
                }
            )

        # Stage 4: VERIFY prerequisites
        if selected_skill and self.skill_verifier:
            logger.info("-" * 60)
            logger.info(f"STAGE 4: VERIFICATION")
            logger.info(f"Skill: {selected_skill.slug}")
            logger.info("-" * 60)

            # Get any previous responses for this skill
            user_responses = verifier_responses.get(selected_skill.slug, {})

            verification_result = await self.skill_verifier.verify(
                plan_step=step,
                selected_skill=selected_skill,
                session_summary=session_summary,
                user_responses=user_responses,
                conversation_history=conversation_history
            )

            if not verification_result.prerequisites_met:
                if verification_result.can_obtain_by_chat:
                    # Need to ask user for info
                    logger.info(f"❓ Prerequisites missing - can obtain via chat")
                    logger.info(f"  Missing: {verification_result.missing_prerequisites}")
                    logger.info(f"  Asking {len(verification_result.clarification_questions)} question(s)")

                    return PipelineResult(
                        type='verifier_clarification',
                        selected_skill=selected_skill,
                        plan_step=step,
                        plan_steps=plan_result.steps,
                        verifier_questions=verification_result.clarification_questions,
                        clarification_type='verifier',
                        clarification_context={
                            'skill_slug': selected_skill.slug,
                            'plan_step': step,
                            'questions': verification_result.clarification_questions
                        }
                    )
                else:
                    # Can't proceed - needs prior work
                    logger.warning(f"⚠️  Prerequisites missing - needs prior work")
                    logger.info(f"  Missing: {verification_result.missing_prerequisites}")
                    logger.info(f"  Advice: {verification_result.advice[:100]}...")

                    return PipelineResult(
                        type='advice',
                        advice_message=verification_result.advice,
                        selected_skill=selected_skill,
                        plan_step=step,
                        plan_steps=plan_result.steps
                    )

            # Prerequisites met - use complete query
            final_query = verification_result.complete_query
            logger.info(f"✓ Prerequisites met")
            logger.info(f"  Complete query: {final_query[:80]}...")
        else:
            # No skill or no verifier - use original query
            final_query = step.refined_query
            if not selected_skill:
                logger.info("  No skill selected - using original query")

        # Stage 5: RESULT ASSEMBLY
        logger.info("-" * 60)
        logger.info("STAGE 5: RESULT ASSEMBLY")
        logger.info(f"Ready for execution with skill: {selected_skill.slug if selected_skill else 'None'}")
        logger.info("-" * 60)

        return PipelineResult(
            type='success',
            selected_skill=selected_skill,
            final_query=final_query,
            plan_step=step,
            plan_steps=plan_result.steps,
            no_skill_matched=False
        )

    async def execute_pipeline_with_events(
        self,
        user_query: str,
        session_summary: Dict[str, Any],
        planner_history: Optional[List[tuple]] = None,
        verifier_responses: Optional[Dict[str, Dict[str, str]]] = None,
        conversation_history: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute pipeline with event streaming (for progress tracking).

        Yields events like:
        - {'type': 'planning_start'}
        - {'type': 'planning_complete', 'steps': 1}
        - {'type': 'filter_complete', 'count': 3}
        - {'type': 'matching_complete', 'matched': 2}
        - {'type': 'verification_start', 'skill': 'skill-slug'}
        - {'type': 'pipeline_result', 'result': PipelineResult}

        Parameters
        ----------
        user_query : str
            User's query
        session_summary : Dict[str, Any]
            Current session state
        planner_history : Optional[List[tuple]]
            Previous planner clarifications
        verifier_responses : Optional[Dict[str, Dict[str, str]]]
            Previous verifier responses
        conversation_history : str
            Formatted conversation history for planner/verifier context

        Yields
        ------
        Dict[str, Any]
            Progress events
        """
        if verifier_responses is None:
            verifier_responses = {}

        logger.info("="*60)
        logger.info("PIPELINE START (STREAMING): Processing user query")
        logger.info(f"Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
        logger.info("="*60)

        # Stage 1: QUERY PLANNING
        yield {'type': 'planning_start'}

        logger.info("-" * 60)
        logger.info("STAGE 1: QUERY PLANNING")
        logger.info("-" * 60)

        plan_result = await self.query_planner.plan(
            user_query=user_query,
            session_summary=session_summary,
            previous_clarifications=planner_history or [],
            conversation_history=conversation_history
        )

        # Handle planner clarification
        if plan_result.needs_clarification:
            logger.info(f"❓ Planner requesting clarification")
            result = PipelineResult(
                type='planner_clarification',
                clarification_question=plan_result.clarification_question,
                clarification_type='planner'
            )
            yield {'type': 'pipeline_result', 'result': result}
            return

        logger.info(f"✓ Planner generated {len(plan_result.steps)} step(s)")
        yield {'type': 'planning_complete', 'steps': len(plan_result.steps)}

        # For now, handle only single-step plans
        if len(plan_result.steps) != 1:
            logger.warning(f"Multi-step plans not yet supported in Phase 2, using first step only")

        step = plan_result.steps[0]
        logger.info(f"Processing Step {step.step_number}: {step.description}")
        yield {'type': 'step_start', 'step_number': step.step_number, 'description': step.description}

        # Stage 2: FILTER
        logger.info("-" * 60)
        logger.info(f"STAGE 2: FILTER")
        logger.info(f"Target slices: {step.target_slice_ids}")
        logger.info("-" * 60)

        compatible_skills = self.skill_filter.filter_skills(
            target_slice_ids=step.target_slice_ids,
            session=self.session,
            all_skills=self.skill_registry.skill_metadata
        )
        logger.info(f"✓ Filter result: {len(compatible_skills)} compatible skills")
        yield {'type': 'filter_complete', 'count': len(compatible_skills)}

        # Stage 3: SEMANTIC MATCHING
        logger.info("-" * 60)
        logger.info(f"STAGE 3: SEMANTIC MATCHING")
        logger.info(f"Query: {step.refined_query}")
        logger.info("-" * 60)

        compatible_skills_dict = {skill.slug: skill for skill in compatible_skills}
        matched_slugs = await self.semantic_matcher(
            request=step.refined_query,
            available_skills=compatible_skills_dict,
            top_k=2
        )

        logger.info(f"✓ Semantic matcher found: {len(matched_slugs)} skill(s)")
        yield {'type': 'matching_complete', 'matched': len(matched_slugs)}

        # Handle matching results
        if len(matched_slugs) == 0:
            logger.warning(f"⚠️  NO SPECIALIZED SKILL MATCHED")
            result = PipelineResult(
                type='no_skill',
                final_query=step.refined_query,
                plan_step=step,
                plan_steps=plan_result.steps,
                no_skill_matched=True
            )
            yield {'type': 'pipeline_result', 'result': result}
            return

        elif len(matched_slugs) == 1:
            selected_skill_slug = matched_slugs[0]
            selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)
            logger.info(f"✓ Single skill matched: {selected_skill_slug}")
            yield {'type': 'skill_matched', 'skill': selected_skill_slug}

        else:
            logger.info(f"🎯 Multiple skills matched: {len(matched_slugs)}")
            result = PipelineResult(
                type='skill_selection',
                skill_options=matched_slugs,
                clarification_type='skill_selection',
                plan_step=step,
                plan_steps=plan_result.steps,
                clarification_context={
                    'plan_step': step,
                    'skill_options': matched_slugs
                }
            )
            yield {'type': 'pipeline_result', 'result': result}
            return

        # Stage 4: VERIFICATION
        if selected_skill and self.skill_verifier:
            logger.info("-" * 60)
            logger.info(f"STAGE 4: VERIFICATION")
            logger.info(f"Skill: {selected_skill.slug}")
            logger.info("-" * 60)
            yield {'type': 'verification_start', 'skill': selected_skill.slug}

            user_responses = verifier_responses.get(selected_skill.slug, {})
            verification_result = await self.skill_verifier.verify(
                plan_step=step,
                selected_skill=selected_skill,
                session_summary=session_summary,
                user_responses=user_responses,
                conversation_history=conversation_history
            )

            if not verification_result.prerequisites_met:
                if verification_result.can_obtain_by_chat:
                    logger.info(f"❓ Prerequisites missing - can obtain via chat")
                    result = PipelineResult(
                        type='verifier_clarification',
                        selected_skill=selected_skill,
                        plan_step=step,
                        plan_steps=plan_result.steps,
                        verifier_questions=verification_result.clarification_questions,
                        clarification_type='verifier',
                        clarification_context={
                            'skill_slug': selected_skill.slug,
                            'plan_step': step,
                            'questions': verification_result.clarification_questions
                        }
                    )
                    yield {'type': 'pipeline_result', 'result': result}
                    return
                else:
                    logger.warning(f"⚠️  Prerequisites missing - needs prior work")
                    result = PipelineResult(
                        type='advice',
                        advice_message=verification_result.advice,
                        selected_skill=selected_skill,
                        plan_step=step,
                        plan_steps=plan_result.steps
                    )
                    yield {'type': 'pipeline_result', 'result': result}
                    return

            final_query = verification_result.complete_query
            logger.info(f"✓ Prerequisites met")
            yield {'type': 'verification_complete', 'status': 'met'}
        else:
            final_query = step.refined_query

        # Stage 5: SUCCESS
        logger.info("-" * 60)
        logger.info("STAGE 5: RESULT ASSEMBLY")
        logger.info(f"Ready for execution")
        logger.info("-" * 60)

        result = PipelineResult(
            type='success',
            selected_skill=selected_skill,
            final_query=final_query,
            plan_step=step,
            plan_steps=plan_result.steps,
            no_skill_matched=False
        )
        yield {'type': 'pipeline_result', 'result': result}

    async def execute_single_step(
        self,
        step: PlanStep,
        session_summary: Dict[str, Any],
        verifier_responses: Optional[Dict[str, Dict[str, str]]] = None,
        conversation_history: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute a single plan step through filter → match → verify.

        This method processes ONE step from a multi-step plan (skipping planning stage).
        Used for executing steps 2, 3, etc. after initial planning.

        Yields pipeline detail events for real-time progress tracking, and
        a final {'type': 'pipeline_result', 'result': PipelineResult} event.

        Parameters
        ----------
        step : PlanStep
            The plan step to execute
        session_summary : Dict[str, Any]
            Current session state
        verifier_responses : Optional[Dict[str, Dict[str, str]]]
            Previous verifier responses
        conversation_history : str
            Formatted conversation history for verifier context

        Yields
        ------
        Dict[str, Any]
            Pipeline detail events and final pipeline_result
        """
        if verifier_responses is None:
            verifier_responses = {}

        logger.info("="*60)
        logger.info(f"EXECUTING SINGLE STEP {step.step_number}")
        logger.info(f"Description: {step.description}")
        logger.info("="*60)

        # Stage 2: FILTER
        logger.info("-" * 60)
        logger.info(f"STAGE 2: FILTER")
        logger.info(f"Target slices: {step.target_slice_ids}")
        logger.info("-" * 60)

        compatible_skills = self.skill_filter.filter_skills(
            target_slice_ids=step.target_slice_ids,
            session=self.session,
            all_skills=self.skill_registry.skill_metadata
        )
        logger.info(f"✓ Filter result: {len(compatible_skills)} compatible skills")
        yield {'type': 'filter_complete', 'count': len(compatible_skills)}

        # Stage 3: SEMANTIC MATCHING
        logger.info("-" * 60)
        logger.info(f"STAGE 3: SEMANTIC MATCHING")
        logger.info(f"Query: {step.refined_query}")
        logger.info("-" * 60)

        compatible_skills_dict = {skill.slug: skill for skill in compatible_skills}
        matched_slugs = await self.semantic_matcher(
            request=step.refined_query,
            available_skills=compatible_skills_dict,
            top_k=2
        )

        logger.info(f"✓ Semantic matcher found: {len(matched_slugs)} skill(s)")
        yield {'type': 'matching_complete', 'matched': len(matched_slugs)}

        # Handle matching results
        if len(matched_slugs) == 0:
            logger.warning(f"⚠️  NO SPECIALIZED SKILL MATCHED")
            yield {'type': 'pipeline_result', 'result': PipelineResult(
                type='no_skill',
                final_query=step.refined_query,
                plan_step=step,
                no_skill_matched=True
            )}
            return

        elif len(matched_slugs) == 1:
            selected_skill_slug = matched_slugs[0]
            selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)
            logger.info(f"✓ Single skill matched: {selected_skill_slug}")
            yield {'type': 'skill_matched', 'skill': selected_skill_slug}

        else:
            logger.info(f"🎯 Multiple skills matched: {len(matched_slugs)}")
            yield {'type': 'pipeline_result', 'result': PipelineResult(
                type='skill_selection',
                skill_options=matched_slugs,
                clarification_type='skill_selection',
                plan_step=step,
                clarification_context={
                    'plan_step': step,
                    'skill_options': matched_slugs
                }
            )}
            return

        # Stage 4: VERIFY
        if selected_skill and self.skill_verifier:
            logger.info("-" * 60)
            logger.info(f"STAGE 4: VERIFICATION")
            logger.info(f"Skill: {selected_skill.slug}")
            logger.info("-" * 60)
            yield {'type': 'verification_start', 'skill': selected_skill.slug}

            user_responses = verifier_responses.get(selected_skill.slug, {})
            verification_result = await self.skill_verifier.verify(
                plan_step=step,
                selected_skill=selected_skill,
                session_summary=session_summary,
                user_responses=user_responses,
                conversation_history=conversation_history
            )

            if not verification_result.prerequisites_met:
                if verification_result.can_obtain_by_chat:
                    logger.info(f"❓ Prerequisites missing - can obtain via chat")
                    yield {'type': 'pipeline_result', 'result': PipelineResult(
                        type='verifier_clarification',
                        selected_skill=selected_skill,
                        plan_step=step,
                        verifier_questions=verification_result.clarification_questions,
                        clarification_type='verifier',
                        clarification_context={
                            'skill_slug': selected_skill.slug,
                            'plan_step': step,
                            'questions': verification_result.clarification_questions
                        }
                    )}
                    return
                else:
                    logger.warning(f"⚠️  Prerequisites missing - needs prior work")
                    yield {'type': 'pipeline_result', 'result': PipelineResult(
                        type='advice',
                        advice_message=verification_result.advice,
                        selected_skill=selected_skill,
                        plan_step=step
                    )}
                    return

            final_query = verification_result.complete_query
            logger.info(f"✓ Prerequisites met")
            yield {'type': 'verification_complete', 'status': 'met'}
        else:
            final_query = step.refined_query

        # Stage 5: SUCCESS
        logger.info("-" * 60)
        logger.info("STAGE 5: RESULT ASSEMBLY")
        logger.info(f"Ready for execution")
        logger.info("-" * 60)

        yield {'type': 'pipeline_result', 'result': PipelineResult(
            type='success',
            selected_skill=selected_skill,
            final_query=final_query,
            plan_step=step,
            no_skill_matched=False
        )}

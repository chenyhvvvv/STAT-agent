"""
Conversation Orchestrator - Multi-turn conversation coordination.

This is the orchestration layer that coordinates everything:
- Multi-turn conversation state
- Pipeline execution (via PipelineExecutor)
- Code execution (via _handle_with_llm_events)
- Clarification handling (all 3 types)
- State change detection
- Event streaming

Key principle: This layer COORDINATES but doesn't execute. It delegates to:
- PipelineExecutor for skill selection
- Code executor for running code
- ClarificationContext for state management

This separation prevents bugs like:
- Lost original query during clarifications
- Early returns skipping state detection
- Tangled control flow
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, AsyncIterator

from .clarification_context import ClarificationContext
from .pipeline_executor import PipelineExecutor, PipelineResult
from .skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Orchestrates multi-turn conversations.

    Responsibilities:
    - Manage conversation state
    - Coordinate pipeline execution
    - Handle clarification flows (3 types: planner, verifier, skill_selection)
    - Coordinate code execution
    - Detect state changes
    - Emit events

    Does NOT:
    - Execute pipeline (delegates to PipelineExecutor)
    - Execute code (delegates to code_executor)
    - Do planning/filtering (pipeline components)

    This design prevents the historical bugs:
    - Clarification workflow bug: Original query preserved at orchestration level
    - State detection bug: State detection always runs (no early returns in pipeline)
    - Filter bypass bug: Clear separation prevents double matching
    """

    def __init__(
        self,
        pipeline_executor: PipelineExecutor,
        code_executor,  # The _handle_with_llm_events method
        clarification_context: ClarificationContext,
        skill_registry: SkillRegistry,
        session,  # SimpleSession
        clear_clarification_fn,  # The _clear_clarification_context method
        memory=None  # ConversationMemory for conversation history context
    ):
        """Initialize conversation orchestrator.

        Parameters
        ----------
        pipeline_executor : PipelineExecutor
            Pipeline execution component
        code_executor : callable
            Async function for code execution (agent's _handle_with_llm_events)
        clarification_context : ClarificationContext
            Clarification state management
        skill_registry : SkillRegistry
            Skill registry for loading skills
        session : SimpleSession
            Current session
        clear_clarification_fn : callable
            Function to clear clarification context
        memory : ConversationMemory, optional
            Conversation memory for history context in planner/verifier
        """
        self.pipeline_executor = pipeline_executor
        self.code_executor = code_executor
        self.clarification_context = clarification_context
        self.skill_registry = skill_registry
        self.session = session
        self.clear_clarification_fn = clear_clarification_fn
        self.memory = memory

    def _get_conversation_history(self) -> str:
        """Get conversation history string for planner/verifier context."""
        if self.memory:
            return self.memory.get_history_string()
        return ""

    async def handle_turn_with_events(
        self,
        user_message: str,
        execute_code: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle a conversation turn with event streaming.

        This is the main entry point for processing user messages.
        It coordinates pipeline execution, clarification handling,
        and code execution.

        Parameters
        ----------
        user_message : str
            User's message (may be clarification response or new query)
        execute_code : bool
            Whether to execute generated code

        Yields
        ------
        Dict[str, Any]
            Events with types:
            - clarification_needed, skill_selection, prerequisites_needed
            - execution_start, execution_output, execution_complete
            - pipeline_complete, warning, advice
        """
        # Check if this is a clarification response
        is_clarification = self.clarification_context.is_waiting_for_clarification()

        if is_clarification:
            # Handle clarification response
            async for event in self._handle_clarification_response(
                user_message, execute_code
            ):
                yield event
        else:
            # New query - execute pipeline
            async for event in self._execute_new_query(
                user_message, execute_code
            ):
                yield event

    async def _handle_clarification_response(
        self,
        response: str,
        execute_code: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle user's response to a clarification.

        This method handles all three clarification types:
        1. Skill selection - User chose from multiple matching skills
        2. Verifier clarification - User provided prerequisite information
        3. Planner clarification - User answered query planner question

        Parameters
        ----------
        response : str
            User's response to the clarification
        execute_code : bool
            Whether to execute code

        Yields
        ------
        Dict[str, Any]
            Events
        """
        clarification_type = self.clarification_context.get_pending_type()
        logger.info(f"Handling {clarification_type} clarification response")

        if clarification_type == 'skill_selection':
            async for event in self._handle_skill_selection_response(response, execute_code):
                yield event

        elif clarification_type == 'verifier':
            async for event in self._handle_verifier_response(response, execute_code):
                yield event

        elif clarification_type == 'planner':
            async for event in self._handle_planner_response(response, execute_code):
                yield event

        else:
            logger.error(f"Unknown clarification type: {clarification_type}")
            yield {"type": "error", "message": "Unknown clarification type"}

    async def _handle_skill_selection_response(
        self,
        response: str,
        execute_code: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle user's skill selection response.

        Parameters
        ----------
        response : str
            User's selection (number or skill name)
        execute_code : bool
            Whether to execute code

        Yields
        ------
        Dict[str, Any]
            Events
        """
        pending = self.clarification_context.get_pending_skill_selection()
        if not pending:
            logger.error("No pending skill selection")
            return

        plan_step = pending['plan_step']
        skill_options = pending['skill_options']
        step_num, total_steps = self.clarification_context.get_step_progress()

        # Parse user's selection
        selected_skill_slug = self.clarification_context.handle_skill_selection(response)

        if not selected_skill_slug:
            # Invalid selection - ask again
            logger.warning(f"Invalid skill selection: {response}")
            skill_options_with_names = []
            for slug in skill_options:
                skill = self.skill_registry.load_full_skill(slug)
                skill_name = skill.name if skill else slug
                skill_options_with_names.append({"slug": slug, "name": skill_name})

            yield {
                "type": "skill_selection",
                "message": "Invalid selection. Please choose one of:",
                "options": skill_options_with_names,
                "step_number": step_num,
                "total_steps": total_steps
            }
            return

        logger.info(f"User selected skill: {selected_skill_slug}")
        selected_skill = self.skill_registry.load_full_skill(selected_skill_slug)

        # Clear selection context
        self.clarification_context.clear_skill_selection()

        # Run verification for selected skill
        if selected_skill and self.pipeline_executor.skill_verifier:
            logger.info(f"Running verification for selected skill: {selected_skill_slug}")
            yield {'type': 'verification_start', 'skill': selected_skill_slug}

            user_responses = self.clarification_context.get_verifier_responses(selected_skill_slug)

            verification_result = await self.pipeline_executor.skill_verifier.verify(
                plan_step=plan_step,
                selected_skill=selected_skill,
                session_summary=self.session.get_summary() if self.session else {},
                user_responses=user_responses,
                conversation_history=self._get_conversation_history()
            )

            if not verification_result.prerequisites_met:
                if verification_result.can_obtain_by_chat:
                    # Need to ask user for prerequisites
                    logger.info(f"Prerequisites missing, asking user for info...")
                    self.clarification_context.store_verifier_clarification(
                        skill_slug=selected_skill_slug,
                        plan_step=plan_step,
                        questions=verification_result.clarification_questions
                    )
                    yield {
                        "type": "prerequisites_needed",
                        "questions": verification_result.clarification_questions,
                        "skill": selected_skill_slug,
                        "step_number": step_num,
                        "total_steps": total_steps
                    }
                    return
                else:
                    # Can't proceed - needs prior work
                    logger.info(f"Prerequisites missing, needs prior work")
                    self.clear_clarification_fn()
                    yield {
                        "type": "advice",
                        "message": verification_result.advice,
                        "step_number": step_num,
                        "total_steps": total_steps
                    }
                    return

            # Prerequisites met - execute
            final_query = verification_result.complete_query
            logger.info(f"Prerequisites met, executing with complete query")
            yield {'type': 'verification_complete', 'status': 'met'}

        else:
            # No verifier or no skill - use original query
            final_query = plan_step.refined_query

        # Emit step_start so frontend can create step container
        # (needed because the original step_start was in a previous stream)
        yield {
            'type': 'step_start',
            'step_number': step_num,
            'total_steps': total_steps,
            'description': plan_step.description
        }

        # Execute with selected skill
        matched_skill_slugs = [selected_skill_slug] if selected_skill else []
        yield {
            'type': 'execution_start',
            'query': final_query,
            'step_number': step_num,
            'total_steps': total_steps
        }

        step_response = None
        step_plots = []
        async for event in self.code_executor(
            final_query,
            execute_code=execute_code,
            allow_planning=False,
            matched_skill_slugs=matched_skill_slugs
        ):
            yield event
            if event['type'] == 'execution_complete':
                step_response = event['response']
                step_plots = event.get('plots', [])

        # Clear per-step state
        self.clarification_context.clear_step_state()

        # Emit step completion
        yield {
            "type": "step_execution_complete",
            "step_number": step_num,
            "total_steps": total_steps,
            "response": step_response,
            "plots": step_plots
        }

        # Continue with remaining steps
        has_more = self.clarification_context.advance_to_next_step()
        if has_more:
            logger.info("Continuing with next step...")
            # Each step gets a fresh pipeline - no verifier info from previous steps
            async for event in self._execute_all_steps(execute_code, {}):
                yield event
        else:
            # All steps complete
            self.clear_clarification_fn()
            yield {
                "type": "pipeline_complete",
                "total_steps": total_steps
            }

    async def _handle_verifier_response(
        self,
        response: str,
        execute_code: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle user's response to verifier clarification.

        Parameters
        ----------
        response : str
            User's answer to prerequisite question(s)
        execute_code : bool
            Whether to execute code

        Yields
        ------
        Dict[str, Any]
            Events
        """
        plan_step = self.clarification_context.get_pending_plan_step()
        skill_slug = self.clarification_context.get_pending_skill()
        step_num, total_steps = self.clarification_context.get_step_progress()

        if not plan_step or not skill_slug:
            logger.error("No pending verifier clarification")
            return

        # Store the response
        self.clarification_context.handle_verifier_response(response)
        logger.info(f"Received verifier clarification: {response[:50]}...")

        # Continue with verification
        skill = self.skill_registry.load_full_skill(skill_slug)
        user_responses = self.clarification_context.get_verifier_responses(skill_slug)

        verification_result = await self.pipeline_executor.skill_verifier.verify(
            plan_step=plan_step,
            selected_skill=skill,
            session_summary=self.session.get_summary() if self.session else {},
            user_responses=user_responses,
            conversation_history=self._get_conversation_history()
        )

        if not verification_result.prerequisites_met:
            if verification_result.can_obtain_by_chat:
                # Still need more info
                self.clarification_context._last_verifier_questions = verification_result.clarification_questions
                yield {
                    "type": "prerequisites_needed",
                    "questions": verification_result.clarification_questions,
                    "skill": skill_slug,
                    "step_number": step_num,
                    "total_steps": total_steps
                }
                return
            else:
                # Can't proceed
                self.clear_clarification_fn()
                yield {
                    "type": "advice",
                    "message": verification_result.advice,
                    "step_number": step_num,
                    "total_steps": total_steps
                }
                return

        # Prerequisites met - execute
        complete_query = verification_result.complete_query
        logger.info(f"Prerequisites met, executing")
        yield {'type': 'verification_complete', 'status': 'met'}

        # Emit step_start so frontend can create step container
        # (needed because the original step_start was in a previous stream)
        yield {
            'type': 'step_start',
            'step_number': step_num,
            'total_steps': total_steps,
            'description': plan_step.description
        }

        matched_skill_slugs = [skill_slug]
        yield {
            'type': 'execution_start',
            'query': complete_query,
            'step_number': step_num,
            'total_steps': total_steps
        }

        step_response = None
        step_plots = []
        async for event in self.code_executor(
            complete_query,
            execute_code=execute_code,
            allow_planning=False,
            matched_skill_slugs=matched_skill_slugs
        ):
            yield event
            if event['type'] == 'execution_complete':
                step_response = event['response']
                step_plots = event.get('plots', [])

        # Clear per-step state
        self.clarification_context.clear_step_state()

        # Emit step completion
        yield {
            "type": "step_execution_complete",
            "step_number": step_num,
            "total_steps": total_steps,
            "response": step_response,
            "plots": step_plots
        }

        # Continue with remaining steps
        has_more = self.clarification_context.advance_to_next_step()
        if has_more:
            logger.info("Continuing with next step...")
            # Each step gets a fresh pipeline - no verifier info from previous steps
            async for event in self._execute_all_steps(execute_code, {}):
                yield event
        else:
            # All steps complete
            self.clear_clarification_fn()
            yield {
                "type": "pipeline_complete",
                "total_steps": total_steps
            }

    async def _handle_planner_response(
        self,
        response: str,
        execute_code: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle user's response to planner clarification.

        CRITICAL: This is where the clarification workflow bug was fixed.
        We must use the ORIGINAL query (not the clarification response)
        when re-planning.

        Parameters
        ----------
        response : str
            User's answer to planner question
        execute_code : bool
            Whether to execute code

        Yields
        ------
        Dict[str, Any]
            Events
        """
        # Get original query (CRITICAL FIX for clarification workflow bug)
        original_query = self.clarification_context.handle_planner_response(response)
        logger.info(f"Using original query for re-planning: {original_query[:100]}...")

        # Re-plan with the ORIGINAL query and updated clarification history
        async for event in self._execute_new_query(original_query, execute_code):
            yield event

    async def _execute_new_query(
        self,
        user_message: str,
        execute_code: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute a new query through the pipeline.

        Plans ONCE, then iterates through all steps.
        Each step goes through: Filter → Match → Verify → Execute independently.

        Parameters
        ----------
        user_message : str
            User's query
        execute_code : bool
            Whether to execute code

        Yields
        ------
        Dict[str, Any]
            Events
        """
        logger.info("="*60)
        logger.info("ORCHESTRATOR: Executing new query")
        logger.info(f"Query: {user_message[:100]}...")
        logger.info("="*60)

        # Get context from clarification_context
        planner_history = self.clarification_context.get_planner_history()

        # STAGE 1: PLAN ONCE (get all steps)
        logger.info("-" * 60)
        logger.info("STAGE 1: QUERY PLANNING")
        logger.info("-" * 60)

        plan_result = await self.pipeline_executor.query_planner.plan(
            user_query=user_message,
            session_summary=self.session.get_summary() if self.session else {},
            previous_clarifications=planner_history,
            conversation_history=self._get_conversation_history()
        )

        # Handle planner clarification
        if plan_result.needs_clarification:
            logger.info(f"❓ Planner requesting clarification")
            self.clarification_context.store_planner_clarification(
                question=plan_result.clarification_question,
                original_query=user_message
            )
            yield {
                "type": "clarification_needed",
                "question": plan_result.clarification_question,
                "context": "planner"
            }
            return

        # Store all plan steps
        logger.info(f"✓ Plan complete: {len(plan_result.steps)} step(s)")
        self.clarification_context.set_plan_steps(plan_result.steps)

        # Emit plan to frontend
        yield {
            'type': 'planning_complete',
            'steps': len(plan_result.steps),
            'plan': [
                {
                    'step_number': s.step_number,
                    'description': s.description,
                    'target_slice_ids': s.target_slice_ids,
                }
                for s in plan_result.steps
            ]
        }

        # Execute each step (each step gets fresh verifier context)
        async for event in self._execute_all_steps(execute_code, {}):
            yield event

    async def _execute_all_steps(
        self,
        execute_code: bool,
        verifier_responses: Dict[str, Dict[str, str]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute all plan steps, one by one.

        Each step goes through: Filter → Match → Verify → Execute.

        Parameters
        ----------
        execute_code : bool
            Whether to execute code
        verifier_responses : Dict[str, Dict[str, str]]
            Previous verifier responses

        Yields
        ------
        Dict[str, Any]
            Events
        """
        all_responses = []
        all_plots = []

        while self.clarification_context.has_more_steps():
            current_step = self.clarification_context.get_current_step()
            step_num, total_steps = self.clarification_context.get_step_progress()

            logger.info("="*60)
            logger.info(f"EXECUTING STEP {step_num}/{total_steps}")
            logger.info(f"Description: {current_step.description}")
            logger.info("="*60)

            yield {
                'type': 'step_start',
                'step_number': step_num,
                'total_steps': total_steps,
                'description': current_step.description
            }

            # Execute this step through pipeline (filter → match → verify)
            # Iterate over generator to forward pipeline detail events in real-time
            pipeline_result = None
            async for event in self.pipeline_executor.execute_single_step(
                step=current_step,
                session_summary=self.session.get_summary() if self.session else {},
                verifier_responses=verifier_responses,
                conversation_history=self._get_conversation_history()
            ):
                if event.get('type') == 'pipeline_result':
                    pipeline_result = event['result']
                else:
                    yield event  # Forward pipeline detail events

            # Handle pipeline result
            async for event in self._handle_step_result(
                pipeline_result, execute_code, step_num, total_steps
            ):
                yield event
                # Collect responses and plots
                if event.get('type') == 'step_execution_complete':
                    all_responses.append(event.get('response', ''))
                    all_plots.extend(event.get('plots', []))

            # If waiting for clarification, pause multi-step execution
            if self.clarification_context.is_waiting_for_clarification():
                logger.info("Pausing multi-step execution - waiting for clarification")
                return

            # Move to next step
            has_more = self.clarification_context.advance_to_next_step()
            if not has_more:
                break

        # All steps complete
        logger.info("="*60)
        logger.info("ALL STEPS COMPLETE")
        logger.info("="*60)

        # Clear context
        self.clear_clarification_fn()

        # Emit completion
        yield {
            "type": "pipeline_complete",
            "total_steps": self.clarification_context.get_step_progress()[1],
            "all_responses": all_responses,
            "plots": all_plots
        }

    async def _handle_step_result(
        self,
        result: PipelineResult,
        execute_code: bool,
        step_number: int,
        total_steps: int
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle pipeline result for a single step.

        Parameters
        ----------
        result : PipelineResult
            Result from pipeline execution
        execute_code : bool
            Whether to execute code
        step_number : int
            Current step number
        total_steps : int
            Total number of steps

        Yields
        ------
        Dict[str, Any]
            Events
        """
        if result.type == 'skill_selection':
            # Store context
            self.clarification_context.store_skill_selection(
                plan_step=result.plan_step,
                skill_options=result.skill_options
            )

            # Build options with names for frontend buttons
            skill_options_with_names = []
            for slug in result.skill_options:
                skill = self.skill_registry.load_full_skill(slug)
                skill_name = skill.name if skill else slug
                skill_options_with_names.append({"slug": slug, "name": skill_name})

            yield {
                "type": "skill_selection",
                "message": f"Step {step_number}/{total_steps}: Multiple skills matched. Please select one:",
                "options": skill_options_with_names,
                "step_number": step_number,
                "total_steps": total_steps
            }

        elif result.type == 'verifier_clarification':
            # Store context
            self.clarification_context.store_verifier_clarification(
                skill_slug=result.clarification_context['skill_slug'],
                plan_step=result.plan_step,
                questions=result.verifier_questions
            )
            yield {
                "type": "prerequisites_needed",
                "questions": result.verifier_questions,
                "skill": result.clarification_context['skill_slug'],
                "step_number": step_number,
                "total_steps": total_steps
            }

        elif result.type == 'advice':
            # Can't proceed with this step
            logger.warning(f"Step {step_number} cannot proceed - needs prior work")
            yield {
                "type": "advice",
                "message": result.advice_message,
                "step_number": step_number,
                "total_steps": total_steps
            }

        elif result.type in ['success', 'no_skill']:
            # Execute this step
            final_query = result.final_query
            matched_skill_slugs = [result.selected_skill.slug] if result.selected_skill else []

            yield {
                'type': 'execution_start',
                'query': final_query,
                'step_number': step_number,
                'total_steps': total_steps
            }

            # Emit warning before execution if no skill matched
            if result.no_skill_matched:
                warning_message = f"No specialized skill found. Using general capabilities."
                yield {'type': 'warning', 'message': warning_message}

            # Execute code
            step_response = None
            step_plots = []
            async for event in self.code_executor(
                final_query,
                execute_code=execute_code,
                allow_planning=False,
                matched_skill_slugs=matched_skill_slugs
            ):
                yield event
                if event['type'] == 'execution_complete':
                    step_response = event['response']
                    step_plots = event.get('plots', [])

            # Clear per-step clarification state
            self.clarification_context.clear_step_state()

            # Emit step completion
            yield {
                "type": "step_execution_complete",
                "step_number": step_number,
                "total_steps": total_steps,
                "response": step_response,
                "plots": step_plots
            }

        else:
            logger.error(f"Unknown pipeline result type: {result.type}")
            yield {"type": "error", "message": f"Unknown result type: {result.type}"}

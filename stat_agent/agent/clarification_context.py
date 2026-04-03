"""
Clarification Context Management.

Manages multi-turn clarification state for the spatial transcriptomics agent.
Tracks three types of clarifications:
1. Planner clarifications - When query planner needs user input about target slices
2. Verifier clarifications - When skill verifier needs prerequisite information
3. Skill selection - When multiple skills match and user needs to choose one
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from .query_planner import PlanStep
from .skill_registry import SkillDefinition

logger = logging.getLogger(__name__)


@dataclass
class PendingClarification:
    """Represents a pending clarification waiting for user response.

    Attributes
    ----------
    type : str
        Type of clarification: 'planner', 'verifier', or 'skill_selection'
    question : str
        The question to ask the user
    context : Dict[str, Any]
        Additional context needed to resume after user responds
    """
    type: str  # 'planner', 'verifier', 'skill_selection'
    question: str
    context: Dict[str, Any] = field(default_factory=dict)


class ClarificationContext:
    """Manages multi-turn clarification state.

    This class isolates all clarification state management logic,
    making it easy to test and reason about multi-turn conversations.

    Three types of clarifications are supported:
    1. **Planner clarifications**: When the query planner needs user input
       (e.g., "Which slice do you want to analyze?")
    2. **Verifier clarifications**: When a skill needs prerequisite information
       (e.g., "What tissue type is this?")
    3. **Skill selection**: When multiple skills match and user must choose
       (e.g., "Do you want to use skill A or B?")
    """

    def __init__(self):
        """Initialize empty clarification context."""
        self._planner_clarifications: List[Tuple[str, str]] = []
        self._verifier_clarifications: Dict[str, List[Tuple[str, str]]] = {}
        self._pending_plan_step: Optional[PlanStep] = None
        self._pending_skill: Optional[str] = None
        self._pending_skill_selection: Optional[Dict[str, Any]] = None
        self._last_planner_question: Optional[str] = None
        self._last_verifier_questions: List[str] = []
        self._original_query: Optional[str] = None

        # Multi-step execution tracking
        self._pending_plan_steps: List[PlanStep] = []
        self._current_step_index: int = 0

    def is_waiting_for_clarification(self) -> bool:
        """Check if there's a pending clarification waiting for user response.

        Returns
        -------
        bool
            True if waiting for clarification, False otherwise
        """
        return (
            self._pending_skill_selection is not None or
            self._pending_plan_step is not None or
            self._last_planner_question is not None
        )

    def get_pending_type(self) -> Optional[str]:
        """Get the type of pending clarification.

        Returns
        -------
        Optional[str]
            'skill_selection', 'verifier', 'planner', or None if no pending clarification
        """
        if self._pending_skill_selection is not None:
            return 'skill_selection'
        elif self._pending_plan_step is not None:
            return 'verifier'
        elif self._last_planner_question is not None:
            return 'planner'
        return None

    # ==================== Planner Clarifications ====================

    def store_planner_clarification(self, question: str, original_query: str) -> None:
        """Store a planner clarification question.

        Parameters
        ----------
        question : str
            The clarification question to ask user
        original_query : str
            The original user query (before clarification)
        """
        logger.debug(f"Storing planner clarification: {question[:100]}...")
        self._last_planner_question = question

        # Store original query so we can use it when user responds
        if not self._original_query:
            self._original_query = original_query
            logger.debug(f"Stored original query: {original_query[:100]}...")

    def handle_planner_response(self, response: str) -> str:
        """Handle user's response to a planner clarification.

        Parameters
        ----------
        response : str
            User's response to the clarification question

        Returns
        -------
        str
            The original query to use for re-planning
        """
        logger.info(f"Received planner clarification: {response[:50]}...")

        # Store the Q&A pair
        question = self._last_planner_question or ""
        self._planner_clarifications.append((question, response))

        # Clear pending planner question - we've handled it.
        # Without this, is_waiting_for_clarification() would still return True,
        # causing the NEXT user query to be misdetected as a clarification response.
        self._last_planner_question = None

        # Return original query for re-planning
        original = self._original_query or ""
        logger.info(f"Using original query for re-planning: {original[:100]}...")
        return original

    def get_planner_history(self) -> List[Tuple[str, str]]:
        """Get history of planner clarifications.

        Returns
        -------
        List[Tuple[str, str]]
            List of (question, answer) tuples
        """
        return self._planner_clarifications.copy()

    def clear_planner_context(self) -> None:
        """Clear planner clarification context after successful planning."""
        self._planner_clarifications = []
        self._original_query = None
        self._last_planner_question = None

    # ==================== Verifier Clarifications ====================

    def store_verifier_clarification(
        self,
        skill_slug: str,
        plan_step: PlanStep,
        questions: List[str]
    ) -> None:
        """Store a verifier clarification request.

        Parameters
        ----------
        skill_slug : str
            The skill requiring verification
        plan_step : PlanStep
            The plan step being verified
        questions : List[str]
            Questions to ask the user
        """
        logger.info(f"Storing verifier clarification for skill: {skill_slug}")
        self._pending_plan_step = plan_step
        self._pending_skill = skill_slug
        self._last_verifier_questions = questions

    def handle_verifier_response(self, response: str) -> None:
        """Handle user's response to a verifier clarification.

        Parameters
        ----------
        response : str
            User's response to the clarification question
        """
        if not self._pending_skill:
            logger.warning("No pending skill for verifier response")
            return

        skill_slug = self._pending_skill

        # Initialize list for this skill if needed
        if skill_slug not in self._verifier_clarifications:
            self._verifier_clarifications[skill_slug] = []

        # Store Q&A pairs
        for question in self._last_verifier_questions:
            self._verifier_clarifications[skill_slug].append((question, response))
            logger.debug(f"Stored verifier clarification for {skill_slug}: {question[:50]}...")

    def get_verifier_responses(self, skill_slug: str) -> Dict[str, str]:
        """Get all verifier responses for a specific skill.

        Parameters
        ----------
        skill_slug : str
            The skill slug to get responses for

        Returns
        -------
        Dict[str, str]
            Dictionary mapping questions to answers
        """
        qa_pairs = self._verifier_clarifications.get(skill_slug, [])
        return dict(qa_pairs)

    def get_pending_plan_step(self) -> Optional[PlanStep]:
        """Get the pending plan step waiting for verification.

        Returns
        -------
        Optional[PlanStep]
            The pending plan step, or None if no verification pending
        """
        return self._pending_plan_step

    def get_pending_skill(self) -> Optional[str]:
        """Get the pending skill slug waiting for verification.

        Returns
        -------
        Optional[str]
            The pending skill slug, or None if no verification pending
        """
        return self._pending_skill

    def get_last_verifier_questions(self) -> List[str]:
        """Get the last verifier questions asked.

        Returns
        -------
        List[str]
            List of questions
        """
        return self._last_verifier_questions.copy()

    # ==================== Skill Selection ====================

    def store_skill_selection(
        self,
        plan_step: PlanStep,
        skill_options: List[str]
    ) -> None:
        """Store a pending skill selection.

        Parameters
        ----------
        plan_step : PlanStep
            The plan step requiring skill selection
        skill_options : List[str]
            List of skill slugs for user to choose from
        """
        logger.info(f"Storing skill selection: {len(skill_options)} options")
        self._pending_skill_selection = {
            'plan_step': plan_step,
            'skill_options': skill_options
        }

    def handle_skill_selection(self, response: str) -> Optional[str]:
        """Handle user's skill selection response.

        Parameters
        ----------
        response : str
            User's selection (number or skill slug)

        Returns
        -------
        Optional[str]
            The selected skill slug, or None if invalid selection
        """
        if not self._pending_skill_selection:
            logger.warning("No pending skill selection")
            return None

        skill_options = self._pending_skill_selection['skill_options']
        user_input = response.strip().lower()

        # Try to match by number (1, 2, 3, ...)
        if user_input.isdigit():
            selection_idx = int(user_input) - 1
            if 0 <= selection_idx < len(skill_options):
                selected = skill_options[selection_idx]
                logger.info(f"User selected skill by number: {selected}")
                return selected

        # Try to match by slug (partial match)
        for slug in skill_options:
            if user_input in slug.lower():
                logger.info(f"User selected skill by name: {slug}")
                return slug

        logger.warning(f"Invalid skill selection: {response}")
        return None

    def get_pending_skill_selection(self) -> Optional[Dict[str, Any]]:
        """Get pending skill selection context.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with 'plan_step' and 'skill_options', or None if no pending selection
        """
        return self._pending_skill_selection

    def clear_skill_selection(self) -> None:
        """Clear pending skill selection after user responds."""
        self._pending_skill_selection = None

    # ==================== Multi-Step Execution Management ====================

    def set_plan_steps(self, steps: List[PlanStep]) -> None:
        """Store all plan steps for multi-step execution.

        Parameters
        ----------
        steps : List[PlanStep]
            All steps from the planner
        """
        self._pending_plan_steps = steps
        self._current_step_index = 0
        logger.info(f"Stored {len(steps)} plan steps for execution")

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed.

        Returns
        -------
        Optional[PlanStep]
            Current step, or None if no steps pending
        """
        if 0 <= self._current_step_index < len(self._pending_plan_steps):
            return self._pending_plan_steps[self._current_step_index]
        return None

    def advance_to_next_step(self) -> bool:
        """Move to the next step in the plan.

        Returns
        -------
        bool
            True if there's a next step, False if all steps complete
        """
        self._current_step_index += 1
        has_more = self._current_step_index < len(self._pending_plan_steps)
        if has_more:
            logger.info(f"Advanced to step {self._current_step_index + 1}/{len(self._pending_plan_steps)}")
        else:
            logger.info("All plan steps completed")
        return has_more

    def get_step_progress(self) -> Tuple[int, int]:
        """Get current step progress.

        Returns
        -------
        Tuple[int, int]
            (current_step_number, total_steps)
        """
        return (self._current_step_index + 1, len(self._pending_plan_steps))

    def has_more_steps(self) -> bool:
        """Check if there are more steps to execute.

        Returns
        -------
        bool
            True if more steps remain
        """
        return self._current_step_index < len(self._pending_plan_steps)

    def clear_step_state(self) -> None:
        """Clear state for the current step (after execution).

        Each step should have its own independent pipeline, so we clear
        both per-step pending state AND verifier clarifications to prevent
        info leakage from step N to step N+1.
        """
        self._pending_plan_step = None
        self._pending_skill = None
        self._pending_skill_selection = None
        # Clear per-step verifier state to prevent cross-step info leakage.
        # Each step should collect its own prerequisites independently.
        self._verifier_clarifications = {}
        self._last_verifier_questions = []

    # ==================== Full Context Management ====================

    def clear_all(self) -> None:
        """Clear all clarification context after successful execution."""
        logger.debug("Clearing all clarification context")
        self._planner_clarifications = []
        self._verifier_clarifications = {}
        self._pending_plan_step = None
        self._pending_skill = None
        self._pending_skill_selection = None
        self._last_planner_question = None
        self._last_verifier_questions = []
        self._original_query = None
        # Clear multi-step state
        self._pending_plan_steps = []
        self._current_step_index = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary (for debugging/logging).

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of current state
        """
        return {
            'planner_clarifications': self._planner_clarifications,
            'verifier_clarifications': self._verifier_clarifications,
            'pending_plan_step': self._pending_plan_step,
            'pending_skill': self._pending_skill,
            'pending_skill_selection': self._pending_skill_selection,
            'last_planner_question': self._last_planner_question,
            'last_verifier_questions': self._last_verifier_questions,
            'original_query': self._original_query,
            'current_step_index': self._current_step_index,
            'total_steps': len(self._pending_plan_steps),
        }

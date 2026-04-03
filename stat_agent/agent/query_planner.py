"""Query planning component for spatial transcriptomics analysis.

The planner determines which slices to analyze and whether clarification is needed.
Uses LLM reasoning with rich session context to handle all edge cases intelligently.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stat_agent.agent.llm_backend import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_number: int
    description: str              # Human-readable description
    target_slice_ids: List[int]   # Which slices this step operates on
    refined_query: str            # Original query refined for this step


@dataclass
class PlanResult:
    """Result from the query planner."""
    # Clarification needed?
    needs_clarification: bool
    clarification_question: Optional[str] = None

    # If plan is ready
    steps: List[PlanStep] = field(default_factory=list)

    # Context
    original_query: str = ""
    session_summary: Dict[str, Any] = field(default_factory=dict)


class QueryPlanner:
    """Plans query execution by determining target slices and breaking into steps.

    This is a stateless component - all context passed as parameters.
    Uses LLM reasoning to handle:
    - Slice inference (explicit IDs, tissue names, "both", "all", etc.)
    - Clarification when ambiguous
    - Multi-step planning (when to split vs single step)
    - Edge cases (ROI references, tissue names, current slice, etc.)

    Examples
    --------
    >>> planner = QueryPlanner(llm_backend)
    >>> result = await planner.plan(
    ...     user_query="Annotate celltype",
    ...     session_summary=session.get_summary()
    ... )
    >>> if result.needs_clarification:
    ...     print(result.clarification_question)
    ... else:
    ...     for step in result.steps:
    ...         print(f"Step {step.step_number}: {step.description}")
    """

    def __init__(self, llm_backend: LLMBackend, prompt_logger=None):
        """Initialize planner with LLM backend.

        Parameters
        ----------
        llm_backend : LLMBackend
            LLM backend for reasoning
        prompt_logger : PromptLogger, optional
            Prompt logger for logging LLM calls
        """
        self.llm = llm_backend
        self.prompt_logger = prompt_logger

    async def plan(
        self,
        user_query: str,
        session_summary: Dict[str, Any],
        previous_clarifications: List[Tuple[str, str]] = None,
        conversation_history: str = ""
    ) -> PlanResult:
        """Plan query execution.

        Parameters
        ----------
        user_query : str
            User's natural language query
        session_summary : Dict[str, Any]
            Session summary from session.get_summary()
            Contains: n_slices, slices (with tissue_name, modality, data_level),
                     rois, modalities, etc.
        previous_clarifications : List[Tuple[str, str]], optional
            Previous clarification Q&A pairs in this conversation
            Format: [(question1, answer1), (question2, answer2), ...]
        conversation_history : str, optional
            Formatted conversation history (assistant + execution summaries)

        Returns
        -------
        PlanResult
            Planning result with steps or clarification question
        """
        if previous_clarifications is None:
            previous_clarifications = []

        # Build rich context for LLM
        planning_prompt = self._build_planning_prompt(
            user_query=user_query,
            session_summary=session_summary,
            previous_clarifications=previous_clarifications,
            conversation_history=conversation_history
        )

        # Get LLM decision
        logger.debug("Calling LLM for query planning")
        import time
        from datetime import datetime

        call_start = time.time()
        response = await self.llm.run(planning_prompt)
        call_duration = time.time() - call_start

        logger.debug(f"Planner LLM response: {response}")

        # Log this LLM call if prompt_logger available
        if self.prompt_logger:
            self.prompt_logger.log_llm_call(
                call_type="query_planner",
                full_prompt=planning_prompt,
                response=response,
                metadata={
                    'model': getattr(self.llm.config, 'model', 'unknown'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': call_duration,
                    'input_tokens': getattr(self.llm, 'last_input_tokens', None),
                    'output_tokens': getattr(self.llm, 'last_output_tokens', None),
                }
            )

        # Parse response
        plan_result = self._parse_planning_response(
            response=response,
            user_query=user_query,
            session_summary=session_summary
        )

        return plan_result

    def _build_planning_prompt(
        self,
        user_query: str,
        session_summary: Dict[str, Any],
        previous_clarifications: List[Tuple[str, str]],
        conversation_history: str = ""
    ) -> str:
        """Build LLM prompt for planning.

        Uses minimal prompt style - trusts LLM reasoning with rich context.
        """
        # Extract session info
        n_slices = session_summary.get('n_slices', 0)
        slices_info = session_summary.get('slices', [])
        rois = session_summary.get('rois', [])

        # Build slice descriptions
        slice_descriptions = []
        for slice_info in slices_info:
            slice_id = slice_info['slice_id']
            tissue = slice_info.get('tissue_name', f'tissue_{slice_id}')
            modality = slice_info.get('modality', 'gene')
            data_level = slice_info.get('data_level', 'cell')
            n_obs = slice_info.get('n_obs', 0)
            n_vars = slice_info.get('n_vars', 0)
            has_celltype = slice_info.get('has_celltype', False)

            desc = (
                f"  - Slice {slice_id}: {tissue} "
                f"({modality}, {data_level}-level, {n_obs:,} observations, {n_vars:,} genes"
            )
            if has_celltype:
                celltypes = slice_info.get('celltypes', [])
                if celltypes:
                    n_types = len(celltypes)
                    desc += f", {n_types} cell types"
            image_names = slice_info.get('image_names', [])
            if image_names:
                desc += f", images: {', '.join(image_names)}"
            desc += ")"
            slice_descriptions.append(desc)

        # Build ROI descriptions
        roi_descriptions = []
        if rois:
            roi_descriptions.append("\nAvailable ROIs:")
            for roi in rois:
                roi_name = roi.get('name', 'unnamed')
                roi_slice = roi.get('slice_id', '?')
                n_cells = roi.get('n_obs', 0)
                roi_descriptions.append(f"  - '{roi_name}' on slice {roi_slice} ({n_cells:,} cells)")

        # Build clarification history
        clarification_context = ""
        if previous_clarifications:
            clarification_context = "\n**Previous Clarifications of Query:**\n"
            for i, (question, answer) in enumerate(previous_clarifications, 1):
                clarification_context += f"{i}. Q: {question}\n   A: {answer}\n"

        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = f"\n**Conversation History (what has been done so far):**\n{conversation_history}\n**[End of Conversation History]**\n"

        # Construct prompt
        prompt = f"""You are a query planner for spatial transcriptomics analysis.

**Session Information:**
- Total slices: {n_slices}
{chr(10).join(slice_descriptions)}
{chr(10).join(roi_descriptions) if roi_descriptions else ""}
{history_context}

**User Query:** "{user_query}"

{clarification_context}

**Your Task:**
Determine which slice(s) the user wants to analyze and how to execute the query.

**Consider:**
- Explicit slice references (e.g., "slice 0", "slice 1")
- Tissue name references (e.g., "breast cancer tissue")
- ROI references (e.g., "in tumor_region")
- Keywords like "both", "all", "each" (may need separate steps)
- Keywords like "compare", "between" (single cross-slice step)
- If only 1 slice exists, assume that slice
- If ambiguous with multiple slices, ASK FOR CLARIFICATION

**Decide:**
1. Do you have enough information to determine target slice(s)?
2. If YES: Which slice(s) and should it be one step or multiple steps?
3. If NO: What clarification question should you ask the user?

**Output Format (JSON):**
```json
{{
  "needs_clarification": true/false,
  "clarification_question": "Your free-text question here" (if needs_clarification),
  "steps": [
    {{
      "step_number": 1,
      "description": "Brief description of what this step does",
      "target_slice_ids": [0],
      "refined_query": "Refined query. Do NOT add information the user did not mention."
    }}
  ] (if not needs_clarification)
}}
```

**Examples:**

Example 1 (Clear - single slice):
User: "Annotate celltype on slice 0"
→ {{"needs_clarification": false, "steps": [{{"step_number": 1, "description": "Annotate cell types on slice 0", "target_slice_ids": [0], "refined_query": "Annotate celltype on slice 0"}}]}}

Example 2 (Ambiguous - multiple slices):
User: "Annotate celltype"
Session: 2 slices
→ {{"needs_clarification": true, "clarification_question": "Which slice would you like to annotate? We have slice 0 (breast cancer) and slice 1 (brain cortex). Or would you like to annotate both?"}}

Example 3 (Both slices - separate steps):
User: "Annotate celltype on both slices"
Session: 2 slices
→ {{"needs_clarification": false, "steps": [{{"step_number": 1, "description": "Annotate cell types on slice 0", "target_slice_ids": [0], "refined_query": "Annotate celltype on slice 0"}}, {{"step_number": 2, "description": "Annotate cell types on slice 1", "target_slice_ids": [1], "refined_query": "Annotate celltype on slice 1"}}]}}

Example 4 (Cross-slice - single step):
User: "Compare gene expression between slice 0 and slice 1"
→ {{"needs_clarification": false, "steps": [{{"step_number": 1, "description": "Compare gene expression between slice 0 and slice 1", "target_slice_ids": [0, 1], "refined_query": "Compare gene expression between slice 0 and slice 1"}}]}}

Example 5 (Tissue name reference):
User: "Perform niche detection in breast cancer tissue"
Session: slice 0 (breast cancer), slice 1 (brain)
→ {{"needs_clarification": false, "steps": [{{"step_number": 1, "description": "Perform niche detection on slice 0 (breast cancer)", "target_slice_ids": [0], "refined_query": "Perform niche detection on slice 0"}}]}}

Example 6 (Only one slice - no ambiguity):
User: "Annotate celltype"
Session: 1 slice only
→ {{"needs_clarification": false, "steps": [{{"step_number": 1, "description": "Annotate cell types on slice 0", "target_slice_ids": [0], "refined_query": "Annotate celltype on slice 0"}}]}}

Now analyze the user's query and output your decision in JSON format:"""

        return prompt

    def _parse_planning_response(
        self,
        response: str,
        user_query: str,
        session_summary: Dict[str, Any]
    ) -> PlanResult:
        """Parse LLM response into PlanResult.

        Handles JSON extraction with multiple fallback strategies.
        """
        # Strategy 1: Try to extract JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Strategy 2: Find any JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Strategy 3: Fallback - assume needs clarification
                logger.warning("Could not extract JSON from planner response, defaulting to clarification")
                return PlanResult(
                    needs_clarification=True,
                    clarification_question="Could you please clarify which slice(s) you want to analyze?",
                    original_query=user_query,
                    session_summary=session_summary
                )

        # Parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from planner: {e}")
            logger.debug(f"JSON string was: {json_str}")
            # Fallback
            return PlanResult(
                needs_clarification=True,
                clarification_question="Could you please clarify which slice(s) you want to analyze?",
                original_query=user_query,
                session_summary=session_summary
            )

        # Extract fields
        needs_clarification = parsed.get('needs_clarification', False)
        clarification_question = parsed.get('clarification_question')
        steps_data = parsed.get('steps', [])

        # Build PlanStep objects
        steps = []
        for step_data in steps_data:
            try:
                step = PlanStep(
                    step_number=step_data.get('step_number', len(steps) + 1),
                    description=step_data.get('description', ''),
                    target_slice_ids=step_data.get('target_slice_ids', []),
                    refined_query=step_data.get('refined_query', user_query)
                )
                steps.append(step)
            except Exception as e:
                logger.error(f"Failed to parse step: {e}")
                logger.debug(f"Step data: {step_data}")
                continue

        # Validate result
        if needs_clarification and not clarification_question:
            logger.warning("LLM indicated clarification needed but didn't provide question")
            clarification_question = "Could you please provide more details about which slice(s) to analyze?"

        if not needs_clarification and not steps:
            logger.warning("LLM didn't indicate clarification but provided no steps")
            # Fallback to clarification
            needs_clarification = True
            clarification_question = "Could you please clarify which slice(s) you want to analyze?"

        # Log result
        if needs_clarification:
            logger.info(f"Planner needs clarification: {clarification_question}")
        else:
            logger.info(f"Planner generated {len(steps)} step(s)")
            for step in steps:
                logger.info(f"  Step {step.step_number}: {step.description} (slices: {step.target_slice_ids})")

        return PlanResult(
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            steps=steps,
            original_query=user_query,
            session_summary=session_summary
        )

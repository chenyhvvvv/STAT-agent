"""Skill prerequisite verification component.

Checks if skill prerequisites are met and handles missing requirements.
Uses LLM reasoning to determine what can be obtained via chat vs needs prior work.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stat_agent.agent.llm_backend import LLMBackend
    from stat_agent.agent.query_planner import PlanStep
    from stat_agent.agent.skill_registry import SkillDefinition

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result from prerequisite verification."""
    # Are all prerequisites met?
    prerequisites_met: bool

    # Missing prerequisites
    missing_prerequisites: List[str] = field(default_factory=list)

    # Can we obtain missing prerequisites by asking user?
    can_obtain_by_chat: bool = False

    # Questions to ask user (if can_obtain_by_chat)
    clarification_questions: List[str] = field(default_factory=list)

    # Complete query with all info (if prerequisites_met or after collecting info)
    complete_query: Optional[str] = None

    # Advice if prerequisites can't be met by chat
    advice: Optional[str] = None

    # Collected user responses (for building complete query)
    collected_info: Dict[str, str] = field(default_factory=dict)


class SkillVerifier:
    """Verifies skill prerequisites and handles missing requirements.

    Uses LLM reasoning with simple prerequisite descriptions and session context
    to determine:
    1. What prerequisites are missing
    2. Whether they can be obtained by asking the user
    3. What questions to ask or what advice to give

    Examples
    --------
    >>> verifier = SkillVerifier(llm_backend)
    >>> result = await verifier.verify(
    ...     plan_step=step,
    ...     selected_skill=skill,
    ...     session_summary=session.get_summary()
    ... )
    >>> if not result.prerequisites_met:
    ...     if result.can_obtain_by_chat:
    ...         ask_user(result.clarification_questions)
    ...     else:
    ...         show_advice(result.advice)
    """

    def __init__(self, llm_backend: LLMBackend, prompt_logger=None):
        """Initialize verifier with LLM backend.

        Parameters
        ----------
        llm_backend : LLMBackend
            LLM backend for reasoning
        prompt_logger : PromptLogger, optional
            Prompt logger for logging LLM calls
        """
        self.llm = llm_backend
        self.prompt_logger = prompt_logger

    async def verify(
        self,
        plan_step: PlanStep,
        selected_skill: Optional[SkillDefinition],
        session_summary: Dict[str, Any],
        user_responses: Dict[str, str] = None,
        conversation_history: str = ""
    ) -> VerificationResult:
        """Verify prerequisites for a skill.

        Parameters
        ----------
        plan_step : PlanStep
            Plan step to execute (contains target_slice_ids, refined_query)
        selected_skill : Optional[SkillDefinition]
            Selected skill (None if no skill matched)
        session_summary : Dict[str, Any]
            Session summary with slice info, ROIs, etc.
        user_responses : Dict[str, str], optional
            User responses to previous clarification questions
            Format: {question: answer, ...}
        conversation_history : str, optional
            Formatted conversation history (assistant + execution summaries)

        Returns
        -------
        VerificationResult
            Verification result with questions or complete query
        """
        if user_responses is None:
            user_responses = {}

        # If no skill selected, no prerequisites to check
        if not selected_skill:
            logger.info("No skill selected, skipping prerequisite verification")
            return VerificationResult(
                prerequisites_met=True,
                complete_query=plan_step.refined_query
            )

        # If skill has no prerequisites, we're good
        if not selected_skill.prerequisites:
            logger.info(f"Skill '{selected_skill.slug}' has no prerequisites")
            return VerificationResult(
                prerequisites_met=True,
                complete_query=plan_step.refined_query
            )

        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            plan_step=plan_step,
            selected_skill=selected_skill,
            session_summary=session_summary,
            user_responses=user_responses,
            conversation_history=conversation_history
        )

        # Get LLM decision
        logger.debug("Calling LLM for prerequisite verification")
        import time
        from datetime import datetime

        call_start = time.time()
        response = await self.llm.run(verification_prompt)
        call_duration = time.time() - call_start

        logger.debug(f"Verifier LLM response: {response}")

        # Log this LLM call if prompt_logger available
        if self.prompt_logger:
            self.prompt_logger.log_llm_call(
                call_type="skill_verifier",
                full_prompt=verification_prompt,
                response=response,
                metadata={
                    'model': getattr(self.llm.config, 'model', 'unknown'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': call_duration,
                    'input_tokens': self.llm.last_usage.input_tokens if self.llm.last_usage else None,
                    'output_tokens': self.llm.last_usage.output_tokens if self.llm.last_usage else None,
                }
            )

        # Parse response
        verification_result = self._parse_verification_response(
            response=response,
            plan_step=plan_step,
            selected_skill=selected_skill,
            user_responses=user_responses
        )

        return verification_result

    def _build_verification_prompt(
        self,
        plan_step: PlanStep,
        selected_skill: SkillDefinition,
        session_summary: Dict[str, Any],
        user_responses: Dict[str, str],
        conversation_history: str = ""
    ) -> str:
        """Build LLM prompt for prerequisite verification."""

        # Get target slice information
        target_slices = plan_step.target_slice_ids
        slices_info = session_summary.get('slices', [])

        slice_descriptions = []
        for slice_id in target_slices:
            # Find slice info
            slice_info = next((s for s in slices_info if s['slice_id'] == slice_id), None)
            if not slice_info:
                continue

            tissue = slice_info.get('tissue_name', f'tissue_{slice_id}')
            modality = slice_info.get('modality', 'gene')
            data_level = slice_info.get('data_level', 'cell')
            n_obs = slice_info.get('n_obs', 0)
            has_celltype = slice_info.get('has_celltype', False)

            desc = f"  - Slice {slice_id}: {tissue} ({modality}, {data_level}-level, {n_obs:,} observations)"

            if has_celltype:
                celltypes = slice_info.get('celltypes', [])
                if celltypes:
                    desc += f", has celltype annotations ({len(celltypes)} types)"
            else:
                desc += ", NO celltype annotations"

            image_names = slice_info.get('image_names', [])
            if image_names:
                desc += f", images: {', '.join(image_names)}"

            # Check for available columns
            columns = slice_info.get('columns', [])
            if columns:
                desc += f"\n    Available columns: {', '.join(columns[:10])}"
                if len(columns) > 10:
                    desc += f" ... ({len(columns)} total)"

            slice_descriptions.append(desc)

        # Build ROI info for target slices
        roi_descriptions = []
        rois = session_summary.get('rois', [])
        relevant_rois = [r for r in rois if r.get('slice_id') in target_slices]
        if relevant_rois:
            roi_descriptions.append("\nAvailable ROIs on target slices:")
            for roi in relevant_rois:
                roi_name = roi.get('name', 'unnamed')
                roi_slice = roi.get('slice_id', '?')
                n_cells = roi.get('n_obs', 0)
                roi_descriptions.append(f"  - '{roi_name}' on slice {roi_slice} ({n_cells:,} cells)")

        # Build user responses context
        user_responses_context = ""
        if user_responses:
            user_responses_context = "\n**Previously Collected Information:**\n"
            for question, answer in user_responses.items():
                user_responses_context += f"  Q: {question}\n  A: {answer}\n"

        # Build prerequisites list
        prerequisites_list = "\n".join(f"  {i}. {prereq}" for i, prereq in enumerate(selected_skill.prerequisites, 1))

        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = f"\n**Conversation History (what has been done so far):**\n{conversation_history}\n"

        # Construct prompt
        prompt = f"""You are a skill prerequisite verifier for spatial transcriptomics analysis.

**Task:** Verify if all prerequisites are met for executing a skill.

**Plan Step:**
- Description: {plan_step.description}
- Query: "{plan_step.refined_query}"
- Target slices: {target_slices}

**Selected Skill:** {selected_skill.name} ({selected_skill.slug})

**Skill Prerequisites:**
{prerequisites_list}

**Current Session State:**
- Total slices: {session_summary.get('n_slices', 0)}
{chr(10).join(slice_descriptions)}
{chr(10).join(roi_descriptions) if roi_descriptions else ""}
{user_responses_context}
{history_context}

**Your Task:**
Analyze whether all prerequisites are met based on:
1. Current session state (what data/columns exist)
2. Information already collected from user
3. What's missing and how to obtain it

**Decision Criteria:**
- Check ONLY the listed Skill Prerequisites above against the current session state and previously collected info. Do NOT invent additional prerequisites.
- If every listed prerequisite is already in session (e.g., celltype column exists), or can be obtained from the session (e.g. tissue name), or has been provided in the previously collected info → MET. Mark MET even if the user query also describes in-line data preparation (subsetting, filtering, relabeling, pooling, building a TME object, dropping cells, renaming categories) — the executor will do that preparation in code at runtime; it is NOT a missing prerequisite.
- If a listed prerequisite needs simple user input (e.g., file path, species) and has not been collected → ASK USER
- If a listed prerequisite cannot be derived in code from current state and genuinely requires a separate prior analytical step (e.g., the slice has no celltype column at all and the skill needs one, while query not mentioned; deconvolution weights don't exist and the skill needs them) → DECLINE USER with advice on what to do when NECESSARY!
- If can pass or can be obtained by chat, do not provide advice. Do not easily to decline the query. Only provide advice if prerequisites can't be met by chat and current conditions and user needs guidance on what to do next.

**Output Format (JSON):**
```json
{{
  "prerequisites_met": true/false,
  "missing_prerequisites": ["List of missing items"],
  "can_obtain_by_chat": true/false,
  "clarification_questions": ["Question 1", "Question 2"],
  "complete_query": "Complete query with all info" (if all met or all collected),
  "advice": "Advice for user" (if can't obtain by chat)
}}
```

**Examples:**

Example 1 (All met):
Prerequisites: ["Cell type annotations in target slice (adata.obs['celltype'])"]
Session: Slice 0 has celltype annotations
→ {{"prerequisites_met": true, "complete_query": "Perform niche detection on slice 0"}}

Example 2 (Can ask user):
Prerequisites: ["Tissue type information", "Annotated reference dataset path (.h5ad file)"]
Session: No info provided yet
→ {{
  "prerequisites_met": false,
  "missing_prerequisites": ["Tissue type", "Reference dataset path"],
  "can_obtain_by_chat": true,
  "clarification_questions": [
    "What type of tissue is this? (e.g., breast cancer, brain, liver)",
    "Please provide the full path to your annotated reference dataset (.h5ad file)"
  ]
}}

Example 3 (Needs prior work):
Prerequisites: ["Cell type annotations in target slice"]
Session: Slice 0 has NO celltype annotations
→ {{
  "prerequisites_met": false,
  "missing_prerequisites": ["Cell type annotations"],
  "can_obtain_by_chat": false,
  "advice": "Niche detection requires cell type annotations in slice 0. Please first annotate cell types using one of the annotation methods (e.g., 'Annotate cell types in slice 0'), then retry niche detection."
}}

Example 4 (Partial info collected):
Prerequisites: ["Reference path", "Celltype column name"]
User already provided: "Reference path: /path/to/ref.h5ad"
Session: Still missing celltype column name
→ {{
  "prerequisites_met": false,
  "missing_prerequisites": ["Celltype column name"],
  "can_obtain_by_chat": true,
  "clarification_questions": ["Which column in the reference dataset contains cell type labels? (default: 'celltype')"]
}}

Example 5 (All info collected):
Prerequisites: ["Tissue type", "Reference path"]
User provided: "Tissue: breast cancer", "Reference: /path/to/ref.h5ad"
→ {{
  "prerequisites_met": true,
  "complete_query": "Annotate cell types in slice 0 using reference dataset at /path/to/ref.h5ad for breast cancer tissue"
}}

Now analyze the current situation and output your decision in JSON format:"""

        return prompt

    def _parse_verification_response(
        self,
        response: str,
        plan_step: PlanStep,
        selected_skill: SkillDefinition,
        user_responses: Dict[str, str]
    ) -> VerificationResult:
        """Parse LLM response into VerificationResult."""

        # Extract JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback - assume prerequisites not met, need clarification
                logger.warning("Could not extract JSON from verifier response")
                return VerificationResult(
                    prerequisites_met=False,
                    missing_prerequisites=selected_skill.prerequisites,
                    can_obtain_by_chat=True,
                    clarification_questions=["Please provide the information needed for this analysis."]
                )

        # Parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from verifier: {e}")
            return VerificationResult(
                prerequisites_met=False,
                missing_prerequisites=selected_skill.prerequisites,
                can_obtain_by_chat=True,
                clarification_questions=["Please provide the information needed for this analysis."]
            )

        # Extract fields
        prerequisites_met = parsed.get('prerequisites_met', False)
        missing_prerequisites = parsed.get('missing_prerequisites', [])
        can_obtain_by_chat = parsed.get('can_obtain_by_chat', False)
        clarification_questions = parsed.get('clarification_questions', [])
        complete_query = parsed.get('complete_query')
        advice = parsed.get('advice')

        # Log result
        if prerequisites_met:
            logger.info(f"Prerequisites met for skill '{selected_skill.slug}'")
            logger.info(f"Complete query: {complete_query}")
        elif can_obtain_by_chat:
            logger.info(f"Prerequisites missing but can ask user ({len(clarification_questions)} questions)")
            for i, q in enumerate(clarification_questions, 1):
                logger.info(f"  Question {i}: {q}")
        else:
            logger.info(f"Prerequisites missing, need prior work")
            logger.info(f"Advice: {advice}")

        return VerificationResult(
            prerequisites_met=prerequisites_met,
            missing_prerequisites=missing_prerequisites,
            can_obtain_by_chat=can_obtain_by_chat,
            clarification_questions=clarification_questions,
            complete_query=complete_query,
            advice=advice,
            collected_info=user_responses.copy()
        )

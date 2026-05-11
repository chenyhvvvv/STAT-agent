"""
Memory system for spatial transcriptomics agent.

Manages conversation history, context, and session state to enable
coherent multi-turn interactions and context-aware responses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation with summarization support."""

    role: str  # "user", "assistant", "system"
    content: str  # Full content (always preserved)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW: Summarization fields
    summary: Optional[str] = None  # Concise summary if content is long
    is_summarized: bool = False  # Whether summary has been generated

    # NEW: Execution context (for assistant messages)
    code_executed: Optional[str] = None  # Code that was executed
    execution_output: Optional[str] = None  # stdout/stderr from execution
    execution_summary: Optional[str] = None  # Summary of what code did
    query: Optional[str] = None  # The refined query for this step (for multi-step plans)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create from dictionary."""
        # Handle old messages without new fields
        if 'summary' not in data:
            data['summary'] = None
        if 'is_summarized' not in data:
            data['is_summarized'] = False
        if 'code_executed' not in data:
            data['code_executed'] = None
        if 'execution_output' not in data:
            data['execution_output'] = None
        if 'execution_summary' not in data:
            data['execution_summary'] = None
        if 'query' not in data:
            data['query'] = None
        return cls(**data)


@dataclass
class ConversationTurn:
    """Represents a complete conversation turn (user query + assistant response)."""

    user_message: str
    assistant_message: str
    code_generated: Optional[str] = None
    execution_result: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationTurn:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DataModification:
    """Track explicit modifications to session data.

    This makes it impossible for the agent to forget what was added/changed.
    """

    modification_type: str  # "column_added", "column_removed", "column_updated", "roi_created", etc.
    target: str  # What was modified (e.g., "slice_0.adata.obs", "session.get_slice(0).adata.obs", "session.dataset.slices[0].adata.obs")
    details: Dict[str, Any]  # Specific details about the modification
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_turn: int = 0  # Which conversation turn caused this

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataModification:
        """Create from dictionary."""
        return cls(**data)

    def get_description(self) -> str:
        """Get human-readable description of this modification."""
        if self.modification_type == "column_added":
            col_name = self.details.get("column_name", "unknown")
            n_values = self.details.get("n_values")
            desc = f"Added column `{col_name}` to `{self.target}`"
            if n_values is not None:
                desc += f" ({n_values} unique values)"
            return desc

        elif self.modification_type == "column_updated":
            col_name = self.details.get("column_name", "unknown")
            method = self.details.get("method", "")
            desc = f"Updated `{col_name}` in `{self.target}`"
            if method:
                desc += f" using {method}"
            return desc

        elif self.modification_type == "roi_created":
            roi_name = self.details.get("roi_name", "unknown")
            n_cells = self.details.get("n_cells", 0)
            return f"Created ROI `{roi_name}` ({n_cells:,} cells)"

        elif self.modification_type == "roi_deleted":
            roi_name = self.details.get("roi_name", "unknown")
            return f"Deleted ROI `{roi_name}`"

        else:
            return f"{self.modification_type}: {self.target}"


class DataModificationHistory:
    """History of all data modifications in the session."""

    def __init__(self):
        self.modifications: List[DataModification] = []

    def add_modification(self, mod: DataModification) -> None:
        """Add a modification to history."""
        self.modifications.append(mod)
        logger.debug(f"Tracked data modification: {mod.get_description()}")

    def get_recent_modifications(self, n: int = 5) -> List[DataModification]:
        """Get N most recent modifications."""
        return self.modifications[-n:] if self.modifications else []

    def get_modifications_by_type(self, mod_type: str) -> List[DataModification]:
        """Get all modifications of specific type."""
        return [m for m in self.modifications if m.modification_type == mod_type]

    def get_context_string(self, n_recent: int = 5) -> str:
        """
        Format recent modifications for prompt injection.

        This ensures the agent knows exactly what columns/data exist.
        """
        recent = self.get_recent_modifications(n_recent)
        if not recent:
            return ""

        lines = ["**Recent Data Modifications:**"]
        for mod in recent:
            lines.append(f"- {mod.get_description()}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "modifications": [m.to_dict() for m in self.modifications]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataModificationHistory:
        """Create from dictionary."""
        history = cls()
        if "modifications" in data:
            history.modifications = [
                DataModification.from_dict(m) for m in data["modifications"]
            ]
        return history


@dataclass
class ConversationSummary:
    """Summary of a range of conversation turns."""

    start_turn: int  # First turn included (0-indexed)
    end_turn: int  # Last turn included (0-indexed)
    summary: str  # Natural language summary
    key_actions: List[str] = field(default_factory=list)  # Bullet points of key actions
    data_changes: List[str] = field(default_factory=list)  # Data modifications made
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationSummary:
        """Create from dictionary."""
        return cls(**data)


class ConversationMemory:
    """
    Manages conversation history with intelligent context windowing and summarization.

    Features:
    - Stores full conversation history
    - Smart summarization for long messages and old conversations
    - Explicit data modification tracking
    - Provides windowed context for LLM (last N messages)
    - Persists to disk for session recovery
    - Tracks session metadata (data loaded, ROIs, etc.)

    Parameters
    ----------
    max_full_messages : int
        Maximum messages to show in full (default: 8 = 4 turns)
    message_summary_threshold : int
        Summarize individual messages longer than this (default: 200 chars)
    max_context_messages : int
        Trigger conversation-level summarization after this many messages (default: 20)
    storage_dir : Optional[Path]
        Directory to store conversation history
    session_id : Optional[str]
        Unique session identifier
    llm_backend : Optional[LLMBackend]
        LLM backend for generating summaries

    Examples
    --------
    >>> memory = ConversationMemory(max_full_messages=8, message_summary_threshold=200)
    >>> memory.add_user_message("Show me cell types")
    >>> memory.add_assistant_message_with_summary("Here are the cell types...", ...)
    >>> context = memory.get_context_for_llm()
    """

    def __init__(
        self,
        max_full_messages: int = 80,  # Show last 80 messages (~40 turns) in full — sized for 200K-context backbones
        message_summary_threshold: int = 200,  # Summarize individual message bodies > 200 chars
        max_context_messages: int = 200,  # Trigger conversation summary after 200 messages
        storage_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        llm_backend: Optional['LLMBackend'] = None  # Will be set by agent
    ):
        # Configuration
        self.max_full_messages = max_full_messages
        self.message_summary_threshold = message_summary_threshold
        self.max_context_messages = max_context_messages

        # Auto-migrate old config directory
        old_dir = Path.home() / ".spatiallab"
        new_dir = Path.home() / ".stat_agent"
        if old_dir.exists() and not new_dir.exists():
            import shutil
            shutil.copytree(old_dir, new_dir)
            logger.info(f"Migrated config directory from {old_dir} to {new_dir}")

        self.storage_dir = storage_dir or Path.home() / ".stat_agent" / "conversations"
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llm_backend = llm_backend  # For generating summaries

        # Storage
        self.messages: List[Message] = []
        self.turns: List[ConversationTurn] = []
        self.session_metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "data_loaded": False,
            "n_cells": 0,
            "n_genes": 0,
            "rois": [],
        }

        # Semantic entity tracking for intelligent context
        self.entities: Dict[str, Any] = {
            'rois': {},        # {name: {created_at, cell_count, analyses_run}}
            'genes': set(),    # {gene_names_queried}
            'analyses': [],    # [{type, params, timestamp, result_summary}]
            'findings': []     # [{finding, confidence, timestamp}]
        }

        # NEW: Data modification tracking
        self.data_modifications = DataModificationHistory()

        # NEW: Conversation summaries
        self.conversation_summaries: List[ConversationSummary] = []

        # Context summary for older messages (will be auto-generated)
        self.context_summary: Optional[str] = None

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized conversation memory (session: {self.session_id}, "
            f"max_full_messages: {max_full_messages}, "
            f"message_summary_threshold: {message_summary_threshold})"
        )

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a user message to the conversation."""
        message = Message(
            role="user",
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        logger.debug(f"Added user message: {content[:50]}...")

    def add_assistant_message(
        self,
        content: str,
        code: Optional[str] = None,
        execution_result: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add an assistant message to the conversation.

        This is the backward-compatible method. For new code with summarization,
        use add_assistant_message_with_summary instead.
        """
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
            code_executed=code,
            execution_output=execution_result
        )
        self.messages.append(message)

        # Create conversation turn if we have a recent user message
        if len(self.messages) >= 2 and self.messages[-2].role == "user":
            turn = ConversationTurn(
                user_message=self.messages[-2].content,
                assistant_message=content,
                code_generated=code,
                execution_result=execution_result,
                metadata=metadata or {}
            )
            self.turns.append(turn)

        logger.debug(f"Added assistant message: {content[:50]}...")

    def add_assistant_message_with_summary(
        self,
        content: str,
        code: Optional[str] = None,
        execution_result: Optional[str] = None,
        summary: Optional[str] = None,
        execution_summary: Optional[str] = None,
        metadata: Optional[Dict] = None,
        query: Optional[str] = None
    ) -> None:
        """
        Add an assistant message with optional summary and execution details.

        This is the new method that supports smart summarization.

        Parameters
        ----------
        content : str
            Full message content (always preserved)
        code : Optional[str]
            Code that was executed
        execution_result : Optional[str]
            stdout/stderr from code execution
        summary : Optional[str]
            Concise summary of the message (for long messages)
        execution_summary : Optional[str]
            Summary of what the code did (e.g., "Added niche_label column")
        metadata : Optional[Dict]
            Additional metadata
        query : Optional[str]
            The refined query for this step (for multi-step plans)
        """
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
            code_executed=code,
            execution_output=execution_result,
            summary=summary,
            execution_summary=execution_summary,
            is_summarized=(summary is not None),
            query=query
        )
        self.messages.append(message)

        # Create conversation turn if we have a recent user message
        if len(self.messages) >= 2 and self.messages[-2].role == "user":
            turn = ConversationTurn(
                user_message=self.messages[-2].content,
                assistant_message=content,
                code_generated=code,
                execution_result=execution_result,
                metadata=metadata or {}
            )
            self.turns.append(turn)

        logger.debug(
            f"Added assistant message with summary: {content[:50]}... "
            f"(summarized: {summary is not None})"
        )

    def add_system_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a system message to the conversation."""
        message = Message(
            role="system",
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        logger.debug(f"Added system message: {content[:50]}...")

    def get_context_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get recent messages for LLM context (backward compatibility).

        For new code, use get_context_for_llm() instead which has smarter summarization.

        Returns messages in OpenAI chat format: {"role": "...", "content": "..."}
        """
        # Get last N messages
        recent_messages = self.messages[-self.max_context_messages:]

        # Convert to chat format
        context = []

        # Add context summary if we have older messages
        if self.context_summary and len(self.messages) > self.max_context_messages:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.context_summary}"
            })

        # Add recent messages
        for msg in recent_messages:
            if msg.role == "system" and not include_system:
                continue
            context.append({
                "role": msg.role,
                "content": msg.content
            })

        return context

    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """
        Build context for LLM with smart summarization.

        Logic:
        1. If <= max_full_messages (8): Show all messages (possibly with individual summaries)
        2. If > max_full_messages:
           a. Summarize old messages (beyond last 8) into conversation summary
           b. Show last max_full_messages with smart content selection
        3. For each message shown:
           - User messages: Always full content (they're usually short)
           - Assistant messages: Use summary if long and summary exists, otherwise full
           - Execution summaries: Always show if available

        Returns:
            List of message dicts in OpenAI format
        """
        context_messages = []
        n_messages = len(self.messages)

        # Case 1: Few messages - show all (skip user messages, assistant already captures intent)
        if n_messages <= self.max_full_messages:
            for msg in self.messages:
                if msg.role == "user":
                    continue
                content = self._get_message_content_for_context(msg)
                context_messages.append({
                    "role": msg.role,
                    "content": content
                })

        # Case 2: Many messages - summarize old, show recent
        else:
            # Generate conversation summary for old messages if not cached
            if self.context_summary is None:
                old_messages = self.messages[:-self.max_full_messages]
                self.context_summary = self._create_conversation_summary(old_messages)

            # Add conversation summary as system message
            if self.context_summary:
                context_messages.append({
                    "role": "system",
                    "content": self.context_summary
                })

            # Show recent messages with smart content selection (skip user messages)
            recent_messages = self.messages[-self.max_full_messages:]
            for msg in recent_messages:
                if msg.role == "user":
                    continue
                content = self._get_message_content_for_context(msg)
                context_messages.append({
                    "role": msg.role,
                    "content": content
                })

        return context_messages

    def _get_message_content_for_context(self, msg: Message) -> str:
        """
        Get appropriate content for a message based on length and role.

        Only called for assistant and system messages (user messages are skipped).
        """
        # System messages: Always show full
        if msg.role == "system":
            return msg.content

        # Assistant messages: Use smart selection
        if msg.role == "assistant":
            parts = []

            # Add query if present (for multi-step plans)
            if msg.query:
                parts.append(f"**Query:** {msg.query}")

            # If we have a summary and message is long, use summary
            if msg.is_summarized and msg.summary and len(msg.content) > self.message_summary_threshold:
                parts.append(f"**Assistant:** {msg.summary}")

                # Always append execution summary if available (critical information)
                if msg.execution_summary:
                    parts.append(f"**Execution:** {msg.execution_summary}")

                return "\n\n".join(parts)
            else:
                # Short message or no summary - show full content
                parts.append(f"**Assistant:** {msg.content}")

                # Still append execution summary if available
                if msg.execution_summary:
                    parts.append(f"**Execution:** {msg.execution_summary}")

                return "\n\n".join(parts)

        # Fallback
        return msg.content

    def _create_conversation_summary(self, messages: List[Message]) -> str:
        """
        Create a conversation-level summary for old messages.

        Only uses assistant summaries and execution summaries (user messages are
        redundant since assistant responses already capture user intent).
        """
        assistant_summaries = []
        key_actions = []

        for msg in messages:
            if msg.role != "assistant":
                continue
            # Collect assistant summary or truncated content
            if msg.summary:
                assistant_summaries.append(msg.summary)
            elif msg.content:
                text = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                assistant_summaries.append(text)
            # Collect execution summaries
            if msg.execution_summary:
                key_actions.append(msg.execution_summary)

        summary_parts = ["**Previous Conversation Summary:**"]

        if assistant_summaries:
            summary_parts.append("\n**Actions Performed:**")
            for i, summary in enumerate(assistant_summaries[:50], 1):
                summary_parts.append(f"{i}. {summary}")

        if key_actions:
            summary_parts.append("\n**Key Data Changes:**")
            for action in key_actions:
                summary_parts.append(f"- {action}")

        return "\n".join(summary_parts)

    async def create_message_summary(self, content: str) -> Optional[str]:
        """
        Create a concise summary of a long assistant message using LLM.

        Args:
            content: Full message content to summarize

        Returns:
            Concise summary (or None if LLM backend not available)
        """
        if not self.llm_backend:
            logger.warning("LLM backend not available for summarization")
            return None

        # Carefully designed prompt for message summarization
        prompt = f"""Summarize the following assistant message into a single concise sentence.

REQUIREMENTS:
- One sentence only (max 100 characters)
- Focus on WHAT was done, not HOW
- Use past tense ("Performed X", "Created Y", "Analyzed Z")
- Be specific about the main action
- NO explanations or details

EXAMPLES:
Input: "I'll perform niche detection using the Harmonics hierarchical model. This model analyzes spatial neighborhoods..."
Output: "Performed niche detection using Harmonics model"

Input: "Let me create a spatial plot showing cell type distributions across the tissue..."
Output: "Created spatial plot of cell type distributions"

Input: "I'll analyze differential expression between tumor and normal regions. First I'll subset the data..."
Output: "Analyzed differential expression between tumor and normal regions"

MESSAGE TO SUMMARIZE:
{content[:500]}

SUMMARY (one sentence, max 100 chars):"""

        try:
            summary = await self.llm_backend.run(prompt)
            summary = summary.strip()

            # Validate summary is concise
            if len(summary) > 150:  # Allow some buffer
                summary = summary[:100].rsplit(' ', 1)[0]  # Truncate at word boundary

            logger.debug(f"Generated message summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate message summary: {e}")
            return None

    async def create_execution_summary(
        self,
        user_query: str,
        code: str,
        execution_output: str
    ) -> Optional[str]:
        """
        Create a summary of what code execution accomplished using LLM.

        This is CRITICAL for the agent to remember what data modifications were made.

        Args:
            user_query: Original user query
            code: Code that was executed
            execution_output: stdout/stderr from execution

        Returns:
            Execution summary (e.g., "Added niche_label column; 7 niches detected")
        """
        if not self.llm_backend:
            logger.warning("LLM backend not available for execution summarization")
            return None

        # Carefully designed prompt for execution summarization
        prompt = f"""Summarize what this code execution accomplished in ONE short sentence.

CRITICAL REQUIREMENTS:
- ONE sentence only (max 80 characters)
- Focus ONLY on data modifications and key results
- Mention column names if columns were added/modified
- Mention counts/numbers if relevant
- Use semicolon to separate multiple actions
- NO code details, NO explanations

EXAMPLES:

Example 1:
User query: "Perform niche detection"
Code: slice_0 = session.get_slice(0); slice_0.adata.obs['niche_label'] = harmonics_model.fit_predict(...)
Output: "Found 7 spatial niches"
Summary: "Added niche_label column; 7 niches detected"

Example 2:
User query: "Annotate cell types"
Code: slice_0 = session.get_slice(0); slice_0.adata.obs['celltype'] = model.predict(...)
Output: "Annotated 50000 cells into 12 types"
Summary: "Added celltype column; 12 cell types"

Example 3:
User query: "Create ROI in tumor region"
Code: session.create_roi('tumor_roi', ...)
Output: "Created ROI with 3686 cells"
Summary: "Created ROI 'tumor_roi' (3686 cells)"

Example 4:
User query: "Plot cell types spatially"
Code: plt.scatter(...); plt.savefig(...)
Output: "Saved plot to output.png"
Summary: "Generated spatial cell type plot"

Now summarize this execution:

USER QUERY: {user_query}

CODE (first 300 chars):
{code[:300]}

EXECUTION OUTPUT (first 200 chars):
{execution_output[:200]}

EXECUTION SUMMARY (one sentence, max 80 chars, focus on data modifications):"""

        try:
            summary = await self.llm_backend.run(prompt)
            summary = summary.strip()

            # Validate summary is concise
            if len(summary) > 100:  # Allow some buffer
                summary = summary[:80].rsplit(' ', 1)[0]  # Truncate at word boundary

            logger.debug(f"Generated execution summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate execution summary: {e}")
            return None

    def update_session_metadata(self, **kwargs) -> None:
        """Update session metadata (data info, ROIs, etc.)."""
        self.session_metadata.update(kwargs)
        logger.debug(f"Updated session metadata: {kwargs}")

    def get_session_context_string(self) -> str:
        """Get a formatted string of current session context."""
        lines = [
            f"Session ID: {self.session_metadata['session_id']}",
            f"Created: {self.session_metadata['created_at']}",
            "",
            "Current Data:"
        ]

        if self.session_metadata.get("data_loaded"):
            lines.append(f"  • Cells: {self.session_metadata.get('n_cells', 0):,}")
            lines.append(f"  • Genes: {self.session_metadata.get('n_genes', 0):,}")

            if self.session_metadata.get("celltypes"):
                celltypes = self.session_metadata.get("celltypes", [])[:5]
                lines.append(f"  • Cell types: {', '.join(celltypes)}")

            if self.session_metadata.get("rois"):
                rois = self.session_metadata.get("rois", [])
                lines.append(f"  • ROIs: {len(rois)} defined")
        else:
            lines.append("  • No data loaded yet")

        lines.append("")
        lines.append(f"Conversation turns: {len(self.turns)}")

        return "\n".join(lines)

    # ========== NEW: Entity Tracking Methods ==========

    def track_roi_created(self, roi_name: str, metadata: Optional[Dict] = None) -> None:
        """
        Track when a ROI is created.

        Args:
            roi_name: Name of the ROI
            metadata: Optional metadata (n_cells, bounds, slice_id, modality, etc.)
        """
        roi_data = {
            'created_at': datetime.now().isoformat(),
            'cell_count': metadata.get('n_cells', 0) if metadata else 0,
            'analyses_run': []
        }

        # CRITICAL: Store slice_id and modality for multi-slice/multi-omics data
        if metadata:
            if 'slice_id' in metadata:
                roi_data['slice_id'] = metadata['slice_id']
            if 'modality' in metadata:
                roi_data['modality'] = metadata['modality']

        self.entities['rois'][roi_name] = roi_data
        logger.debug(f"Tracked ROI creation: {roi_name} (slice_id={roi_data.get('slice_id')}, modality={roi_data.get('modality')})")

    def track_gene_queried(self, gene_name: str) -> None:
        """Track when a gene is queried."""
        self.entities['genes'].add(gene_name)
        logger.debug(f"Tracked gene query: {gene_name}")

    def track_analysis(
        self,
        analysis_type: str,
        params: Optional[Dict] = None,
        result_summary: Optional[str] = None
    ) -> None:
        """
        Track an analysis that was performed.

        Args:
            analysis_type: Type of analysis (e.g., "DE", "clustering", "spatial")
            params: Analysis parameters
            result_summary: Brief summary of results
        """
        analysis = {
            'type': analysis_type,
            'params': params or {},
            'timestamp': datetime.now().isoformat(),
            'result_summary': (result_summary[:200] if result_summary else "")
        }
        self.entities['analyses'].append(analysis)

        # If analysis was on specific ROI, track it
        if params and 'roi' in params:
            roi_name = params['roi']
            if roi_name in self.entities['rois']:
                self.entities['rois'][roi_name]['analyses_run'].append(analysis_type)

        logger.debug(f"Tracked analysis: {analysis_type}")

    def track_finding(self, finding: str, confidence: float = 1.0) -> None:
        """
        Track a key biological finding.

        Args:
            finding: The finding (e.g., "ROI_1 has 3x more malignant cells")
            confidence: Confidence level (0-1)
        """
        self.entities['findings'].append({
            'finding': finding,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        logger.debug(f"Tracked finding: {finding[:100]}")

    def get_relevant_entities(self, query: str) -> Dict[str, Any]:
        """
        Return entities relevant to current query.

        Uses simple keyword matching to find mentioned ROIs, genes, etc.

        Args:
            query: User query text

        Returns:
            Dict with relevant ROIs, genes, and recent analyses
        """
        relevant = {
            'rois': [],
            'genes': [],
            'recent_analyses': []
        }

        query_lower = query.lower()

        # Find mentioned ROIs (fuzzy match)
        for roi_name in self.entities['rois']:
            roi_normalized = roi_name.lower().replace('_', '').replace(' ', '')
            query_normalized = query_lower.replace('_', '').replace(' ', '')

            if roi_normalized in query_normalized or roi_name.lower() in query_lower:
                relevant['rois'].append(roi_name)

        # Find mentioned genes
        for gene in self.entities['genes']:
            if gene.lower() in query_lower:
                relevant['genes'].append(gene)

        # Get recent analyses (last 3)
        relevant['recent_analyses'] = self.entities['analyses'][-3:] if self.entities['analyses'] else []

        return relevant

    def get_entity_context_string(self) -> str:
        """
        Format entity context for prompt injection.

        Returns:
            Formatted string with ROIs created, recent findings, etc.
        """
        lines = []

        # ROIs created in this session (with slice/modality info)
        if self.entities['rois']:
            lines.append("**ROIs Created in this Session:**")
            for roi_name, roi_data in self.entities['rois'].items():
                # Format: "ROI_1 (slice 0, gene modality, 3686 cells)"
                info_parts = [roi_name]

                # Add slice info if available
                slice_id = roi_data.get('slice_id')
                if slice_id is not None:  # Handle slice_id = 0
                    info_parts.append(f"slice {slice_id}")

                # Add modality info
                modality = roi_data.get('modality')
                if modality:
                    info_parts.append(f"{modality} modality")

                # Add cell count
                cell_count = roi_data.get('cell_count', 0)
                if cell_count > 0:
                    info_parts.append(f"{cell_count:,} cells")

                # Format line
                if len(info_parts) > 1:
                    roi_desc = f"{info_parts[0]} ({', '.join(info_parts[1:])})"
                else:
                    roi_desc = info_parts[0]

                lines.append(f"- {roi_desc}")

        # Recent biological findings - REMOVED (already in conversation history)

        return "\n".join(lines) if lines else ""

    # ==========================================================

    def summarize_old_context(self) -> None:
        """Create a summary of older messages to save context window space."""
        if len(self.messages) <= self.max_context_messages:
            return

        # Get messages beyond the context window
        old_messages = self.messages[:-self.max_context_messages]

        # Simple summarization (can be enhanced with LLM later)
        user_queries = [m.content for m in old_messages if m.role == "user"]

        if user_queries:
            self.context_summary = (
                f"Earlier in this conversation, you discussed: "
                f"{'; '.join(user_queries[:3])}..."
            )
            logger.info("Created context summary for old messages")

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages.clear()
        self.turns.clear()
        self.context_summary = None
        logger.info("Cleared conversation memory")

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save conversation to disk.

        Returns
        -------
        Path
            Path to saved conversation file
        """
        if filename is None:
            filename = f"conversation_{self.session_id}.json"

        filepath = self.storage_dir / filename

        data = {
            "session_metadata": self.session_metadata,
            "messages": [msg.to_dict() for msg in self.messages],
            "turns": [turn.to_dict() for turn in self.turns],
            "context_summary": self.context_summary,
            "data_modifications": self.data_modifications.to_dict(),
            "conversation_summaries": [s.to_dict() for s in self.conversation_summaries],
            "entities": {
                'rois': self.entities['rois'],
                'genes': list(self.entities['genes']),  # Convert set to list for JSON
                'analyses': self.entities['analyses'],
                'findings': self.entities['findings']
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved conversation to {filepath}")
        return filepath

    def load(self, filepath: Path) -> None:
        """Load conversation from disk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.session_metadata = data["session_metadata"]
        self.messages = [Message.from_dict(msg) for msg in data["messages"]]
        self.turns = [ConversationTurn.from_dict(turn) for turn in data["turns"]]
        self.context_summary = data.get("context_summary")

        # Load new fields (with backward compatibility)
        if "data_modifications" in data:
            self.data_modifications = DataModificationHistory.from_dict(data["data_modifications"])
        else:
            self.data_modifications = DataModificationHistory()

        if "conversation_summaries" in data:
            self.conversation_summaries = [
                ConversationSummary.from_dict(s) for s in data["conversation_summaries"]
            ]
        else:
            self.conversation_summaries = []

        if "entities" in data:
            entities_data = data["entities"]
            self.entities['rois'] = entities_data.get('rois', {})
            self.entities['genes'] = set(entities_data.get('genes', []))  # Convert list back to set
            self.entities['analyses'] = entities_data.get('analyses', [])
            self.entities['findings'] = entities_data.get('findings', [])

        logger.info(f"Loaded conversation from {filepath}")

    def get_history_string(self) -> str:
        """Get formatted conversation history string for prompt injection.

        Returns formatted conversation context (query + assistant summary + execution).
        Suitable for injecting into planner/verifier/agent prompts.
        """
        context = self.get_context_for_llm()
        if not context:
            return ""

        parts = []
        for msg_dict in context:
            content = msg_dict['content']
            # Content is already formatted by _get_message_content_for_context
            # with **Query:**, **Assistant:**, **Execution:** prefixes
            parts.append(content)

        return "\n\n".join(parts)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        return {
            "session_id": self.session_id,
            "n_messages": len(self.messages),
            "n_turns": len(self.turns),
            "session_metadata": self.session_metadata
        }

    def __len__(self) -> int:
        """Return number of messages in conversation."""
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(session={self.session_id}, "
            f"messages={len(self.messages)}, turns={len(self.turns)})"
        )


__all__ = [
    "Message",
    "ConversationTurn",
    "DataModification",
    "DataModificationHistory",
    "ConversationSummary",
    "ConversationMemory"
]

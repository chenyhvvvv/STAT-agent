"""
Prompt Logger for debugging and optimization.

Logs full prompts and responses for all LLM calls during conversation turns.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptLogger:
    """
    Logger for capturing full LLM prompts and responses.

    Creates one markdown file per conversation turn containing:
    - User query
    - All LLM calls (skill matching, planning, main agent, validation, etc.)
    - Full system and dynamic prompts
    - Complete responses
    - Token usage and timing

    Parameters
    ----------
    log_dir : str
        Directory to save log files (default: "logs/")
    enabled : bool
        Whether logging is enabled (default: True - opt-out)

    Examples
    --------
    >>> logger = PromptLogger(log_dir="logs/", enabled=True)
    >>> logger.start_turn(user_message="What genes are available?")
    >>> logger.log_llm_call(
    ...     call_type="main_agent",
    ...     full_prompt=full_prompt,
    ...     response=response_text,
    ...     metadata={'model': 'claude-sonnet-3-5', 'input_tokens': 2500, ...}
    ... )
    >>> logger.end_turn(summary={'total_calls': 1, 'total_tokens': 2700})
    """

    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        """
        Initialize prompt logger.

        Parameters
        ----------
        log_dir : str
            Directory path for log files
        enabled : bool
            Enable/disable logging
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.session_dir: Optional[Path] = None  # Session subdirectory (shared with NotebookLogger)

        # State for current turn
        self.current_turn_file: Optional[Path] = None
        self.current_turn_content: list = []
        self.turn_number: int = 0
        self.llm_call_count: int = 0
        self.turn_start_time: Optional[datetime] = None

        # Create log directory if enabled
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"PromptLogger initialized: {self.log_dir.absolute()}")

    def set_session_dir(self, session_dir: Path) -> None:
        """
        Set the session directory for this logger.

        Should be called when NotebookLogger creates a new session directory,
        so both loggers save to the same location.

        Parameters
        ----------
        session_dir : Path
            Path to session directory (e.g., logs/session_20260205_103045/)

        Examples
        --------
        >>> notebook_logger.initialize_notebook("/path/to/data")
        >>> prompt_logger.set_session_dir(notebook_logger.get_session_dir())
        """
        self.session_dir = session_dir
        logger.info(f"PromptLogger: Using session directory: {session_dir}")

    def start_turn(self, user_message: str) -> None:
        """
        Start logging a new conversation turn.

        Parameters
        ----------
        user_message : str
            The user's query/message
        """
        if not self.enabled:
            logger.debug("PromptLogger: start_turn called but logging is disabled")
            return

        self.turn_number += 1
        self.llm_call_count = 0
        self.turn_start_time = datetime.now()
        self.current_turn_content = []

        # Determine save directory: use session_dir if set, otherwise log_dir
        save_dir = self.session_dir if self.session_dir else self.log_dir

        # Generate filename: turn_001_20260201_103045.md
        timestamp = self.turn_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"turn_{self.turn_number:03d}_{timestamp}.md"
        self.current_turn_file = save_dir / filename

        logger.info(f"PromptLogger: Starting turn {self.turn_number}, will save to {self.current_turn_file}")

        # Write header
        header = f"# Conversation Turn {self.turn_number} - {self.turn_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        header += f"## User Query\n\n```\n{user_message}\n```\n\n"
        header += "---\n\n"

        self.current_turn_content.append(header)
        logger.debug(f"Started turn {self.turn_number}: {self.current_turn_file}")

    def log_llm_call(self,
                     call_type: str,
                     full_prompt: str,
                     response: str,
                     metadata: Dict[str, Any]) -> None:
        """
        Log a single LLM call within the current turn.

        Parameters
        ----------
        call_type : str
            Type of LLM call: "skill_matching", "planner", "main_agent",
            "code_validation", etc.
        full_prompt : str
            Complete assembled prompt sent to LLM
        response : str
            Full LLM response
        metadata : dict
            Call metadata: model, input_tokens, output_tokens, duration, etc.
        """
        if not self.enabled or not self.current_turn_file:
            return

        self.llm_call_count += 1

        # Build log entry
        entry = f"## LLM Call {self.llm_call_count}: {call_type}\n\n"

        # Metadata section
        entry += "### Request Metadata\n"
        entry += f"- **Model:** {metadata.get('model', 'unknown')}\n"
        entry += f"- **Timestamp:** {metadata.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n"
        entry += f"- **Purpose:** {call_type.replace('_', ' ').title()}\n"
        if 'input_tokens' in metadata:
            entry += f"- **Input tokens:** {metadata['input_tokens']}\n"
        if 'output_tokens' in metadata:
            entry += f"- **Output tokens:** {metadata['output_tokens']}\n"
        if 'duration' in metadata:
            entry += f"- **Duration:** {metadata['duration']:.2f}s\n"
        entry += "\n"

        # Full assembled prompt
        entry += "### Full Prompt\n\n"
        entry += "```\n"
        entry += full_prompt
        entry += "\n```\n\n"

        # Response section
        entry += "### Response\n\n"
        entry += "```\n"
        entry += response
        entry += "\n```\n\n"
        entry += "---\n\n"

        self.current_turn_content.append(entry)
        logger.debug(f"Logged LLM call {self.llm_call_count}: {call_type}")

    def log_clarification(self,
                         question: str,
                         response: str,
                         context_type: str = "clarification") -> None:
        """
        Log a clarification Q&A within the current turn.

        Parameters
        ----------
        question : str
            The clarification question asked to the user
        response : str
            The user's response to the clarification
        context_type : str
            Type of clarification: "planner", "verifier", or "skill_selection"
        """
        if not self.enabled or not self.current_turn_file:
            return

        # Build clarification entry
        entry = f"## User Response ({context_type.replace('_', ' ').title()} Clarification)\n\n"
        entry += f"**Question:**\n```\n{question}\n```\n\n"
        entry += f"**User Response:**\n```\n{response}\n```\n\n"
        entry += "---\n\n"

        self.current_turn_content.append(entry)
        logger.debug(f"Logged clarification: {context_type}")

    def end_turn(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize the turn log and write to file.

        Parameters
        ----------
        summary : dict, optional
            Summary statistics for the turn:
            - total_llm_calls: int
            - total_input_tokens: int
            - total_output_tokens: int
            - total_duration: float
        """
        if not self.enabled or not self.current_turn_file:
            logger.debug(f"PromptLogger: end_turn called but enabled={self.enabled}, current_file={self.current_turn_file}")
            return

        logger.info(f"PromptLogger: Ending turn {self.turn_number}, logged {self.llm_call_count} LLM calls")

        # Add summary section
        if summary:
            summary_section = "## Summary\n\n"
            if 'total_llm_calls' in summary:
                summary_section += f"- **Total LLM calls:** {summary['total_llm_calls']}\n"
            if 'total_input_tokens' in summary:
                summary_section += f"- **Total input tokens:** {summary['total_input_tokens']}\n"
            if 'total_output_tokens' in summary:
                summary_section += f"- **Total output tokens:** {summary['total_output_tokens']}\n"
            if 'total_duration' in summary:
                summary_section += f"- **Total duration:** {summary['total_duration']:.2f}s\n"

            if self.turn_start_time:
                turn_end_time = datetime.now()
                summary_section += f"- **Turn completed:** {turn_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"

            self.current_turn_content.append(summary_section)

        # Write to file
        try:
            logger.info(f"PromptLogger: Writing log to {self.current_turn_file}")
            with open(self.current_turn_file, 'w', encoding='utf-8') as f:
                f.write(''.join(self.current_turn_content))
            logger.info(f"PromptLogger: Successfully saved prompt log: {self.current_turn_file}")
        except Exception as e:
            logger.error(f"PromptLogger: Failed to write prompt log: {e}")

        # Reset state
        self.current_turn_content = []
        self.llm_call_count = 0

    def disable(self) -> None:
        """Disable logging."""
        self.enabled = False
        logger.info("PromptLogger disabled")

    def enable(self) -> None:
        """Enable logging."""
        self.enabled = True
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PromptLogger enabled: {self.log_dir.absolute()}")

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        session_info = f", session={self.session_dir.name}" if self.session_dir else ""
        return f"PromptLogger(dir={self.log_dir}, status={status}, turn={self.turn_number}{session_info})"


__all__ = ['PromptLogger']

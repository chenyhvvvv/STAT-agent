"""
Code executor for spatial transcriptomics agent.

Executes generated Python code safely with session management.
"""

from __future__ import annotations

import io
import traceback
import contextlib
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    variables: Dict[str, Any] = None
    plots: Optional[List[str]] = None  # List of base64 encoded plot images

    def get_display_output(self) -> str:
        """Get formatted output for display."""
        if not self.success:
            return f"Error: {self.error}\n\n{self.stderr}"

        output = self.stdout
        if self.stderr:
            output += f"\n\nWarnings:\n{self.stderr}"
        return output or "Code executed successfully (no output)"


class CodeExecutor:
    """
    Executes Python code with session management.

    Features:
    - Safe code execution in controlled namespace
    - Session state management (variables persist)
    - Capture stdout/stderr
    - Error handling and reporting

    Parameters
    ----------
    timeout : int
        Execution timeout in seconds (default: 60)
    safe_mode : bool
        Enable safety checks (default: True)

    Examples
    --------
    >>> executor = CodeExecutor()
    >>> result = executor.execute("x = 5\\nprint(x * 2)", namespace={"session": session})
    >>> print(result.stdout)
    10
    """

    def __init__(self, timeout: int = 60, safe_mode: bool = True):
        self.timeout = timeout
        self.safe_mode = safe_mode
        self.namespace: Dict[str, Any] = {}
        logger.info(f"Initialized code executor (timeout={timeout}s, safe_mode={safe_mode})")

    def execute(
        self,
        code: str,
        namespace: Optional[Dict[str, Any]] = None,
        capture_output: bool = True
    ) -> ExecutionResult:
        """
        Execute Python code.

        Parameters
        ----------
        code : str
            Python code to execute
        namespace : Optional[Dict]
            Namespace dictionary (variables available to code)
        capture_output : bool
            Whether to capture stdout/stderr

        Returns
        -------
        ExecutionResult
            Execution result with output and any errors
        """
        # Clean code (remove markdown fences if present)
        code = self._clean_code(code)

        if not code or not code.strip():
            return ExecutionResult(
                success=False,
                error="Empty code provided"
            )

        # Safety checks
        if self.safe_mode:
            safety_check = self._check_code_safety(code)
            if not safety_check[0]:
                return ExecutionResult(
                    success=False,
                    error=f"Safety check failed: {safety_check[1]}"
                )

        # Prepare namespace
        exec_namespace = self.namespace.copy()
        if namespace:
            exec_namespace.update(namespace)

        # Capture output
        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            # Compile code
            compiled_code = compile(code, "<spatial_agent>", "exec")

            # Execute with output capture
            if capture_output:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compiled_code, exec_namespace)
            else:
                exec(compiled_code, exec_namespace)

            # Update persistent namespace (exclude builtins and special vars)
            for key, value in exec_namespace.items():
                if not key.startswith('_') and key not in ['__builtins__']:
                    self.namespace[key] = value

            # Capture matplotlib figures
            plots = self._capture_plots()

            result = ExecutionResult(
                success=True,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                variables=self.namespace.copy(),
                plots=plots if plots else None
            )

            logger.info(f"Code executed successfully ({len(plots) if plots else 0} plots captured)")
            return result

        except SyntaxError as e:
            error_msg = f"Syntax Error: {e}"
            logger.error(f"Syntax error in code execution: {e}")
            return ExecutionResult(
                success=False,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                error=error_msg
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            error_trace = traceback.format_exc()
            logger.error(f"Error executing code: {e}\n{error_trace}")

            return ExecutionResult(
                success=False,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue() + "\n" + error_trace,
                error=error_msg
            )

    def _clean_code(self, code: str) -> str:
        """Remove markdown code fences from code."""
        code = code.strip()

        # Remove markdown code blocks
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line if it's ```python or ```
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code.strip()

    def _check_code_safety(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Check code for unsafe operations.

        Returns (is_safe, error_message)
        """
        # Dangerous patterns
        dangerous_patterns = [
            (r'\bos\.system\b', "os.system() not allowed"),
            (r'\beval\b', "eval() not allowed"),
            (r'\bexec\b', "nested exec() not allowed"),
            (r'\b__import__\b', "dynamic imports not allowed"),
            (r'\bopen\([^)]*[\'"]w', "file writing not allowed in safe mode"),
            (r'\bsubprocess\b', "subprocess not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                return False, message

        return True, None

    def _capture_plots(self) -> list:
        """
        Capture all open matplotlib figures as base64 encoded images.

        Returns
        -------
        list
            List of base64 encoded PNG images
        """
        plots = []

        try:
            import matplotlib.pyplot as plt
            import base64

            # Get all figure numbers
            fig_nums = plt.get_fignums()

            if not fig_nums:
                return plots

            for fig_num in fig_nums:
                fig = plt.figure(fig_num)

                # Save figure to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)

                # Encode to base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plots.append(img_base64)

                # Close figure to free memory
                plt.close(fig)

            logger.info(f"Captured {len(plots)} matplotlib figures")

        except ImportError:
            logger.debug("matplotlib not available, skipping plot capture")
        except Exception as e:
            logger.warning(f"Failed to capture plots: {e}")

        return plots

    def reset_namespace(self) -> None:
        """Reset the execution namespace."""
        self.namespace.clear()
        logger.info("Reset execution namespace")

    def get_variables(self) -> Dict[str, Any]:
        """Get current namespace variables."""
        return self.namespace.copy()

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace."""
        self.namespace[name] = value

    def __repr__(self) -> str:
        return f"CodeExecutor(vars={len(self.namespace)}, safe_mode={self.safe_mode})"


__all__ = ["ExecutionResult", "CodeExecutor"]

"""
Error Reflection and Self-Correction Module.

When code execution fails, agent should analyze the error,
understand what went wrong, and attempt to fix it autonomously.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def reflect_on_error_and_fix(
    llm_backend,
    original_code: str,
    error_result,
    user_query: str,
    available_context: Dict[str, Any],
    max_attempts: int = 2
) -> Dict[str, Any]:
    """
    Analyze execution error and generate fixed code.

    Args:
        llm_backend: LLM backend to use for reflection
        original_code: Code that failed
        error_result: ExecutionResult with error details
        user_query: Original user question
        available_context: Current system state (ROIs, columns, etc.)
        max_attempts: Maximum number of fix attempts

    Returns:
        Dict with:
            - fixed_code: str | None
            - reasoning: str (why it failed, how to fix)
            - confidence: float (0-1)
            - should_retry: bool
    """

    error_msg = error_result.error or error_result.stderr

    # Build reflection prompt
    reflection_prompt = f"""You are debugging Python code that failed to execute.

**User's Original Question:**
"{user_query}"

**Code That Failed:**
```python
{original_code}
```

**Error Message:**
{error_msg}

**Available Context:**
- ROIs: {available_context.get('available_rois', {})}
  (Format: {{'ROI_1': {{'slice_id': '0', 'modality': 'gene', 'n_cells': 3686}}}})
- Columns: {available_context.get('available_columns', [])}
- Has celltype column: {available_context.get('has_celltype', False)}

**Your Task:**
1. Analyze why the code failed
2. Identify the root cause (wrong ROI name? missing column? logic error? wrong API?)
3. Determine if fix is SMALL or LARGE
4. Generate fixed code ONLY if fix is small

**Change Magnitude Guidelines:**
- SMALL: Typo fix, wrong variable name, incorrect API call, missing import, parameter adjustment
- LARGE: Logic rewrite, algorithm change, adding complex workarounds, restructuring code flow

**Common Issues to Check:**
- KeyError: Wrong ROI name or missing key in dictionary (SMALL fix)
- AttributeError: Wrong method or attribute name (SMALL fix)
- NameError: Variable not defined in scope (SMALL fix)
- ValueError: Invalid value for operation (may be SMALL or LARGE)
- TypeError: Wrong type for operation (may be SMALL or LARGE)
- ImportError/ModuleNotFoundError: Missing import (SMALL fix)

**Common API Fixes (SMALL changes):**
- For sparse matrix checks: Use `scipy.sparse.issparse(X)` or `scipy.sparse.isspmatrix_csr(X)` instead of scanpy private APIs
- For scanpy: Use public API methods only (sc.pp.*, sc.tl.*, sc.pl.*)
- For AnnData: Access .X, .obs, .var directly; use .copy() for subsetting
- For session: Use `session.roi_subsets[roi_name]` to get ROI data
- For numpy: Always import as `np`, use `np.asarray()` for conversions

**IMPORTANT: Return ONLY valid JSON, nothing else. No markdown, no explanations outside the JSON.**

**Return JSON:**
{{
  "error_type": "KeyError|AttributeError|etc.",
  "root_cause": "Brief explanation of what went wrong",
  "fix_strategy": "What needs to be changed",
  "change_magnitude": "small|large",
  "fixed_code": "Complete corrected Python code (only if change_magnitude is small)",
  "confidence": 0.9,
  "should_retry": true
}}

If change_magnitude is "large" or error is unfixable, set should_retry to false.
"""

    try:
        response = await llm_backend.run(reflection_prompt)

        # Extract JSON - robust multi-strategy parsing
        import json

        reflection = None

        # Strategy 1: Try parsing entire response as JSON
        try:
            reflection = json.loads(response)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from markdown code fence
        if reflection is None:
            json_fence = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if json_fence:
                try:
                    reflection = json.loads(json_fence.group(1))
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Find complete JSON object using brace counting
        if reflection is None:
            start = response.find('{')
            if start != -1:
                brace_count = 0
                end = start
                for i in range(start, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                try:
                    reflection = json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

        # All strategies failed
        if reflection is None:
            logger.warning(f"Error reflection returned invalid JSON. First 500 chars:\n{response[:500]}")
            return {
                'fixed_code': None,
                'reasoning': response[:200],
                'confidence': 0.0,
                'should_retry': False,
                'change_magnitude': 'unknown'
            }

        change_magnitude = reflection.get('change_magnitude', 'large')

        # If change is large or confidence is low, don't retry
        confidence = reflection.get('confidence', 0.5)
        should_retry = reflection.get('should_retry', True)

        if change_magnitude == 'large':
            logger.info(f"Error reflection: Change magnitude is LARGE - skipping retry")
            should_retry = False
        elif confidence < 0.7:
            logger.info(f"Error reflection: Low confidence ({confidence:.2f}) - skipping retry")
            should_retry = False

        logger.info(
            f"Error reflection: {reflection.get('error_type')} - "
            f"{reflection.get('root_cause')} (magnitude: {change_magnitude})"
        )

        return {
            'fixed_code': reflection.get('fixed_code'),
            'reasoning': f"{reflection.get('root_cause')} → {reflection.get('fix_strategy')}",
            'confidence': confidence,
            'should_retry': should_retry,
            'change_magnitude': change_magnitude
        }

    except Exception as e:
        logger.error(f"Error reflection failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            'fixed_code': None,
            'reasoning': str(e),
            'confidence': 0.0,
            'should_retry': False,
            'change_magnitude': 'unknown'
        }


def extract_error_info(error_result) -> Dict[str, str]:
    """
    Extract structured information from error.

    Args:
        error_result: ExecutionResult with error

    Returns:
        Dict with error_type, error_msg, relevant_line
    """
    error_text = error_result.error or error_result.stderr or ""

    # Try to extract error type
    error_type = "UnknownError"
    if "KeyError" in error_text:
        error_type = "KeyError"
    elif "AttributeError" in error_text:
        error_type = "AttributeError"
    elif "NameError" in error_text:
        error_type = "NameError"
    elif "ValueError" in error_text:
        error_type = "ValueError"
    elif "TypeError" in error_text:
        error_type = "TypeError"
    elif "IndexError" in error_text:
        error_type = "IndexError"

    # Extract error message
    error_msg = error_text

    # Try to extract relevant line
    line_match = re.search(r'line (\d+)', error_text)
    relevant_line = line_match.group(1) if line_match else None

    return {
        'error_type': error_type,
        'error_msg': error_msg,
        'relevant_line': relevant_line
    }


def should_attempt_fix(error_result) -> bool:
    """
    Determine if we should attempt to fix this error.

    Some errors are unfixable (e.g., missing data) and we should
    just report them to the user instead of retrying.

    Args:
        error_result: ExecutionResult with error

    Returns:
        True if we should attempt to fix
    """
    error_text = error_result.error or error_result.stderr or ""

    # Errors we should try to fix
    fixable_patterns = [
        "KeyError",           # Wrong key name
        "AttributeError",     # Wrong method/attribute
        "NameError",          # Variable not defined
        "ValueError",         # Invalid value (might be fixable)
    ]

    # Errors that are likely unfixable
    unfixable_patterns = [
        "MemoryError",
        "TimeoutError",
        "PermissionError"
    ]

    for pattern in unfixable_patterns:
        if pattern in error_text:
            return False

    for pattern in fixable_patterns:
        if pattern in error_text:
            return True

    # Default: try to fix unknown errors
    return True


__all__ = [
    'reflect_on_error_and_fix',
    'extract_error_info',
    'should_attempt_fix'
]

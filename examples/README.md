# Examples

Example scripts and demos for STAT.

## Files

### `demo_spatial_agent.py`
Demo script showing basic usage of the SpatialAgent for spatial transcriptomics analysis.

**Usage**:
```bash
python examples/demo_spatial_agent.py
```

**What it demonstrates**:
- Loading spatial data
- Initializing the agent
- Basic chat interactions
- Code generation and execution

### `test_prompt_logger.py`
Test script to verify the prompt logger is working correctly.

**Usage**:
```bash
python examples/test_prompt_logger.py
```

**What it tests**:
- Prompt logger initialization
- Log file creation
- Full prompt and response capture
- Metadata logging

## Running Examples

All examples should be run from the repository root:

```bash
# From repository root
python examples/demo_spatial_agent.py
python examples/test_prompt_logger.py
```

## Requirements

Make sure you have installed the package and its dependencies:

```bash
pip install -r requirements.txt
```

For examples requiring API keys (OpenAI, Anthropic, etc.), set them as environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

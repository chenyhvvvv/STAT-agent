"""
Quick test script to verify prompt logger is working.

Run this to check if logs are created in logs/ directory.
"""

import asyncio
from stat_agent.agent.spatial_agent_core import SpatialAgent
from stat_agent.core.session import SimpleSession

async def test_prompt_logger():
    print("Testing prompt logger...")

    # Create agent with logging enabled
    agent = SpatialAgent(
        model="claude-sonnet-3-5-20241022",
        enable_prompt_logging=True,  # Should be default
        prompt_log_dir="logs"
    )

    print(f"Prompt logger status: {agent.prompt_logger}")
    print(f"Logging enabled: {agent.prompt_logger.enabled}")
    print(f"Log directory: {agent.prompt_logger.log_dir.absolute()}")

    # Test with a simple query (no session needed for testing)
    print("\nSending test query...")
    try:
        response = await agent.chat("Hello, this is a test")
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"Error during chat: {e}")

    # Check if log file was created
    import os
    log_dir = agent.prompt_logger.log_dir
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.md') and f.startswith('turn_')]

    print(f"\nLog files created: {len(log_files)}")
    for log_file in log_files:
        print(f"  - {log_file}")
        # Show first few lines
        with open(log_dir / log_file, 'r') as f:
            lines = f.readlines()[:10]
            print("    Content preview:")
            for line in lines:
                print(f"    {line.rstrip()}")

    if log_files:
        print("\n✅ Prompt logger is working!")
    else:
        print("\n❌ No log files created - logger may not be working")

if __name__ == "__main__":
    asyncio.run(test_prompt_logger())

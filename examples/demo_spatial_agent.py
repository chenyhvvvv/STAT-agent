"""
Demo script for Spatial Transcriptomics Agent.

This demonstrates the full agent capabilities including:
- LLM integration
- Conversation memory
- Task planning
- Code execution
- Skill usage
"""

import asyncio
import os
import sys
from pathlib import Path

# Add stat_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from stat_agent.agent.spatial_agent_core import SpatialAgent
from stat_agent.core.session import SimpleSession


async def demo_agent():
    """Demonstrate agent capabilities."""

    print("=" * 60)
    print("Spatial Transcriptomics Agent Demo")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  No API key found!")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use the agent.")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nContinuing with rule-based responses only...\n")
        model = "python"  # Local execution only
    else:
        model = os.getenv("SPATIAL_AGENT_MODEL", "gpt-4o")
        print(f"\n✅ Using model: {model}")

    # Create session with example data
    print("\n1. Creating session...")
    session = SimpleSession(name="demo_session")

    # Use default paths if available
    adata_path = os.getenv(
        "DEMO_ADATA_PATH",
        "/import/home3/yhchenmath/Dataset/CellARTPaper/figure_4/adata_breast_cancer_rep1_x_y.h5ad"
    )
    image_path = os.getenv(
        "DEMO_IMAGE_PATH",
        "/import/home3/yhchenmath/Code/ucs/paper_data/materials_xenium_breast_cancer/he.tif"
    )

    try:
        if Path(adata_path).exists() and Path(image_path).exists():
            print(f"   Loading data from: {adata_path}")
            session.load_data_legacy(adata_path, image_path)

            summary = session.get_summary()
            slice_0 = session.get_slice(0)
            print(f"   ✓ Loaded {slice_0.n_obs:,} cells/spots, {slice_0.n_vars:,} features")
            print(f"   ✓ Modality: {slice_0.modality}, Data level: {slice_0.data_level}")
        else:
            print(f"   ⚠️  Demo data not found at default paths")
            print(f"   Set DEMO_ADATA_PATH and DEMO_IMAGE_PATH environment variables")
            return
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return

    # Initialize agent
    print("\n2. Initializing agent...")
    agent = SpatialAgent(
        model=model,
        api_key=api_key,
        session=session,
        enable_planning=True,
        enable_skills=True
    )
    print(f"   ✓ Agent ready!")
    print(f"   - Model: {agent.llm.config.model}")
    print(f"   - Planning: {agent.enable_planning}")
    print(f"   - Skills: {agent.enable_skills}")

    # Demo queries
    queries = [
        "What data is loaded?",
        "Show me the cell type distribution",
        "How many cell types are there?",
    ]

    print("\n3. Testing agent with example queries...")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 60)

        try:
            response = await agent.chat(query, execute_code=True)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 60)

    # Show agent status
    print("\n4. Agent Status:")
    print("=" * 60)
    status = agent.get_status()
    print(f"   Model: {status['model']}")
    print(f"   Session active: {status['session_active']}")
    print(f"   Data loaded: {status['data_loaded']}")
    print(f"   Conversation length: {status['conversation_length']} messages")
    print(f"   Skills loaded: {status['skills_loaded']}")

    # Save conversation
    print("\n5. Saving conversation...")
    save_path = agent.memory.save("demo_conversation.json")
    print(f"   ✓ Saved to: {save_path}")

    print("\n" + "=" * 60)
    print("Demo completed! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Run the web interface: ./start_web.sh")
    print("  • Read the guide: AGENT_GUIDE.md")
    print("  • Create custom skills in: .claude/skills/")
    print("=" * 60)


def main():
    """Main entry point."""
    try:
        asyncio.run(demo_agent())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

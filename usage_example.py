#!/usr/bin/env python3
"""
Usage Example: GPT-Researcher Tool Selector Integration

This file demonstrates how to use the new gpt_researcher_tool_selection function
as a replacement or enhancement to the original select_tools_node function.
"""

import asyncio
from gpt_researcher_tool_selector import gpt_researcher_tool_selection


async def main():
    """
    Example showing how to use the new tool selector function.

    This preserves the same interface and prompt structure as the original
    select_tools_node but adds gpt-researcher capabilities.
    """

    print("GPT-Researcher Tool Selector Usage Examples")
    print("=" * 50)

    # Example 1: Basic usage with user input and task description
    print("\n🔍 Example 1: Basic Tool Selection")
    print("-" * 30)

    user_input = "For the period of 2024-01-01 to 2025-08-01 can you please create moving average crossover of 10 day vs 50 day for AAPL and TSLA"
    task_description = "Technical analysis requiring stock price data"

    result = await gpt_researcher_tool_selection(
        user_input=user_input,
        task_description=task_description,
        use_research=False  # Start with direct analysis
    )

    print(f"User Input: {user_input[:60]}...")
    print(f"Selected Tools: {result['selected_tools']}")
    print(f"Reasoning: {result['reasoning']}")

    # Example 2: Using with gpt-researcher (if API keys available)
    print("\n🧠 Example 2: Enhanced Research Mode")
    print("-" * 35)

    try:
        enhanced_result = await gpt_researcher_tool_selection(
            user_input=user_input,
            task_description=task_description,
            use_research=True  # Enable gpt-researcher
        )

        print("✅ Research mode successful:")
        print(f"Selected Tools: {enhanced_result['selected_tools']}")
        print(f"Reasoning: {enhanced_result['reasoning']}")

    except Exception as e:
        print(f"⚠️  Research mode not available: {e}")
        print("   (This usually means API keys are not configured)")

    # Example 3: Integration pattern similar to original select_tools_node
    print("\n🔄 Example 3: Integration Pattern")
    print("-" * 32)

    # Simulate how this would integrate with existing AgentState pattern
    class MockAgentState(dict):
        """Mock AgentState for demonstration"""
        pass

    # Create a mock state similar to what select_tools_node receives
    state = MockAgentState({
        'user_input': "Extract US CPI data and calculate rolling beta of tech stocks",
        'task_description': "Economic analysis with stock correlation"
    })

    # Use the new function
    tool_selection_result = await gpt_researcher_tool_selection(
        user_input=state['user_input'],
        task_description=state.get('task_description', ''),
        use_research=False
    )

    # Update state like the original function would
    state['selected_tools'] = tool_selection_result['selected_tools']

    print(f"State updated:")
    print(f"  user_input: {state['user_input'][:50]}...")
    print(f"  selected_tools: {state['selected_tools']}")

    # Show how this preserves the original interface
    print("\n📋 Summary")
    print("-" * 10)
    print("✅ Preserves REGISTRY.get_available_tools() functionality")
    print("✅ Maintains same prompt structure as original select_tools_node")
    print("✅ Returns same format: {'selected_tools': [...], 'reasoning': '...'}")
    print("✅ Adds gpt-researcher enhancement option")
    print("✅ Graceful fallback when API keys not available")


if __name__ == "__main__":
    asyncio.run(main())
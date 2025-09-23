#!/usr/bin/env python3
"""
Test script for GPT-Researcher Tool Selector

This script demonstrates how to use the gpt_researcher_tool_selection function
with different types of requests.
"""

import asyncio
import os
from gpt_researcher_tool_selector import gpt_researcher_tool_selection


async def run_tests():
    """Run various test scenarios for tool selection."""

    print("=" * 60)
    print("GPT-Researcher Tool Selector Test Suite")
    print("=" * 60)
    print()

    # Test scenarios
    test_cases = [
        {
            "name": "Single Stock Analysis",
            "user_input": "Analyze AAPL stock performance for Q4 2024",
            "task_description": "Need daily stock price data for Apple",
            "use_research": False
        },
        {
            "name": "Multiple Stocks Comparison",
            "user_input": "Compare AAPL, MSFT, GOOGL, AMZN stock performance",
            "task_description": "Portfolio analysis of tech giants",
            "use_research": False
        },
        {
            "name": "Economic Analysis",
            "user_input": "Analyze unemployment rate trends and their correlation with inflation",
            "task_description": "Macroeconomic analysis using FRED data",
            "use_research": False
        },
        {
            "name": "Fundamental Analysis",
            "user_input": "Extract P/E ratios and earnings data for TSLA over 2 years",
            "task_description": "Fundamental analysis of Tesla",
            "use_research": False
        },
        {
            "name": "Mixed Analysis",
            "user_input": "Create a trading strategy based on moving averages and economic indicators",
            "task_description": "Need both price data and macro indicators",
            "use_research": False
        }
    ]

    # Run each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"User Input: {test_case['user_input']}")
        print(f"Task Description: {test_case['task_description']}")
        print()

        try:
            result = await gpt_researcher_tool_selection(
                user_input=test_case['user_input'],
                task_description=test_case['task_description'],
                use_research=test_case['use_research']
            )

            print(f"✅ Selected Tools: {result['selected_tools']}")
            print(f"📝 Reasoning: {result['reasoning']}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()
        print("=" * 60)
        print()


async def test_with_research():
    """Test with gpt-researcher if API keys are available."""

    print("Testing with GPT-Researcher (requires API keys)...")
    print("-" * 50)

    # Check if API keys are set
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')

    if not openai_key or not tavily_key:
        print("⚠️  API keys not found. Set OPENAI_API_KEY and TAVILY_API_KEY to test with research.")
        print("   Using direct analysis mode instead.")
        use_research = False
    else:
        print("✅ API keys found. Testing with research mode...")
        use_research = True

    try:
        result = await gpt_researcher_tool_selection(
            user_input="Compare the financial performance of renewable energy companies vs traditional energy companies in 2024",
            task_description="Comprehensive analysis requiring multiple data sources",
            use_research=use_research
        )

        print(f"Selected Tools: {result['selected_tools']}")
        print(f"Reasoning: {result['reasoning']}")

    except Exception as e:
        print(f"Error: {e}")

    print()


def main():
    """Main function to run all tests."""
    print("Starting GPT-Researcher Tool Selector Tests...")
    print()

    # Run basic tests
    asyncio.run(run_tests())

    # Run research test
    asyncio.run(test_with_research())

    print("🎉 All tests completed!")


if __name__ == "__main__":
    main()
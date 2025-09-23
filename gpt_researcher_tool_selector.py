"""
GPT-Researcher Tool Selection Module

This module provides a tool selection function that uses gpt-researcher library
for enhanced research capabilities while preserving the prompt structure and
registry integration from the original select_tools_node function.

REQUIRED API KEYS:
==================
To use this module, you need to set the following environment variables:

1. OPENAI_API_KEY - Required for GPT-Researcher LLM operations
   - Get from: https://platform.openai.com/api-keys
   - Set in environment: export OPENAI_API_KEY="your-api-key"

2. TAVILY_API_KEY - Required for web search functionality
   - Get from: https://tavily.com/
   - Set in environment: export TAVILY_API_KEY="your-api-key"

3. Additional data source API keys (based on your tools):
   - ALPHA_VANTAGE_API_KEY - For stock data (if using extract_daily_stock_data)
   - FRED_API_KEY - For economic data (if using extract_economic_data_from_fred)
   - FMP_API_KEY - For fundamental data (if using extract_fundamentals_from_fmp)
   - POLYGON_API_KEY - For bulk stock data (if using bulk_extract_daily_closing_prices_from_polygon)

Note: Without proper API keys, the module will fail to function correctly.
"""

import asyncio
from typing import Dict, Any
from gpt_researcher import GPTResearcher
from tools_registry import ToolRegistry, auto_register
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress Pydantic warnings for compatibility
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function name constant (preserved from original)
TOOL_SELECTION_FUNCTION_NAME = "select_tools"

# Initialize registry (same as in data_analytics_agent.py)
REGISTRY = ToolRegistry()
try:
    auto_register(REGISTRY)
except Exception:
    pass


async def gpt_researcher_tool_selection(
    user_input: str,
    task_description: str = ""
) -> Dict[str, Any]:
    """
    Select relevant tools based on user input using gpt-researcher for enhanced decision making.

    Preserves the prompt structure from the original select_tools_node function while
    leveraging gpt-researcher's research capabilities.

    Args:
        user_input (str): The user's request/query
        task_description (str): Additional task context (optional)
        use_research (bool): Whether to use gpt-researcher for enhanced tool selection

    Returns:
        Dict[str, Any]: Dictionary containing 'selected_tools' and 'reasoning'
    """
    logger.info("Starting tool selection with gpt-researcher...")

    # Get available tools from registry (preserved from original)
    try:
        registry_tools = REGISTRY.get_available_tools()
    except Exception as e:
        logger.warning(f"Failed to get registry tools: {e}")
        registry_tools = []

    available_tool_names = sorted(registry_tools)

    if not available_tool_names:
        logger.warning("No tools available in registry")
        return {
            "selected_tools": [],
            "reasoning": "No tools available in the registry"
        }

    # Build the enhanced system prompt (preserved structure from original)
    system_content = f"""
    Based on this request, I need to select the most appropriate data extraction tools from the following available options:
{', '.join(available_tool_names)}

Please analyze what type of data would be needed for this request and recommend which specific tools should be used.
ALso plan out different subagents which can handle paraller workflow and also their tool needs so that downstream i can run mutltiple agent with their tools required that you will mention.You are an expert tool selection assistant for data analytics tasks.
Based on the user's request, determine which tools from the available tools are needed to complete the task effectively.

Available tools: {', '.join(available_tool_names)}
"""


    # Prepare the query for gpt-researcher
    research_query = f"""
User Request: {user_input}
Task Description: {task_description}
Based on this request, I need to select the most appropriate data extraction tools from the following available options:
{', '.join(available_tool_names)}

Please analyze what type of data would be needed for this request and recommend which specific tools should be used.
ALso plan out different subagents which can handle paraller workflow and also their tool needs so that downstream i can run mutltiple agent with their tools required that you will mention.You are an expert tool selection assistant for data analytics tasks.
Based on the user's request, determine which tools from the available tools are needed to complete the task effectively.
please let me know just the name of different subagents i should have and what ools i should assign to them
Available tools: {', '.join(available_tool_names)}
"""

    try:
        # Use gpt-researcher for tool selection
        logger.info("Conducting research for tool selection...")
        researcher = GPTResearcher(
            query=research_query,
            report_type="deep",
        )

        # Conduct research
        research_result = await researcher.conduct_research()
        print(research_result)
        # Generate a focused report on tool selection
        custom_prompt = f"""Based on the research conducted, provide a comprehensive analysis for tool selection.

{system_content}

Research Context: {research_result}



Provide a detailed analysis including:
1. Different subagents or tasks needed
2. Their required tools from the available tools
3. Reasoning for each selection
4. Parallel workflow recommendations"""

        try:
            report = await researcher.write_report(custom_prompt=custom_prompt)
        except Exception as e:
            report = await researcher.write_report()
          
        print (report)
    except Exception as e:
        logger.error(f"Error in tool selection: {e}")
        return {"report": f"Error occurred during research: {str(e)}"}










# Example usage function
async def example_usage():
    """Example of how to use the gpt_researcher_tool_selection function."""

    # Example 1: Stock analysis request
    user_input = "For the last 4 years can you extract US CPI data on monthly basis and then calculate the rolling 2 year beta of AAPL, TSLA, NVDA, GOOGL, AMZN, MSFT, NFLX returns with CPI change then, and backtest a strategy by  rank ordering them each month and go long on stocks with low rank values and short stock with high rank values"
    task_description = "Downstream i will use your results in runnng parallel workflow for genrating charts,organized csv for customer  as dashboards for easy visulaiztion of scenario"

    result = await gpt_researcher_tool_selection(user_input, task_description)
    print("Example 1 - Stock Analysis:")
    print(f"Selected tools: {result['selected_tools']}")
    print(f"Reasoning: {result['reasoning']}")
    print()

    # # Example 2: Economic analysis request
    # user_input = "Compare unemployment rate trends with inflation data"
    # task_description = "Economic analysis using FRED data"

    # result = await gpt_researcher_tool_selection(user_input, task_description)
    # print("Example 2 - Economic Analysis:")
    # print(f"Selected tools: {result['selected_tools']}")
    # print(f"Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())
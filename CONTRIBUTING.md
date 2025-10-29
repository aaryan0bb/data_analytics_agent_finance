# Contributing to Data Analytics Agent

Thank you for your interest in contributing to the Data Analytics Agent! This guide will help you understand how to add new features, tools, and examples to the codebase.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Adding New Tools](#adding-new-tools)
- [Adding Few-Shot Examples](#adding-few-shot-examples)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

---

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- API keys for data providers (see `.env.example`)

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git checkout https://github.com/YOUR_USERNAME/data_analytics_agent_finance.git
cd data_analytics_agent_finance

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/data_analytics_agent_finance.git
```

---

## Development Setup

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Verify installation**:
   ```bash
   python -c "from data_analytics_agent_new import DataAnalyticsAgent; print('✓ Setup successful')"
   ```

---

## Adding New Tools

Tools are implemented as plugins that inherit from `ToolPlugin`. Follow these steps:

### 1. Create Tool Plugin

Create a new plugin in `tools_clean.py` (or create a new file if thematically different):

```python
from tools_registry import ToolPlugin
from typing import Dict, Any
import pandas as pd

class MyNewToolPlugin(ToolPlugin):
    """
    Brief description of what this tool does.
    """

    @property
    def name(self) -> str:
        """Unique tool name used in function calling."""
        return "my_new_tool"

    @property
    def description(self) -> str:
        """Description shown to LLM for tool selection."""
        return "Fetches XYZ data from ABC source with filtering capabilities"

    @property
    def semantic_key(self) -> str:
        """Semantic key for caching and data referencing."""
        return "xyz_data"

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters (OpenAI function calling format)."""
        return {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "The XYZ identifier to fetch data for"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                }
            },
            "required": ["identifier", "start_date", "end_date"]
        }

    def execute(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute the tool and return a DataFrame.

        Args:
            params: Validated parameters dictionary

        Returns:
            pd.DataFrame with standardized 'date' column as first column
        """
        identifier = params["identifier"]
        start_date = params["start_date"]
        end_date = params["end_date"]

        # Your implementation here
        # Example: Fetch data from API, database, etc.
        data = self._fetch_from_source(identifier, start_date, end_date)

        # Convert to DataFrame with 'date' as first column
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date'] + [col for col in df.columns if col != 'date']]

        return df

    def _fetch_from_source(self, identifier, start_date, end_date):
        """Helper method for actual data fetching."""
        # Implementation details
        pass
```

### 2. Add Validation Logic

Add parameter validation in `validator.py`:

```python
def validate_tool_params(tool_name: str, params: Dict[str, Any]):
    """Validate and fix tool parameters."""

    if tool_name == "my_new_tool":
        issues = []
        fixed = params.copy()

        # Validate identifier
        if not fixed.get("identifier"):
            issues.append("identifier is required")

        # Validate and fix dates
        if not fixed.get("start_date"):
            fixed["start_date"] = "2023-01-01"
            issues.append("start_date missing, defaulted to 2023-01-01")

        if not fixed.get("end_date"):
            fixed["end_date"] = datetime.now().strftime("%Y-%m-%d")
            issues.append("end_date missing, defaulted to today")

        # Additional validation...

        ok = len(issues) == 0
        return ok, fixed if issues else None, issues
```

### 3. Register the Tool

The tool will be auto-registered if placed in a file that's imported. Verify registration:

```python
from tools_registry import ToolRegistry, auto_register

registry = ToolRegistry()
auto_register(registry)

print(f"Available tools: {registry.get_available_tools()}")
# Should include "my_new_tool"
```

### 4. Test Your Tool

```python
# Test execution
tool = MyNewToolPlugin()
result = tool.execute({
    "identifier": "TEST123",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
})

print(f"Result shape: {result.shape}")
print(f"Columns: {result.columns.tolist()}")
assert "date" in result.columns
assert result["date"].dtype == "datetime64[ns]"
```

---

## Adding Few-Shot Examples

Few-shot examples teach the LLM how to use tools and write analysis code. Add examples to the `few_shots/` directory.

### 1. Choose Appropriate File

- `factor_generation.py` - Factor engineering patterns
- `signal_generation.py` - Trading signals and rankings
- `simple_data_processing.py` - Data transformation
- `simple_visuals.py` - Visualization examples
- `statistical_analysis.py` - Statistical modeling
- Create new file for new categories

### 2. Follow the Standard Format

```python
# few_shots/my_category.py

[
    {
        "question": "How do I calculate rolling correlation between two time series?",
        "executable_code": """
import pandas as pd
import numpy as np
import os

# Load data from tool outputs
prices = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'stock_prices.csv'))

# Calculate rolling correlation
rolling_corr = prices['AAPL'].rolling(window=60).corr(prices['SPY'])

# Create result DataFrame
result = pd.DataFrame({
    'date': prices['date'],
    'rolling_correlation': rolling_corr
})

# Save result
result.to_csv(os.path.join(os.environ['DATA_DIR'], 'rolling_correlation.csv'), index=False)

# Create manifest
import json
manifest = {
    'tables': [{
        'path': 'rolling_correlation.csv',
        'rows': len(result),
        'columns': result.columns.tolist(),
        'description': '60-day rolling correlation between AAPL and SPY'
    }],
    'figures': [],
    'metrics': {
        'avg_correlation': float(rolling_corr.mean()),
        'max_correlation': float(rolling_corr.max()),
        'min_correlation': float(rolling_corr.min())
    },
    'explanation': 'Calculated 60-day rolling correlation to measure relationship strength over time.'
}

with open(os.path.join(os.environ['DATA_DIR'], 'result.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
""",
        "code_description": "Calculates rolling correlation between two time series with configurable window size"
    }
]
```

### 3. Example Quality Guidelines

**Good Examples**:
- ✅ Use `os.environ['DATA_DIR']` for all file operations
- ✅ Always create `result.json` manifest
- ✅ Include tables, figures, and metrics in manifest
- ✅ Use descriptive variable names
- ✅ Add comments explaining key steps
- ✅ Handle edge cases (empty data, NaNs, etc.)
- ✅ Follow PLOTLY_CONVENTIONS for visualizations
- ✅ Implement backtest hygiene (1-bar delay, realistic costs)

**Avoid**:
- ❌ Hardcoded file paths
- ❌ Missing manifest creation
- ❌ Network calls (use tool outputs only)
- ❌ Time-based randomness without seed
- ❌ Look-ahead bias in backtests

### 4. Test Your Example

```python
# Verify example loads correctly
from data_analytics_agent_new import FewShotRetriever

retriever = FewShotRetriever("few_shots/")
examples = retriever.retrieve("rolling correlation analysis", top_k=5)

# Check if your example appears in results
for ex in examples:
    if "rolling correlation" in ex.get("description", "").lower():
        print("✓ Example found in retrieval results")
        break
```

---

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 120 characters
- Use docstrings for all public classes and functions

### Naming Conventions

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Tool names: `lowercase_with_underscores`
- Semantic keys: `lowercase_with_underscores`

### Documentation

- Add docstrings to all public APIs
- Include type hints in function signatures
- Provide usage examples in docstrings
- Update README.md for major feature additions

### Example:

```python
from typing import Dict, Any, List
import pandas as pd

class ExampleProcessor:
    """
    Processes example data with advanced transformations.

    This class provides methods for data cleaning, transformation,
    and validation following enterprise best practices.

    Example:
        >>> processor = ExampleProcessor(config={"threshold": 0.5})
        >>> result = processor.transform(data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize processor with configuration.

        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data according to configuration.

        Args:
            data: Input DataFrame with 'date' column

        Returns:
            Transformed DataFrame with same schema

        Raises:
            ValueError: If data is missing required columns
        """
        if "date" not in data.columns:
            raise ValueError("Data must contain 'date' column")

        # Implementation...
        return data
```

---

## Testing

### Unit Tests

Create tests in `tests/` directory:

```python
# tests/test_my_tool.py
import pytest
from tools_clean import MyNewToolPlugin

def test_my_tool_basic():
    """Test basic tool functionality."""
    tool = MyNewToolPlugin()
    result = tool.execute({
        "identifier": "TEST",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    })

    assert result is not None
    assert "date" in result.columns
    assert len(result) > 0

def test_my_tool_validation():
    """Test parameter validation."""
    from validator import validate_tool_params

    ok, fixed, issues = validate_tool_params("my_new_tool", {})
    assert not ok
    assert fixed is not None
    assert len(issues) > 0
```

### Integration Tests

```python
def test_end_to_end_workflow():
    """Test complete workflow with new tool."""
    from data_analytics_agent_new import DataAnalyticsAgent

    agent = DataAnalyticsAgent()
    response = agent.process_request(
        "Fetch XYZ data for TEST123 from 2023 and calculate summary statistics"
    )

    assert "TEST123" in response
    assert "statistics" in response.lower()
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_tool.py

# Run with coverage
pytest --cov=. tests/
```

---

## Submitting Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-tool
```

### 2. Commit Changes

```bash
git add .
git commit -m "Add MyNewTool for XYZ data extraction

- Implements ToolPlugin interface
- Adds parameter validation
- Includes unit tests
- Updates documentation"
```

### 3. Push and Create PR

```bash
git push origin feature/my-new-tool
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to any related issues
- Screenshots/examples if applicable
- Test results

### 4. PR Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No sensitive data (API keys, etc.) committed
- [ ] `.env` not committed
- [ ] New dependencies added to `requirements.txt`
- [ ] Examples tested with FAISS retrieval

---

## Questions or Issues?

- Open an issue on GitHub for bugs or feature requests
- Check existing issues before creating new ones
- Provide detailed reproduction steps for bugs
- Include code examples when relevant

---

Thank you for contributing to Data Analytics Agent! 🚀

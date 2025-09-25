# 📦 Installation Guide - Data Analytics Agent

## 🚀 Quick Start (Recommended - UV)

### Option 1: Automated Setup
```bash
# Run the setup script
./setup.sh
```

### Option 2: Manual UV Setup
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run agent
uv run python data_analytics_agent.py
```

## 🐍 Traditional Python Setup

### Option 3: pip + venv
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run agent
python data_analytics_agent.py
```

### Option 4: conda
```bash
# Create conda environment
conda create -n data-analytics python=3.11
conda activate data-analytics

# Install dependencies
pip install -r requirements.txt
```

## 📋 Dependencies Summary

**Total Packages**: 24 core dependencies
- **LLM/AI**: 5 packages (OpenAI, LangGraph, etc.)
- **Research**: 4 packages (Tavily, embeddings, etc.)
- **Data**: 10 packages (Pandas, NumPy, Plotly, etc.)
- **Utilities**: 5 packages (async, environment, etc.)

## 🔧 Development Setup

```bash
# UV with dev dependencies
uv sync --extra dev

# pip with dev dependencies (uncomment in requirements.txt)
pip install pytest pytest-asyncio hypothesis black ruff
```

## 📊 Environment Configuration

### Required API Keys (.env file):
```bash
# Critical - Required for agent operation
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

# Financial data APIs
FMP_API_KEY=your_fmp_key_here
POLYGON_API_KEY=your_polygon_key_here
FRED_API_KEY=your_fred_key_here

# Optional - for tracing
LANGCHAIN_API_KEY=your_langchain_key_here
```

## ✅ Verification Tests

### Test Basic Installation:
```bash
python -c "
import openai, langgraph, pandas
from tools_registry import ToolRegistry
print('✅ Core imports successful')
"
```

### Test Agent Initialization:
```bash
python -c "
from data_analytics_agent import DataAnalyticsAgent
agent = DataAnalyticsAgent()
print('✅ Agent initialized successfully')
"
```

## 🐳 Docker Setup (Production)

```dockerfile
FROM python:3.11-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY . .

# Install dependencies
RUN uv sync --frozen

EXPOSE 8000
CMD ["uv", "run", "python", "data_analytics_agent.py"]
```

## 🔍 Troubleshooting

### Common Issues:

**Import Errors**:
```bash
# Check Python version (3.9+ required)
python --version

# Verify environment activation
which python
```

**API Key Errors**:
```bash
# Check .env file exists and has correct keys
cat .env | grep -E "OPENAI|TAVILY"
```

**Permission Errors (macOS)**:
```bash
# Install UV with brew
brew install uv
```

**UV Not Found**:
```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc
```

## 🚀 Performance Comparison

| Method | Install Time | Dependency Resolution |
|--------|-------------|----------------------|
| UV | ~3 seconds | Lightning fast ⚡ |
| pip + venv | ~60 seconds | Standard speed |
| conda | ~90 seconds | Slower |

## 📈 Recommended Setup by Use Case

- **Production**: UV (fastest, deterministic)
- **Development**: UV with `--extra dev`
- **Research**: conda (scientific packages)
- **CI/CD**: UV (consistent, fast)

## 🎯 Next Steps After Installation

1. **Configure Environment**: Update `.env` with your API keys
2. **Test Agent**: Run a simple request
3. **Read Documentation**: Check README.md for usage examples
4. **Enable Logging**: Configure LangSmith for monitoring

Your data analytics agent is ready for high-stakes deployment! 🚀
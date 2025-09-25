# 🚀 UV Setup Guide for Data Analytics Agent

## 🎯 Why UV?

UV is **10-100x faster** than pip and provides deterministic dependency management perfect for production deployments.

## 📦 Quick Setup (Recommended)

### 1. Install UV
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: with pip
pip install uv
```

### 2. Setup Project Environment
```bash
cd /Users/aaryangoyal/Desktop/coffee_code/data_analytics_agent_copy_2

# Create virtual environment and install dependencies
uv sync
```

### 3. Run Your Agent
```bash
# Activate environment and run
uv run python data_analytics_agent.py

# Or run example
uv run python -c "
from data_analytics_agent import DataAnalyticsAgent
agent = DataAnalyticsAgent()
response = agent.process_request('Analyze AAPL stock data for 2024')
print(response)
"
```

## 🔧 Advanced UV Commands

### Development Setup
```bash
# Install with dev dependencies
uv sync --extra dev

# Add new dependency
uv add requests

# Add dev dependency
uv add --dev pytest

# Add optional jupyter support
uv sync --extra jupyter
```

### Dependency Management
```bash
# Update dependencies
uv lock --upgrade

# Install from requirements.txt (fallback)
uv pip install -r requirements.txt

# Export current environment
uv export > requirements.lock.txt
```

### Virtual Environment Control
```bash
# Show environment info
uv info

# Clean environment
uv clean

# Run commands in environment
uv run python --version
uv run pytest
uv run jupyter notebook
```

## ⚡ Performance Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|----|---------|
| Install 30+ deps | ~60s | ~3s | **20x faster** |
| Resolve dependencies | ~15s | ~0.5s | **30x faster** |
| Create venv | ~5s | ~0.1s | **50x faster** |

## 🐳 Production Deployment

### Docker Integration
```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy project files
COPY . /app
WORKDIR /app

# Install dependencies
RUN uv sync --frozen

# Run application
CMD ["uv", "run", "python", "data_analytics_agent.py"]
```

### GitHub Actions CI/CD
```yaml
- name: Set up uv
  uses: astral-sh/setup-uv@v4
  with:
    version: "latest"

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: uv run pytest
```

## 🔒 Dependency Locking

UV creates `uv.lock` with exact versions:
```bash
# Generate lockfile
uv lock

# Install from lockfile (production)
uv sync --frozen

# Update specific package
uv lock --upgrade-package openai
```

## 🚀 Migration from pip

If you have existing `requirements.txt`:
```bash
# Import from requirements.txt
uv add $(cat requirements.txt | grep -v "^#" | tr '\n' ' ')

# Or install directly
uv pip install -r requirements.txt
```

## 📊 Environment Variables

UV respects your `.env` file automatically:
```bash
# Your .env is loaded automatically
uv run python data_analytics_agent.py
```

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `uv: command not found`
```bash
# Restart shell or run:
source ~/.bashrc  # Linux
source ~/.zshrc   # macOS
```

**Issue**: Permission errors
```bash
# Use user install
pip install --user uv
```

**Issue**: Slow package resolution
```bash
# Clear cache
uv clean
uv cache clean
```

## 📈 Production Benefits

✅ **Deterministic**: `uv.lock` ensures identical environments
✅ **Fast**: 10-100x faster installations
✅ **Reliable**: Better dependency resolution
✅ **Portable**: Works across platforms
✅ **Secure**: Built-in security scanning

## 🎯 Next Steps

1. **Run setup**: `uv sync`
2. **Test agent**: `uv run python data_analytics_agent.py`
3. **Development**: `uv sync --extra dev`
4. **Deploy**: Use `uv.lock` for production

Your agent is now ready for high-performance, production-grade deployment! 🚀
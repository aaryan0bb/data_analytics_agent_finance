#!/bin/bash
# Data Analytics Agent - Quick Setup Script
# Installs UV and sets up the complete environment

set -e  # Exit on error

echo "🚀 Data Analytics Agent - Production Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if UV is installed
check_uv() {
    if command -v uv >/dev/null 2>&1; then
        print_success "UV is already installed: $(uv --version)"
        return 0
    else
        print_status "UV not found, installing..."
        return 1
    fi
}

# Install UV
install_uv() {
    print_status "Installing UV (ultra-fast Python package manager)..."

    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        print_error "Unsupported OS: $OSTYPE"
        print_status "Please install UV manually: https://docs.astral.sh/uv/"
        exit 1
    fi

    # Source the shell configuration to make uv available
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v uv >/dev/null 2>&1; then
        print_success "UV installed successfully: $(uv --version)"
    else
        print_error "UV installation failed"
        print_status "Please restart your terminal and run this script again"
        exit 1
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python version..."

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        print_success "Python $PYTHON_VERSION found"

        # Check if version is 3.9+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python version is compatible (3.9+ required)"
        else
            print_error "Python 3.9+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        print_status "Please install Python 3.9+ and run this script again"
        exit 1
    fi
}

# Setup environment
setup_environment() {
    print_status "Setting up project environment..."

    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        print_warning ".env file not found"
        print_status "Make sure to configure your API keys in .env before running the agent"
    else
        print_success ".env file found"
    fi

    # Install dependencies with UV
    print_status "Installing dependencies (this may take a moment)..."
    uv sync

    print_success "Dependencies installed successfully"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."

    # Test import of key modules
    uv run python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import openai
    print('✅ OpenAI imported successfully')
except ImportError as e:
    print(f'❌ OpenAI import failed: {e}')

try:
    import langgraph
    print('✅ LangGraph imported successfully')
except ImportError as e:
    print(f'❌ LangGraph import failed: {e}')

try:
    import pandas
    print('✅ Pandas imported successfully')
except ImportError as e:
    print(f'❌ Pandas import failed: {e}')

try:
    from tools_registry import ToolRegistry, auto_register
    registry = ToolRegistry()
    auto_register(registry)
    tools = registry.get_available_tools()
    print(f'✅ Tool registry: {len(tools)} tools discovered')
except Exception as e:
    print(f'❌ Tool registry error: {e}')
"

    if [[ $? -eq 0 ]]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Main setup process
main() {
    print_status "Starting Data Analytics Agent setup..."

    # Step 1: Check and install UV
    if ! check_uv; then
        install_uv
    fi

    # Step 2: Check Python
    check_python

    # Step 3: Setup environment
    setup_environment

    # Step 4: Verify installation
    verify_installation

    # Success message
    echo ""
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Configure your API keys in .env (if not already done)"
    echo "2. Run the agent:"
    echo "   uv run python data_analytics_agent.py"
    echo ""
    echo "📖 For more information:"
    echo "   - Read UV_SETUP.md for advanced usage"
    echo "   - Check README.md for agent documentation"
    echo ""
    echo "🚀 Your production-ready Data Analytics Agent is ready!"
}

# Run main function
main "$@"
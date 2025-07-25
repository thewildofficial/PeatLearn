#!/usr/bin/env python3
"""
Ray Peat Legacy - Project Setup and Management Script

This script handles project initialization, dependency installation,
and environment setup for the Ray Peat Legacy platform.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent

def run_command(command, cwd=None, check=True):
    """Run a shell command with error handling."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd or PROJECT_ROOT,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("🐍 Setting up Python environment...")
    
    venv_path = PROJECT_ROOT / "venv"
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command(f"{sys.executable} -m venv venv")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    print("Installing Python dependencies...")
    run_command(f"{pip_path} install --upgrade pip")
    run_command(f"{pip_path} install -r requirements.txt")
    
    print("✅ Python environment setup complete!")

def setup_directories():
    """Create required project directories."""
    print("📁 Creating project directories...")
    
    from config.settings import ensure_directories
    ensure_directories()
    
    print("✅ Project directories created!")

def setup_environment_file():
    """Set up environment configuration file."""
    print("⚙️ Setting up environment configuration...")
    
    env_file = PROJECT_ROOT / ".env"
    env_template = PROJECT_ROOT / ".env.template"
    
    if not env_file.exists() and env_template.exists():
        shutil.copy(env_template, env_file)
        print("📝 Created .env file from template")
        print("⚠️  Please edit .env file and add your API keys!")
    else:
        print("✅ Environment file already exists")

def setup_frontend():
    """Set up React frontend."""
    print("⚛️ Setting up React frontend...")
    
    frontend_dir = PROJECT_ROOT / "web_ui" / "frontend"
    package_json = frontend_dir / "package.json"
    
    if package_json.exists():
        print("Installing Node.js dependencies...")
        run_command("npm install", cwd=frontend_dir)
        print("✅ Frontend dependencies installed!")
    else:
        print("⚠️  Frontend package.json not found. Creating basic React app...")
        run_command("npx create-react-app frontend", cwd=PROJECT_ROOT / "web_ui")
        print("✅ Basic React app created!")

def setup_database():
    """Initialize database."""
    print("🗄️ Setting up database...")
    
    # This will be implemented when we have the database models
    print("⚠️  Database setup will be implemented with SQLAlchemy models")

def run_tests():
    """Run project tests."""
    print("🧪 Running tests...")
    
    test_result = run_command("python -m pytest tests/ -v", check=False)
    if test_result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above.")

def check_api_keys():
    """Check if required API keys are configured."""
    print("🔑 Checking API key configuration...")
    
    from config.settings import settings
    
    if not settings.GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not configured in .env file")
        return False
    
    print("✅ API keys configured!")
    return True

def run_data_processing_demo():
    """Run a demo of the data processing pipeline."""
    print("🔄 Running data processing demo...")
    
    if not check_api_keys():
        print("❌ Cannot run demo without API keys. Please configure .env file.")
        return
    
    # Run a small sample of the cleaning pipeline
    cleaning_script = PROJECT_ROOT / "preprocessing" / "cleaning" / "main_pipeline.py"
    if cleaning_script.exists():
        run_command(f"python {cleaning_script} --limit 3 --verbose")
        print("✅ Data processing demo complete!")
    else:
        print("⚠️  Data processing pipeline not found")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Ray Peat Legacy Setup Script")
    parser.add_argument("--all", action="store_true", help="Run full setup")
    parser.add_argument("--python", action="store_true", help="Setup Python environment")
    parser.add_argument("--frontend", action="store_true", help="Setup frontend")
    parser.add_argument("--env", action="store_true", help="Setup environment file")
    parser.add_argument("--dirs", action="store_true", help="Create directories")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--demo", action="store_true", help="Run data processing demo")
    parser.add_argument("--check", action="store_true", help="Check configuration")
    
    args = parser.parse_args()
    
    print("🧬 Ray Peat Legacy - Bioenergetic Knowledge Platform Setup")
    print("=" * 60)
    
    if args.all or args.dirs:
        setup_directories()
    
    if args.all or args.env:
        setup_environment_file()
    
    if args.all or args.python:
        setup_python_environment()
    
    if args.all or args.frontend:
        setup_frontend()
    
    if args.test:
        run_tests()
    
    if args.check:
        check_api_keys()
    
    if args.demo:
        run_data_processing_demo()
    
    if not any(vars(args).values()):
        print("No setup options specified. Use --help for options or --all for full setup.")
        print("\nQuick start:")
        print("  python setup.py --all      # Full setup")
        print("  python setup.py --check    # Check configuration")
        print("  python setup.py --demo     # Run demo")
    
    print("\n🎉 Setup complete! Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python setup.py --demo")
    print("3. Start development: docker-compose up -d")

if __name__ == "__main__":
    main() 
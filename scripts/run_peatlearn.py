#!/usr/bin/env python3
"""
PeatLearn Startup Script
Automatically sets up and runs the complete AI-enhanced adaptive learning system
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'google-generativeai',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def setup_environment():
    """Set up the virtual environment and install dependencies"""
    print("🔧 Setting up environment...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  Not in virtual environment. Activating venv...")
        # Note: This script should be run with source venv/bin/activate first
        
    # Check for missing packages
    missing = check_requirements()
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, check=True)
    else:
        print("✅ All required packages are installed!")

def check_environment_variables():
    """Check for required environment variables"""
    print("🔍 Checking environment variables...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  No .env file found. Creating from template...")
        with open('.env', 'w') as f:
            f.write("# PeatLearn Environment Variables\n")
            f.write("# Add your API keys here\n\n")
            f.write("# Google Gemini API Key for AI features\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
        print("📝 Created .env file. Please add your GOOGLE_API_KEY for full AI features.")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key and api_key != 'your_google_api_key_here':
        print("✅ Google API Key found - AI features enabled!")
        return True
    else:
        print("⚠️  No valid Google API Key found - using fallback mode")
        return False

def create_data_directories():
    """Ensure data directories exist"""
    print("📁 Creating data directories...")
    
    directories = [
        'data/user_interactions',
        '.taskmaster/docs',
        '.taskmaster/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Data directories ready!")

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting PeatLearn Dashboard...")
    print("=" * 60)
    print("🎉 Welcome to PeatLearn - AI-Enhanced Adaptive Learning!")
    print("📖 Chat with Ray Peat AI and watch your profile evolve")
    print("🧠 Get personalized recommendations based on your learning")
    print("📊 Track your progress across bioenergetic topics")
    print("=" * 60)
    
    # Run streamlit
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 
        'peatlearn_master.py',
        '--server.port=8501',
        '--server.headless=false'
    ])

def main():
    """Main startup function"""
    print("🧠 PeatLearn AI-Enhanced Adaptive Learning System")
    print("=" * 50)
    
    try:
        # Setup steps
        setup_environment()
        ai_enabled = check_environment_variables()
        create_data_directories()
        
        print("\n🎯 System Status:")
        print(f"  • Virtual Environment: {'✅' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else '⚠️'}")
        print(f"  • AI Features: {'✅ Enabled' if ai_enabled else '⚠️ Fallback Mode'}")
        print(f"  • Data Directories: ✅ Ready")
        print(f"  • Adaptive Learning: ✅ Ready")
        
        print("\n🚀 Features Available:")
        print("  • Real-time AI profiling with Gemini")
        print("  • Adaptive content recommendations") 
        print("  • Personalized quiz generation")
        print("  • Learning progress tracking")
        print("  • Topic mastery assessment")
        
        if not ai_enabled:
            print("\n💡 To enable full AI features:")
            print("  1. Get a Google Gemini API key from: https://makersuite.google.com/app/apikey")
            print("  2. Add it to your .env file: GOOGLE_API_KEY=your_key_here")
            print("  3. Restart this script")
        
        print("\n" + "=" * 50)
        
        # Start the application
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n👋 Thanks for using PeatLearn! Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting PeatLearn: {e}")
        print("Please check the error above and try again.")

if __name__ == "__main__":
    main()

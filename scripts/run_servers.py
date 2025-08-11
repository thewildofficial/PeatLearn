#!/usr/bin/env python3
"""
PeatLearn Development Server Launcher
Starts both the backend API and Streamlit frontend in the virtual environment
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import signal

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

def find_venv_python():
    """Find the Python executable in the virtual environment."""
    venv_path = PROJECT_ROOT / "venv"
    
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if python_exe.exists():
        return str(python_exe)
    else:
        print("❌ Virtual environment not found. Please run: python -m venv venv")
        sys.exit(1)

def check_embeddings():
    """Check if embeddings are available, download if needed."""
    emb_file = PROJECT_ROOT / "embedding" / "vectors" / "embeddings_20250728_221826.npy"
    
    if not emb_file.exists():
        print("📥 Downloading embeddings from Hugging Face (one-time ~700MB)...")
        try:
            python_exe = find_venv_python()
            result = subprocess.run([
                python_exe, "embedding/download_from_hf.py"
            ], check=True)
            print("✅ Embeddings downloaded successfully!")
        except subprocess.CalledProcessError:
            print("⚠️  Failed to download embeddings. Backend may start without them.")
    else:
        print("✅ Embeddings already available.")

def main():
    print("🧬 PeatLearn Development Server")
    print("=" * 50)
    
    python_exe = find_venv_python()
    print(f"🐍 Using Python: {python_exe}")
    
    # Check embeddings
    check_embeddings()
    
    processes = []
    
    try:
        # Start basic RAG backend
        print("🚀 Starting RAG backend server (port 8000)...")
        backend_env = os.environ.copy()
        backend_env["PYTHONPATH"] = str(PROJECT_ROOT / "inference" / "backend") + ":" + str(PROJECT_ROOT)
        backend_env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")
        backend_env["PATH"] = str(PROJECT_ROOT / "venv" / "bin") + ":" + backend_env.get("PATH", "")
        
        backend_process = subprocess.Popen([
            python_exe, "-m", "uvicorn", 
            "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], env=backend_env, cwd=PROJECT_ROOT / "inference" / "backend")
        processes.append(backend_process)
        
        # Start advanced ML backend
        print("🧠 Starting Advanced ML backend server (port 8001)...")
        advanced_process = subprocess.Popen([
            python_exe, "-m", "uvicorn", 
            "advanced_app:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        ], env=backend_env, cwd=PROJECT_ROOT / "inference" / "backend")
        processes.append(advanced_process)
        
        # Wait a moment for backends to start
        time.sleep(3)
        
        # Start Streamlit frontend
        print("📊 Starting Streamlit frontend (port 8501)...")
        streamlit_env = os.environ.copy()
        streamlit_env["PYTHONPATH"] = str(PROJECT_ROOT)
        streamlit_env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")
        streamlit_env["PATH"] = str(PROJECT_ROOT / "venv" / "bin") + ":" + streamlit_env.get("PATH", "")
        
        streamlit_process = subprocess.Popen([
            python_exe, "-m", "streamlit", "run",
            "scripts/streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], env=streamlit_env, cwd=PROJECT_ROOT)
        processes.append(streamlit_process)
        
        print("\n🎉 Development servers are running:")
        print("📡 RAG API:       http://localhost:8000")
        print("📋 RAG API Docs:  http://localhost:8000/docs")
        print("🧠 Advanced ML:   http://localhost:8001") 
        print("📋 ML API Docs:   http://localhost:8001/docs")
        print("📊 Streamlit:     http://localhost:8501")
        print("\n⏹️  Press Ctrl+C to stop all servers")
        
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process died
            for process in processes:
                if process.poll() is not None:
                    print(f"⚠️  Process {process.pid} exited")
                    raise KeyboardInterrupt
                    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        
        # Terminate all processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        print("✅ All servers stopped.")

if __name__ == "__main__":
    main()

#!/bin/bash
# Launch PeatLearn in Development Mode
# This script enables auto-refresh and development features

echo "🚀 Launching PeatLearn in Development Mode..."
echo "🔧 Auto-refresh and development features will be available"
echo ""

# Method 1: Using command line flag
python peatlearn_master.py --dev

# Alternative methods (commented out):
# Method 2: Using environment variable
# export PEATLEARN_DEV_MODE=true
# python peatlearn_master.py

# Method 3: Direct Streamlit with environment
# STREAMLIT_DEV_MODE=true streamlit run peatlearn_master.py

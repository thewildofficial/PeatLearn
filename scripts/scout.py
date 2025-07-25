import json
import os
from pathlib import Path

def analyze_file_system(root_dir, output_file):
    """
    Analyzes the file system starting from the root directory and creates a JSON report.

    Args:
        root_dir (str): The absolute path to the root directory to analyze.
        output_file (str): The absolute path to the output JSON file.
    """
    file_data = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('.'):
                continue  # Skip hidden files

            file_path = Path(root) / file
            relative_path = file_path.relative_to(root_dir)
            
            # Basic categorization based on parent directory
            category = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

            if category not in file_data:
                file_data[category] = []
            
            file_data[category].append({
                "file_path": str(relative_path),
                "file_type": file_path.suffix.lower(),
            })

    with open(output_file, 'w') as f:
        json.dump(file_data, f, indent=4)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python scout.py <root_dir> <output_file>")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    analyze_file_system(root_dir, output_file)

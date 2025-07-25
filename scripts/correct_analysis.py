import json
import re

def correct_analysis_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Analysis file not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return

    corrected_count = 0
    for category in data:
        if isinstance(data[category], list):
            for item in data[category]:
                is_corrected = False
                if 'suggested_actions' in item and isinstance(item['suggested_actions'], list):
                    original_len = len(item['suggested_actions'])
                    item['suggested_actions'] = [
                        action for action in item['suggested_actions'] 
                        if 'truncate' not in action.lower() and 'truncation' not in action.lower()
                    ]
                    if len(item['suggested_actions']) != original_len:
                        is_corrected = True

                if 'notes' in item and isinstance(item['notes'], str):
                    original_notes = item['notes']
                    sentences = re.split(r'(?<=[.!?])\s+', item['notes'])
                    good_sentences = [
                        sentence for sentence in sentences 
                        if 'truncate' not in sentence.lower() and 'truncation' not in sentence.lower()
                    ]
                    item['notes'] = ' '.join(good_sentences)
                    if item['notes'] != original_notes:
                        is_corrected = True
                
                if is_corrected:
                    corrected_count += 1

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Corrected {corrected_count} entries in {file_path}")

if __name__ == '__main__':
    ANALYSIS_FILE = "/Users/aban/drive/Projects/PeatLearn/file_analysis_with_scores.json"
    correct_analysis_file(ANALYSIS_FILE)

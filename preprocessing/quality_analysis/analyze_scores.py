import json
import pandas as pd

def analyze_corpus_scores(json_file):
    """
    Analyzes the corpus scores from the JSON file and prints a summary.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return

    # Flatten the data into a list of records
    records = []
    for category, files in data.items():
        if isinstance(files, list):
            for item in files:
                record = {
                    'category': category,
                    'semantic_noise_score': item.get('semantic_noise_score'),
                    'speaker_attribution_score': item.get('speaker_attribution_score'),
                    'document_atomicity_score': item.get('document_atomicity_score'),
                    'textual_fidelity_score': item.get('textual_fidelity_score')
                }
                records.append(record)

    df = pd.DataFrame(records)

    # Calculate average scores for each category
    category_analysis = df.groupby('category').agg({
        'semantic_noise_score': 'mean',
        'speaker_attribution_score': 'mean',
        'document_atomicity_score': 'mean',
        'textual_fidelity_score': 'mean',
    }).round(2)

    # Calculate overall average scores
    overall_analysis = df.agg({
        'semantic_noise_score': 'mean',
        'speaker_attribution_score': 'mean',
        'document_atomicity_score': 'mean',
        'textual_fidelity_score': 'mean',
    }).round(2)

    print("--- Corpus Analysis Summary ---")
    print("\nAverage Scores by Category:")
    print(category_analysis)
    print("\nOverall Average Scores:")
    print(overall_analysis)
    print("\n--- End of Summary ---")

if __name__ == '__main__':
    JSON_FILE = "/Users/aban/drive/Projects/PeatLearn/file_analysis_with_scores.json"
    analyze_corpus_scores(JSON_FILE)

import json
import csv

def convert_json_to_csv(json_file, csv_file):
    """
    Converts the analysis JSON file to a CSV file.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return

    # Define the headers for the CSV file
    headers = [
        "category",
        "file_path",
        "file_type",
        "semantic_noise_score",
        "speaker_attribution_score",
        "document_atomicity_score",
        "textual_fidelity_score",
        "suggested_actions",
        "notes"
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for category, files in data.items():
            if isinstance(files, list):
                for item in files:
                    # Ensure all headers are present in the row
                    row = {header: item.get(header, '') for header in headers}
                    row['category'] = category
                    # Join list of actions into a single string
                    if isinstance(row['suggested_actions'], list):
                        row['suggested_actions'] = '\n'.join(row['suggested_actions'])
                    writer.writerow(row)

    print(f"Successfully converted {json_file} to {csv_file}")

if __name__ == '__main__':
    JSON_FILE = "/Users/aban/drive/Projects/PeatLearn/file_analysis_with_scores.json"
    CSV_FILE = "/Users/aban/drive/Projects/PeatLearn/corpus_analysis.csv"
    convert_json_to_csv(JSON_FILE, CSV_FILE)

import json, csv
from metrics.chrf import calculate_chrf
from metrics.exact_match import calculate_exact_match
from metrics.jaccard_similarity import calculate_jaccard
from metrics.rouge_ import calculate_rouge

def calculate_metrics():
    with open("evaluated_data\\all\codes.json", "r") as f:
        data = json.load(f)

    chrf_scores, chrf_average = calculate_chrf(data)
    exact_matches, exact_matches_average = calculate_exact_match(data)
    jaccard_scores, jaccard_average = calculate_jaccard(data)
    rouge_scores, rouge_average = calculate_rouge(data)

    for i, item in enumerate(data):
        item["metrics"] = {
            "chrf": chrf_scores[i]/100,
            "exact_match": 1.0 if exact_matches[i] else 0.0,
            "jaccard": jaccard_scores[i],
            "rouge": rouge_scores[i]
        }

    with open("evaluated_data/all/codes_with_metrics.json", "w") as f:
        json.dump(data, f, indent=4)

def print_report_from_json_to_csv():
    with open("evaluated_data/all/codes_with_metrics.json", "r") as f:
        data = json.load(f)

    csv_file_path = "evaluated_data/all/codes_with_metrics.csv"
    headers = data[0].keys()

    with open(csv_file_path, "w", newline='', encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    calculate_metrics()
    print_report_from_json_to_csv()




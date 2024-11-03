import json
from metrics.chrf import calculate_chrf
from metrics.exact_match import calculate_exact_match
from metrics.jaccard_similarity import calculate_jaccard
from metrics.rouge_ import calculate_rouge

if __name__ == "__main__":
    with open("evaluated_data\\all\codes.json", "r") as f:
        data = json.load(f)

    chrf_scores, chrf_average = calculate_chrf(data)
    exact_matches, exact_matches_average = calculate_exact_match(data)
    jaccard_scores, jaccard_average = calculate_jaccard(data)
    rouge_scores, rouge_average = calculate_rouge(data)
    print(chrf_average, exact_matches_average, jaccard_average, rouge_average)

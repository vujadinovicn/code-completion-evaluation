def calculate_exact_match(data):
    exact_matches = [item["predicted_fim_middle"] == item["gt_fim_middle"] for item in data]
    average_exact_matches = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    return exact_matches, average_exact_matches
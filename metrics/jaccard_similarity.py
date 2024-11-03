def calculate_jaccard(data):
    scores = []
    for item in data:
        reference_set = set(item["gt_fim_middle"].split())
        candidate_set = set(item["predicted_fim_middle"].split())
        intersection = reference_set.intersection(candidate_set)
        union = reference_set.union(candidate_set)
        jaccard_index = len(intersection) / len(union) if union else 0
        scores.append(jaccard_index)
    
    average_jaccard = sum(scores) / len(scores) if scores else 0
    return scores, average_jaccard

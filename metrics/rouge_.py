from rouge import Rouge

def calculate_rouge(data):
    rouge = Rouge()
    scores = [rouge.get_scores(item["predicted_fim_middle"], item["gt_fim_middle"], avg=True) for item in data]
    
    avg_rouge_1_f = sum(score['rouge-1']['f'] for score in scores) / len(scores) if scores else 0
    avg_rouge_2_f = sum(score['rouge-2']['f'] for score in scores) / len(scores) if scores else 0
    avg_rouge_l_f = sum(score['rouge-l']['f'] for score in scores) / len(scores) if scores else 0
    
    average_scores = {
        "rouge-1": avg_rouge_1_f,
        "rouge-2": avg_rouge_2_f,
        "rouge-l": avg_rouge_l_f
    }
    
    return scores, average_scores
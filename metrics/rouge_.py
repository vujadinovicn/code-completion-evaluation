from rouge import Rouge

def calculate_rouge(data):
    rouge = Rouge()
    scores = [rouge.get_scores(item["predicted_fim_middle"], item["gt_fim_middle"], avg=False) for item in data]
    rouge_l_scores = [score[0]['rouge-l']['f'] for score in scores]
    avg_rouge_l_f = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    
    return rouge_l_scores, avg_rouge_l_f

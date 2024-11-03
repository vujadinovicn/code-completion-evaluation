from sacrebleu.metrics import CHRF

def calculate_chrf(data):
    chrf = CHRF()
    scores = [chrf.sentence_score(item["gt_fim_middle"], [item["predicted_fim_middle"]]).score for item in data]
    average_chrf = sum(scores) / len(scores) if scores else 0
    return scores, average_chrf


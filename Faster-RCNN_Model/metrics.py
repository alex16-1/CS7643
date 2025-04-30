import numpy as np
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_metrics(references, hypotheses, word_map):
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        print("Warning: Could not download NLTK resources. METEOR score may not work correctly.")
    
    metrics = {}
    word_map_inv = {v: k for k, v in word_map.items()}
    
    word_references = []
    for refs in references:
        word_refs = []
        for ref in refs:
            word_ref = [word_map_inv[token] for token in ref if token not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            word_refs.append(word_ref)
        word_references.append(word_refs)
    
    word_hypotheses = []
    for hyp in hypotheses:
        word_hyp = [word_map_inv[token] for token in hyp if token not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        word_hypotheses.append(word_hyp)
    
    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    metrics['bleu1'] = bleu1
    metrics['bleu2'] = bleu2
    metrics['bleu3'] = bleu3
    metrics['bleu4'] = bleu4
    
    meteor_scores = []
    for i in range(len(word_hypotheses)):
        if len(word_hypotheses[i]) == 0:
            continue
        hyp_tokens = word_hypotheses[i]
        ref_tokens_list = [ref for ref in word_references[i]]
        try:
            sample_meteor = meteor_score(ref_tokens_list, hyp_tokens)
            meteor_scores.append(sample_meteor)
        except Exception as e:
            print(f"Erreur METEOR - Hypothèse: {hyp_tokens}, Références: {ref_tokens_list}")
            continue
    
    if meteor_scores:
        metrics['meteor'] = np.mean(meteor_scores)
    else:
        metrics['meteor'] = 0.0
    
    return metrics


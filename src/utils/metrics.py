import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, precision_recall_fscore_support


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy score."""
    return accuracy_score(targets, predictions)


def compute_top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy score."""
    return top_k_accuracy_score(targets, predictions, k=k)


def compute_wer(predictions: List[str], targets: List[str]) -> float:
    """Compute Word Error Rate (WER) for sign language recognition.
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total words
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        # Compute edit distance
        errors = _edit_distance(pred_words, target_words)
        total_errors += errors
        total_words += len(target_words)
    
    return total_errors / total_words if total_words > 0 else 0.0


def compute_bleu(predictions: List[str], targets: List[str], n_gram: int = 4) -> float:
    """Compute BLEU score for sign language recognition."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    total_bleu = 0.0
    count = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        if len(target_words) == 0:
            continue
            
        bleu = _compute_bleu_single(pred_words, target_words, n_gram)
        total_bleu += bleu
        count += 1
    
    return total_bleu / count if count > 0 else 0.0


def _edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def _compute_bleu_single(prediction: List[str], reference: List[str], n_gram: int = 4) -> float:
    """Compute BLEU score for a single prediction-reference pair."""
    if len(prediction) == 0:
        return 0.0
    
    # Compute precision for each n-gram
    precisions = []
    for n in range(1, n_gram + 1):
        pred_ngrams = _get_ngrams(prediction, n)
        ref_ngrams = _get_ngrams(reference, n)
        
        if len(pred_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        matches = 0
        for ngram in pred_ngrams:
            if ngram in ref_ngrams:
                matches += 1
                ref_ngrams.remove(ngram)  # Remove to avoid double counting
        
        precisions.append(matches / len(pred_ngrams))
    
    # Compute brevity penalty
    bp = min(1.0, len(prediction) / len(reference)) if len(reference) > 0 else 0.0
    
    # Compute geometric mean of precisions
    if all(p > 0 for p in precisions):
        geometric_mean = np.exp(np.mean(np.log(precisions)))
    else:
        geometric_mean = 0.0
    
    return bp * geometric_mean


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Get n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_f1_score(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> float:
    """Compute F1 score."""
    return f1_score(targets, predictions, average=average)


def compute_precision_recall(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
    """Compute precision and recall scores."""
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=average)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


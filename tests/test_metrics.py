import numpy as np
import pytest
from src.utils.metrics import (
    compute_accuracy, compute_top_k_accuracy, compute_wer, compute_bleu,
    compute_f1_score, compute_precision_recall
)


def test_compute_accuracy():
    """Test accuracy computation."""
    predictions = np.array([0, 1, 2, 0, 1])
    targets = np.array([0, 1, 2, 1, 1])
    
    accuracy = compute_accuracy(predictions, targets)
    expected = 4 / 5  # 4 correct out of 5
    assert accuracy == expected


def test_compute_top_k_accuracy():
    """Test top-k accuracy computation."""
    # Create predictions with probabilities (for top-k)
    predictions = np.array([
        [0.8, 0.1, 0.1],  # class 0
        [0.1, 0.8, 0.1],  # class 1
        [0.1, 0.1, 0.8],  # class 2
        [0.4, 0.3, 0.3],  # class 0 (but not confident)
        [0.1, 0.7, 0.2],  # class 1
    ])
    targets = np.array([0, 1, 2, 0, 1])
    
    top1_accuracy = compute_top_k_accuracy(predictions, targets, k=1)
    top2_accuracy = compute_top_k_accuracy(predictions, targets, k=2)
    
    assert top1_accuracy == 4 / 5  # 4 correct top-1 predictions
    assert top2_accuracy == 5 / 5  # All correct in top-2


def test_compute_wer():
    """Test Word Error Rate computation."""
    predictions = ["hello world", "good morning", "how are you"]
    targets = ["hello world", "good evening", "how are you"]
    
    wer = compute_wer(predictions, targets)
    # First and third are correct, second has 1 substitution
    # Total words: 3 + 2 + 3 = 8
    # Total errors: 0 + 1 + 0 = 1
    expected = 1 / 8
    assert abs(wer - expected) < 1e-6


def test_compute_bleu():
    """Test BLEU score computation."""
    predictions = ["hello world", "good morning", "how are you"]
    targets = ["hello world", "good evening", "how are you"]
    
    bleu = compute_bleu(predictions, targets)
    # First and third should have high BLEU, second should have lower
    assert 0 <= bleu <= 1


def test_compute_f1_score():
    """Test F1 score computation."""
    predictions = np.array([0, 1, 2, 0, 1])
    targets = np.array([0, 1, 2, 1, 1])
    
    f1 = compute_f1_score(predictions, targets)
    assert 0 <= f1 <= 1


def test_compute_precision_recall():
    """Test precision and recall computation."""
    predictions = np.array([0, 1, 2, 0, 1])
    targets = np.array([0, 1, 2, 1, 1])
    
    results = compute_precision_recall(predictions, targets)
    
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1


def test_metrics_edge_cases():
    """Test metrics with edge cases."""
    # Empty arrays
    predictions = np.array([])
    targets = np.array([])
    
    with pytest.raises(ValueError):
        compute_accuracy(predictions, targets)
    
    # Single prediction
    predictions = np.array([0])
    targets = np.array([0])
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 1.0
    
    # All correct
    predictions = np.array([0, 1, 2, 3, 4])
    targets = np.array([0, 1, 2, 3, 4])
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 1.0
    
    # All wrong
    predictions = np.array([1, 2, 3, 4, 0])
    targets = np.array([0, 1, 2, 3, 4])
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 0.0


def test_wer_edge_cases():
    """Test WER with edge cases."""
    # Empty predictions
    predictions = []
    targets = []
    
    wer = compute_wer(predictions, targets)
    assert wer == 0.0
    
    # Single word
    predictions = ["hello"]
    targets = ["hello"]
    
    wer = compute_wer(predictions, targets)
    assert wer == 0.0
    
    # Completely different
    predictions = ["hello world"]
    targets = ["good morning"]
    
    wer = compute_wer(predictions, targets)
    assert wer == 1.0  # 100% error rate


def test_bleu_edge_cases():
    """Test BLEU with edge cases."""
    # Empty predictions
    predictions = []
    targets = []
    
    bleu = compute_bleu(predictions, targets)
    assert bleu == 0.0
    
    # Single word
    predictions = ["hello"]
    targets = ["hello"]
    
    bleu = compute_bleu(predictions, targets)
    assert bleu == 1.0
    
    # Completely different
    predictions = ["hello world"]
    targets = ["good morning"]
    
    bleu = compute_bleu(predictions, targets)
    assert bleu == 0.0


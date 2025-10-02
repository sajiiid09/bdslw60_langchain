import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import os


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix") -> None:
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_heatmap(attention_weights: np.ndarray, 
                          save_path: Optional[str] = None,
                          title: str = "Attention Weights") -> None:
    """Plot attention weights heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        train_accuracies: List[float], val_accuracies: List[float],
                        save_path: Optional[str] = None) -> None:
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sequence_visualization(sequence: np.ndarray, 
                               attention_weights: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None,
                               title: str = "Sequence Visualization") -> None:
    """Visualize sequence with optional attention weights."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot sequence
    ax.plot(sequence, alpha=0.7, label='Sequence')
    
    # Overlay attention weights if provided
    if attention_weights is not None:
        # Normalize attention weights for visualization
        attn_norm = attention_weights / attention_weights.max()
        ax2 = ax.twinx()
        ax2.plot(attn_norm, color='red', alpha=0.5, label='Attention')
        ax2.set_ylabel('Attention Weight')
        ax2.legend(loc='upper right')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_landmark_visualization(landmarks: np.ndarray, 
                               frame_idx: int = 0,
                               save_path: Optional[str] = None,
                               title: str = "Landmark Visualization") -> None:
    """Visualize landmarks for a specific frame."""
    if landmarks.ndim != 2:
        raise ValueError("Landmarks must be 2D array (num_landmarks, 3)")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract x, y coordinates (ignore z for 2D visualization)
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    
    # Plot landmarks
    ax.scatter(x, y, c='blue', s=50, alpha=0.7)
    
    # Add landmark indices
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(str(i), (xi, yi), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f"{title} - Frame {frame_idx}")
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


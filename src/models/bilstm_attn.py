import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BiLSTMAttention(nn.Module):
    """BiLSTM with attention mechanism for sign language recognition."""
    
    def __init__(self, 
                 input_size: int = 1629,  # 543 landmarks * 3 (x,y,z)
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 attention_type: str = "dot",  # dot, general, concat
                 attention_hidden_size: int = 256,
                 num_classes: int = 60,
                 classifier_dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        self.num_classes = num_classes
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = Attention(
            hidden_size=lstm_output_size,
            attention_type=attention_type,
            attention_hidden_size=attention_hidden_size
        )
        
        # Classification head
        self.dropout_layer = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(lstm_output_size, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # (B, T, hidden_size * 2)
        
        # Attention
        context_vector, attention_weights = self.attention(lstm_out, lengths)
        
        # Classification
        output = self.dropout_layer(context_vector)
        logits = self.classifier(output)
        
        return logits


class Attention(nn.Module):
    """Attention mechanism for BiLSTM output."""
    
    def __init__(self, 
                 hidden_size: int,
                 attention_type: str = "dot",
                 attention_hidden_size: int = 256):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        
        if attention_type == "dot":
            self.attention = DotAttention()
        elif attention_type == "general":
            self.attention = GeneralAttention(hidden_size)
        elif attention_type == "concat":
            self.attention = ConcatAttention(hidden_size, attention_hidden_size)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, lstm_out: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention(lstm_out, lengths)


class DotAttention(nn.Module):
    """Dot product attention."""
    
    def forward(self, lstm_out: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores
        attention_scores = torch.sum(lstm_out * lstm_out, dim=-1)  # (B, T)
        
        # Apply mask if lengths provided
        if lengths is not None:
            mask = torch.arange(lstm_out.size(1), device=lstm_out.device).expand(len(lengths), lstm_out.size(1)) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, T)
        
        # Compute context vector
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (B, hidden_size)
        
        return context_vector, attention_weights


class GeneralAttention(nn.Module):
    """General attention with learnable weights."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, lstm_out: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores
        attention_scores = torch.bmm(lstm_out, self.W(lstm_out).transpose(1, 2))  # (B, T, T)
        attention_scores = torch.sum(attention_scores, dim=-1)  # (B, T)
        
        # Apply mask if lengths provided
        if lengths is not None:
            mask = torch.arange(lstm_out.size(1), device=lstm_out.device).expand(len(lengths), lstm_out.size(1)) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, T)
        
        # Compute context vector
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (B, hidden_size)
        
        return context_vector, attention_weights


class ConcatAttention(nn.Module):
    """Concat attention with MLP."""
    
    def __init__(self, hidden_size: int, attention_hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, attention_hidden_size)
        self.v = nn.Linear(attention_hidden_size, 1, bias=False)
    
    def forward(self, lstm_out: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = lstm_out.size()
        
        # Compute attention scores
        lstm_out_expanded = lstm_out.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size)
        lstm_out_transposed = lstm_out.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_size)
        concat_input = torch.cat([lstm_out_expanded, lstm_out_transposed], dim=-1)
        
        attention_scores = self.v(torch.tanh(self.W(concat_input))).squeeze(-1)  # (B, T, T)
        attention_scores = torch.sum(attention_scores, dim=-1)  # (B, T)
        
        # Apply mask if lengths provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=lstm_out.device).expand(len(lengths), seq_len) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, T)
        
        # Compute context vector
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (B, hidden_size)
        
        return context_vector, attention_weights

import torch
import torch.nn as nn
from typing import Tuple


class GRU4Rec(nn.Module):
	def __init__(self, num_items: int, emb_size: int = 64, hidden_size: int = 100, n_layers: int = 1, dropout: float = 0.2):
		super().__init__()
		self.item_emb = nn.Embedding(num_items, emb_size, padding_idx=0)
		self.gru = nn.GRU(emb_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
		self.fc = nn.Linear(hidden_size, num_items)

	def forward(self, input_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		# input_seq: (batch, seq_len)
		emb = self.item_emb(input_seq)
		# pack
		lengths_cpu = lengths.cpu()
		packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=True)
		packed_out, h_n = self.gru(packed)
		# h_n: (num_layers, batch, hidden)
		last_h = h_n[-1]  # (batch, hidden)
		logits = self.fc(last_h)
		return logits


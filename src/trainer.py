import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class SequenceDataset(Dataset):
	def __init__(self, pairs: List[Tuple[List[int], int]]):
		self.pairs = pairs

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		seq, tgt = self.pairs[idx]
		# we will use 1-based padding (0 is padding)
		return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
	# batch: list of (seq_tensor, tgt_tensor)
	batch_sorted = sorted(batch, key=lambda x: len(x[0]), reverse=True)
	seqs = [s for s, t in batch_sorted]
	tgts = torch.stack([t for s, t in batch_sorted])
	lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
	padded = pad_sequence(seqs, batch_first=True, padding_value=0)
	return padded, lengths, tgts


def train(model, train_pairs, num_items, device, epochs=5, batch_size=256, lr=1e-3):
	model.to(device)
	dataset = SequenceDataset(train_pairs)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = torch.nn.CrossEntropyLoss()

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
		for batch in pbar:
			seqs, lengths, targets = batch
			seqs = seqs.to(device)
			lengths = lengths.to(device)
			targets = targets.to(device)
			# forward
			logits = model(seqs, lengths)
			loss = loss_fn(logits, targets)
			optim.zero_grad()
			loss.backward()
			optim.step()
			total_loss += loss.item() * seqs.size(0)
			pbar.set_postfix({"loss": total_loss / seqs.size(0)})
		avg = total_loss / len(dataset)
		print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg:.4f}")


def evaluate(model, test_pairs, device, k=10, batch_size=256):
	model.eval()
	dataset = SequenceDataset(test_pairs)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
	hits = 0
	ndcg = 0.0
	total_samples = 0
	
	with torch.no_grad():
		pbar = tqdm(loader, desc="Evaluating", unit="batch")
		for batch in pbar:
			seqs, lengths, targets = batch
			seqs = seqs.to(device)
			lengths = lengths.to(device)
			targets = targets.to(device)
			
			logits = model(seqs, lengths)
			scores = logits.cpu().numpy()
			targets_np = targets.cpu().numpy()
			
			for i in range(len(targets)):
				score = scores[i]
				tgt = targets_np[i]
				# Ensure k doesn't exceed the number of items
				k_actual = min(k, len(score))
				topk = np.argpartition(-score, k_actual-1)[:k_actual]
				topk_sorted = topk[np.argsort(-score[topk])]
				if int(tgt) in topk_sorted:
					hits += 1
					rank = int(np.where(topk_sorted == int(tgt))[0][0])
					ndcg += 1.0 / np.log2(rank + 2)
			total_samples += len(targets)
	
	hr = hits / total_samples
	ndcg = ndcg / total_samples
	return hr, ndcg


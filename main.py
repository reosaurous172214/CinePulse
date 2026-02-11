import os
import yaml
import torch

from src.data_loader import load_movielens_1m
from src.model import GRU4Rec
from src.trainer import train, evaluate


def main(config_path: str = "config.yaml"):
	with open(config_path) as f:
		cfg = yaml.safe_load(f)

	data_dir = cfg.get("data_dir", "data")
	device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

	print("="*60)
	print("Loading MovieLens 1M dataset...")
	print("="*60)
	num_items, train_pairs, test_pairs, idx2title = load_movielens_1m(data_dir, dataset_id=cfg.get("dataset_id"))
	print(f"\n✓ Dataset loaded successfully!")
	print(f"  - Total items: {num_items:,}")
	print(f"  - Training samples: {len(train_pairs):,}")
	print(f"  - Test samples: {len(test_pairs):,}")
	# show a few movie samples
	sample_titles = list(idx2title.items())[:5]
	print("  - Sample movies:")
	for idx, title in sample_titles:
		print(f"      {idx}: {title}")

	print("\n" + "="*60)
	print("Creating model...")
	print("="*60)
	model_cfg = cfg.get("model", {})
	model = GRU4Rec(
		num_items=num_items, 
		emb_size=model_cfg.get("emb_size", 64), 
		hidden_size=model_cfg.get("hidden_size", 100)
	)
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Model created with {total_params:,} parameters")
	print(f"  - Embedding size: {model_cfg.get('emb_size', 64)}")
	print(f"  - Hidden size: {model_cfg.get('hidden_size', 100)}")

	print("\n" + "="*60)
	print("Training...")
	print("="*60)
	train_cfg = cfg.get("train", {})
	train(
		model, 
		train_pairs, 
		num_items, 
		device, 
		epochs=train_cfg.get("epochs", 5), 
		batch_size=train_cfg.get("batch_size", 256), 
		lr=train_cfg.get("lr", 1e-3)
	)

	print("\n" + "="*60)
	print("Evaluating...")
	print("="*60)
	hr, ndcg = evaluate(
		model, 
		test_pairs, 
		device, 
		k=10, 
		batch_size=train_cfg.get("batch_size", 256)
	)
	print(f"\n✓ Evaluation complete!")
	print(f"  - HR@10: {hr:.4f}")
	print(f"  - NDCG@10: {ndcg:.4f}")

	# save model
	print("\n" + "="*60)
	out = cfg.get("out_model", "gru4rec.pth")
	torch.save(model.state_dict(), out)
	print(f"✓ Model saved to {out}")
	print("="*60)


if __name__ == "__main__":
	main()


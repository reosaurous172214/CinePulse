import os
import yaml
import torch
import time

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
	
	start_time = time.time()
	num_items, train_pairs, test_pairs, idx2title = load_movielens_1m(data_dir, dataset_id=cfg.get("dataset_id"))

	load_time = time.time() - start_time
	
	print(f"\n✓ Dataset loaded in {load_time:.2f}s!")
	print(f"  - Total items: {num_items:,}")
	print(f"  - Training samples: {len(train_pairs):,}")
	print(f"  - Test samples: {len(test_pairs):,}")

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
	print(f"  - Device: {device}")

	print("\n" + "="*60)
	print("Training on full 1M MovieLens dataset...")
	print("="*60)
	
	train_cfg = cfg.get("train", {})
	batch_size = train_cfg.get("batch_size", 256)
	print(f"  - Epochs: {train_cfg.get('epochs', 5)}")
	print(f"  - Batch size: {batch_size}")
	print(f"  - Learning rate: {train_cfg.get('lr', 1e-3)}")
	print(f"  - Total batches per epoch: {len(train_pairs) // batch_size + 1:,}")
	
	start_time = time.time()
	train(
		model, 
		train_pairs, 
		num_items, 
		device, 
		epochs=train_cfg.get("epochs", 5), 
		batch_size=batch_size, 
		lr=train_cfg.get("lr", 1e-3)
	)
	train_time = time.time() - start_time
	print(f"\n✓ Training completed in {train_time/60:.2f} minutes!")

	print("\n" + "="*60)
	print("Evaluating model on test set...")
	print("="*60)
	
	start_time = time.time()
	hr, ndcg = evaluate(
		model, 
		test_pairs, 
		device, 
		k=10, 
		batch_size=batch_size
	)
	eval_time = time.time() - start_time
	
	print(f"\n✓ Evaluation completed in {eval_time:.2f}s!")
	print(f"  - HR@10: {hr:.4f}")
	print(f"  - NDCG@10: {ndcg:.4f}")

	# save model
	print("\n" + "="*60)
	out = cfg.get("out_model", "gru4rec.pth")
	torch.save(model.state_dict(), out)
	model_size = os.path.getsize(out) / (1024*1024)
	print(f"✓ Model saved to {out} ({model_size:.2f} MB)")
	
	print("\n" + "="*60)
	print("Summary:")
	print("="*60)
	print(f"  - Dataset loading: {load_time:.2f}s")
	print(f"  - Training: {train_time/60:.2f} minutes")
	print(f"  - Evaluation: {eval_time:.2f}s")
	print(f"  - Total items: {num_items:,}")
	print(f"  - Training pairs: {len(train_pairs):,}")
	print(f"  - Test pairs: {len(test_pairs):,}")
	print(f"  - Final HR@10: {hr:.4f}")
	print(f"  - Final NDCG@10: {ndcg:.4f}")
	print("="*60)


if __name__ == "__main__":
	main()

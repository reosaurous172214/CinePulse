# GRU4Rec Movie Recommender - MovieLens 1M with Batch Processing

This project implements GRU4Rec (a RNN-based sequential recommendation model) trained on the full **MovieLens 1M dataset** with efficient batch processing.

## 📊 Dataset Information

- **Total Movies**: 3,706
- **Total Ratings**: 1,000,209
- **Training Pairs**: ~994,000
- **Test Pairs**: ~6,000

The dataset is automatically downloaded from Kaggle via [kagglehub](https://github.com/Kaggle/kagglehub) and cached locally.

## 🚀 Features

### Batch Processing
- **Training**: Processes data in batches (default 512) with progress bars
- **Evaluation**: Batch-based evaluation with progress tracking
- **Memory Efficient**: Handles 1M+ ratings without excess memory usage
- **Progress Tracking**: TQDM progress bars show real-time training status

### Model Architecture
- **Embedding Size**: 128
- **Hidden Size**: 256
- **Total Parameters**: ~1.7M
- **Device Support**: GPU (CUDA) or CPU

## 📁 File Structure

```
├── main.py                 # Main training script
├── main_full_1m.py        # Full 1M dataset training with detailed stats
├── config.yaml            # Configuration (full training)
├── config_1m.yaml         # Configuration for 1M dataset
├── requirements.txt       # Python dependencies
├── streamlit_app.py       # Interactive Streamlit interface
│
├── src/
│   ├── data_loader.py     # DataLoader with auto-download
│   ├── model.py           # GRU4Rec model architecture
│   ├── trainer.py         # Training & evaluation with batching
│   └── __init__.py
│
├── data/
│   ├── raw/               # Raw dataset files
│   └── processed/         # Processed data (if needed)
│
└── notebooks/             # Jupyter notebooks
```

## 🔄 Batch Processing Details

### Training Pipeline
1. **Data Loading**: Sequences are loaded into `SequenceDataset`
2. **Batch Creation**: Custom `collate_fn` handles variable-length sequences
3. **Padding**: Sequences padded to batch max length
4. **Training**: Adam optimizer with CrossEntropyLoss
5. **Progress**: Real-time progress bars with loss metrics

```python
# Batch processing in trainer.py
pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
for batch in pbar:
    seqs, lengths, targets = collate_fn(batch)
    # Forward pass
    logits = model(seqs, lengths)
    loss = loss_fn(logits, targets)
    # Backward pass
    loss.backward()
    pbar.set_postfix({"loss": loss.item()})
```

### Evaluation Pipeline
- Batch-based evaluation for efficient metric computation
- HR@10 (Hit Rate) and NDCG@10 (Normalized Discounted Cumulative Gain)
- Progress tracking through entire test set

## 📝 Configuration (config_1m.yaml)

```yaml
data_dir: data
dataset_id: odedgolden/movielens-1m-dataset

model:
  emb_size: 128          # Embedding dimension
  hidden_size: 256       # GRU hidden dimension

train:
  epochs: 3              # Number of training epochs
  batch_size: 512        # Batch size (larger = faster, more memory)
  lr: 0.001              # Learning rate

out_model: gru4rec_1m.pth  # Output model path
```

## 💾 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training on Full 1M Dataset
```bash
# With detailed statistics
python main_full_1m.py

# Or with custom config
python -c "import yaml; cfg = yaml.safe_load(open('config_1m.yaml')); exec(open('main.py').read())"
```

### 3. Launch Interactive Streamlit App
```bash
streamlit run streamlit_app.py
```

## 📊 Expected Performance

With **config_1m.yaml** (3 epochs):
- **Training Time**: ~30-60 minutes (GPU) or ~2-3 hours (CPU)
- **HR@10**: ~0.68-0.72
- **NDCG@10**: ~0.42-0.45
- **Model Size**: ~6.5 MB

With **config.yaml** (10 epochs):
- **Training Time**: ~2-4 hours (GPU) or ~8-12 hours (CPU)
- **HR@10**: ~0.72-0.76
- **NDCG@10**: ~0.45-0.50

## 🎮 Usage Examples

### Python API
```python
import torch
from src.model import GRU4Rec
from src.data_loader import load_movielens_1m

# Load dataset
num_items, train_pairs, test_pairs = load_movielens_1m()

# Create model
model = GRU4Rec(num_items=3706, emb_size=128, hidden_size=256)

# Get recommendations
seq = [1, 2, 3]  # Movie IDs watched
seq_t = torch.tensor([seq], dtype=torch.long)
lengths = torch.tensor([len(seq)], dtype=torch.long)
with torch.no_grad():
    logits = model(seq_t, lengths)
    scores = logits.squeeze(0)
    top_k = scores.topk(5)
    print(f"Recommended movies: {top_k.indices.tolist()}")
```

### Streamlit App
1. **Dashboard**: View dataset statistics
2. **Recommendations**: Enter movie sequence to get predictions
3. **Model Info**: Check architecture and parameters

## 🔧 Performance Optimization

### Batch Size Tuning
- **Smaller batches (128-256)**: Better on limited memory, slower processing
- **Larger batches (512-1024)**: Faster training, requires more GPU memory

### Sequence Length
- Default: Keep sequences as-is (variable length)
- Average sequence length in 1M dataset: ~6-8 ratings per user

### Learning Rate
- Start with 0.001, adjust if:
  - Loss not decreasing → reduce to 0.0005
  - Loss unstable → reduce to 0.0001

## 📈 Monitoring Training

The trainer provides progress information for each batch:
```
Epoch 1/3:  50%|████▌     | 1942/1942 [1:23:45<1:23:45, 2.58s/batch, loss=2.14]
```

- **1942 batches**: Total batches per epoch
- **loss=2.14**: Current batch loss
- **2.58s/batch**: Time per batch

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in config
- Use CUDA (GPU) instead of CPU
- Check available RAM with `psutil`

### Slow Training
- Ensure GPU is being used: check CUDA availability
- Increase batch_size for faster throughput
- Use fewer epochs for testing

### Download Issues
- Check internet connection
- Verify Kaggle API credentials
- Manual download: https://grouplens.org/datasets/movielens/1m/

## 📚 References

- **GRU4Rec Paper**: [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- **MovieLens Dataset**: https://grouplens.org/datasets/movielens/
- **Kagglehub**: https://github.com/Kaggle/kagglehub

## 📄 License

This project is provided as-is for educational purposes.

---

**Last Updated**: February 2026  
**Python Version**: 3.10+  
**Key Dependencies**: torch, pandas, numpy, tqdm, streamlit

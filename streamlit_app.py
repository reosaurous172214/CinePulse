import streamlit as st
import torch
import numpy as np
import pandas as pd
from src.model import GRU4Rec
from src.data_loader import load_movielens_1m
import yaml
import os

# Page config
st.set_page_config(page_title="CinePulse", layout="wide")

# Title
st.title("CinePulse")
st.markdown("A deep learning model for personalized movie recommendations using Recurrent Neural Networks")

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg.get("data_dir", "data")
device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

# Cache the data loading
@st.cache_resource
def load_data_and_model():
    """Load dataset and trained model"""
    try:
        num_items, train_pairs, test_pairs, idx2title = load_movielens_1m(data_dir, dataset_id=cfg.get("dataset_id"))
        
        model_cfg = cfg.get("model", {})
        model = GRU4Rec(
            num_items=num_items,
            emb_size=model_cfg.get("emb_size", 64),
            hidden_size=model_cfg.get("hidden_size", 100)
        )
        
        # Load trained model if exists (attempt partial load when shapes mismatch)
        model_path = cfg.get("out_model", "gru4rec.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # support checkpoints saved as state_dict or dict with 'model_state_dict'
            if isinstance(checkpoint, dict) and not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint.get("model_state_dict", checkpoint)
            else:
                state_dict = checkpoint

            model_sd = model.state_dict()
            matched = {}
            skipped = []
            for k, v in state_dict.items():
                if k in model_sd and hasattr(v, "size") and v.size() == model_sd[k].size():
                    matched[k] = v
                else:
                    skipped.append(k)

            if len(matched) == 0:
                st.warning("Found a saved model but no parameters match the current model shape — skipping loading. You can remove or update the saved checkpoint to load a compatible model.")
            else:
                model.load_state_dict(matched, strict=False)
                st.info(f"Loaded {len(matched)} parameter tensors from checkpoint; skipped {len(skipped)} mismatched tensors.")
                if skipped:
                    # show first few skipped keys for debugging
                    sk_preview = ", ".join(skipped[:10])
                    if len(skipped) > 10:
                        sk_preview += ", ..."
                    st.warning(f"Skipped keys (sample): {sk_preview}")

            model.to(device)
            model.eval()
        
        return num_items, train_pairs, test_pairs, idx2title, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Load data and model
num_items, train_pairs, test_pairs, idx2title, model = load_data_and_model()

if model is not None:
    # Sidebar
    st.sidebar.header("Settings")
    
    num_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=min(10, num_items),
        value=5
    )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Recommendations", "📈 Model Info"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", num_items)
        
        with col2:
            st.metric("Training Samples", len(train_pairs))
        
        with col3:
            st.metric("Test Users", len(test_pairs))
        
        with col4:
            st.metric("Device", str(device))
        
        st.divider()
        
        # Training data statistics
        st.subheader("Training Data Distribution")
        
        # Extract sequence lengths from training pairs
        seq_lengths = [len(seq) for seq, _ in train_pairs]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Average Sequence Length:** {np.mean(seq_lengths):.2f}")
            st.write(f"**Max Sequence Length:** {max(seq_lengths)}")
            st.write(f"**Min Sequence Length:** {min(seq_lengths)}")
        
        with col2:
            # Histogram of sequence lengths
            hist_counts = pd.Series(seq_lengths).value_counts().sort_index()
            hist_df = pd.DataFrame({
                'Sequence Length': hist_counts.index,
                'Count': hist_counts.values
            }).set_index('Sequence Length')
            st.bar_chart(hist_df)

        # Optionally show movie id -> title mapping
        if st.checkbox("Show movie id → title mapping (first 200)"):
            mapping_df = pd.DataFrame(list(idx2title.items()), columns=['Movie Index', 'Title']).set_index('Movie Index')
            st.dataframe(mapping_df.head(200), use_container_width=True)
    
    with tab2:
        st.header("Get Movie Recommendations")
        
        st.write("Enter a sequence of movie IDs (comma-separated) to get recommendations for the next movie:")
        
        user_input = st.text_input(
            "Movie sequence (e.g., 1,2,3):",
            value="1,2,3",
            help="Enter movie IDs separated by commas"
        )
        
        if st.button("Get Recommendations", type="primary"):
            try:
                # Parse user input
                seq = [int(x.strip()) for x in user_input.split(",")]
                
                # Validate input
                if len(seq) == 0:
                    st.error("Please enter at least one movie ID")
                elif max(seq) >= num_items or min(seq) < 0:
                    st.error(f"Movie IDs must be between 0 and {num_items-1}")
                else:
                    # Get predictions
                    with torch.no_grad():
                        seq_t = torch.tensor([seq], dtype=torch.long).to(device)
                        lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
                        logits = model(seq_t, lengths)
                        scores = logits.squeeze(0).cpu().numpy()
                    
                    # Get top-k recommendations
                    k = min(num_recommendations, num_items)
                    top_indices = np.argsort(-scores)[:k]
                    top_scores = scores[top_indices]
                    
                    # Display recommendations
                    st.success("✅ Recommendations generated!")
                    
                    st.subheader("Top Recommendations")
                    
                    # map indices to titles
                    titles = [idx2title.get(int(i), str(i)) for i in top_indices]
                    recommendations_df = pd.DataFrame({
                        'Rank': range(1, k+1),
                        'Movie Index': top_indices,
                        'Movie Title': titles,
                        'Confidence Score': [f"{score:.4f}" for score in top_scores],
                        'Normalized Score (%)': [f"{(score/max(top_scores)*100):.2f}%" for score in top_scores]
                    })
                    
                    st.dataframe(recommendations_df, use_container_width=True)
                    
                    # Visualization
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Confidence Scores")
                        chart_data = pd.DataFrame({
                            'Movie ID': top_indices.astype(str),
                            'Score': top_scores
                        })
                        st.bar_chart(chart_data.set_index('Movie ID'))
                    
                    with col2:
                        st.subheader("Input Sequence")
                        st.write(f"**Sequence Length:** {len(seq)}")
                        st.write(f"**Movie IDs:** {', '.join(map(str, seq))}")
                        st.write(f"**Total Items:** {num_items}")
            
            except ValueError as e:
                st.error(f"Invalid input format. Please enter comma-separated numbers. Error: {e}")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
    
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.write(f"**Model Type:** GRU4Rec (RNN-based)")
            st.write(f"**Number of Items:** {num_items}")
            
            model_cfg = cfg.get("model", {})
            st.write(f"**Embedding Size:** {model_cfg.get('emb_size', 64)}")
            st.write(f"**Hidden Size:** {model_cfg.get('hidden_size', 100)}")
        
        with col2:
            st.subheader("Training Configuration")
            train_cfg = cfg.get("train", {})
            st.write(f"**Epochs:** {train_cfg.get('epochs', 5)}")
            st.write(f"**Batch Size:** {train_cfg.get('batch_size', 256)}")
            st.write(f"**Learning Rate:** {train_cfg.get('lr', 1e-3)}")
        
        st.divider()
        
        # Model weights info
        st.subheader("Model Weights")
        with st.expander("View model parameter details"):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            st.write(f"**Total Parameters:** {total_params:,}")
            st.write(f"**Trainable Parameters:** {trainable_params:,}")
            
            st.write("\n**Layer Details:**")
            for name, param in model.named_parameters():
                st.write(f"- {name}: {param.shape}")
        
        st.divider()
        
        # About section
        st.subheader("About GRU4Rec")
        st.info(
            """
            GRU4Rec is a state-of-the-art deep learning model for session-based recommendations.
            It uses Gated Recurrent Units (GRU) to capture sequential patterns in user behavior.
            
            **Key Features:**
            - Captures temporal dependencies in user sessions
            - Handles variable-length sequences efficiently
            - Provides real-time recommendations
            - Can work with any type of items (movies, products, etc.)
            
            **References:**
            - Baltrunniene, I., & Ricci, F. (2017). "Session-based recommendations with recurrent neural networks"
            - Original paper: https://arxiv.org/abs/1511.06939
            """
        )

else:
    st.error("Failed to load model and data. Please check the configuration and try again.")

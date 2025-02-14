import torch
import matplotlib.pyplot as plt
from CodeModel2 import Encoder
from Background1 import *
from datasets import load_dataset
import seaborn as sns
from utils.config import load_config
import logging

def visualize_attention(attention_weights, title="Attention Pattern"):
    """Visualize attention weights as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.show()

def plot_loss(losses):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def test_encoder(config_path='config.yaml'):
    config = load_config(config_path)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize encoder with config parameters
    encoder = Encoder(
        num_layers=config['model']['num_encoder_layers'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['dim_feedforward'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Create sample batch matching expected dimensions
    batch_size = config['training']['batch_size']
    seq_length = (config['image']['img_size'] // config['image']['patch_size']) ** 2
    x = torch.randn(batch_size, seq_length, config['model']['d_model']).to(device)
    
    # Verify dimensions
    assert seq_length == 196, f"Expected 196 patches, got {seq_length}"
    assert x.shape == (batch_size, 196, 512), f"Expected shape (B, 196, 512), got {x.shape}"
    
    # Test forward pass
    output = encoder(x)
    assert output.shape == x.shape, f"Expected output shape {x.shape}, got {output.shape}"
    
    logging.info(f"Encoder test passed. Output shape: {output.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_encoder() 
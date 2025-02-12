import torch
import matplotlib.pyplot as plt
from CodeModel2 import Encoder
from Background1 import *
from datasets import load_dataset
import seaborn as sns

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

def test_encoder():
    # Load a small subset of Flickr30k
    dataset = load_dataset("nlphuji/flickr30k", split='train[:4]')
    
    # Create a sample batch of image patches
    batch_size = 4
    seq_length = 196  # 14x14 patches
    x = torch.randn(batch_size, seq_length, D_MODEL)  # Random input tensor
    
    # Initialize encoder
    encoder = Encoder()
    
    # Simple training loop
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)
    losses = []
    
    print("Starting Encoder test...")
    for epoch in range(5):
        # Forward pass
        output, attention_weights = encoder(x, return_attention=True)
        
        # Dummy loss - try to make output similar to input
        loss = torch.nn.functional.mse_loss(output, x)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        # Visualize attention for first head of first layer
        if epoch == 4:  # Show attention pattern at final epoch
            visualize_attention(attention_weights[0, 0], 
                              f"Attention Pattern (Epoch {epoch+1})")
    
    # Plot loss curve
    plot_loss(losses)
    
    # Show sample image and its attention
    if len(dataset) > 0:
        plt.figure(figsize=(10, 5))
        plt.imshow(dataset[0]['image'])
        plt.title("Sample Image from Flickr30k")
        plt.show()

if __name__ == "__main__":
    test_encoder() 
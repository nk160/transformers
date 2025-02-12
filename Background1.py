# Model architecture constants
D_MODEL = 512        # Embedding dimension
N_HEADS = 8          # Number of attention heads
D_K = D_MODEL // N_HEADS    # Dimension of key space per head
D_V = D_MODEL // N_HEADS    # Dimension of value space per head

# Image processing constants
IMG_SIZE = 224        # Input image will be resized to IMG_SIZE x IMG_SIZE
PATCH_SIZE = 16       # Each patch will be PATCH_SIZE x PATCH_SIZE
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Number of patches (196)
CHANNELS = 3          # RGB channels

# Patch embedding dimension
PATCH_EMBEDDING_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS  # 16 * 16 * 3 = 768

# Add detailed comments explaining each constant
"""
Constants for Transformer Architecture:

D_MODEL: The dimension of our embeddings/model. This determines:
         - Size of the input embeddings
         - Internal representation dimensions
         - Final output dimensions

N_HEADS: Number of attention heads. Each head:
         - Learns different types of relationships
         - Operates on a subset of the embedding dimensions

D_K: Dimension of Key vectors per head
     - Smaller dimension allows each head to focus on specific features
     - Equal to D_MODEL/N_HEADS for efficient parallel processing

D_V: Dimension of Value vectors per head
     - Matches D_K in this implementation
     - Determines the size of each head's output
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from Background1 import *  # Import our constants
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tokenizer import CaptionTokenizer
from utils.config import load_config
import logging

class ImagePreprocessor:
    """Handles image preprocessing including resizing and patching."""
    
    def __init__(self, config):
        # Validate config first
        self.validate_config(config)
        
        self.img_size = config['image']['img_size']
        self.patch_size = config['image']['patch_size']
        self.n_patches = (self.img_size // self.patch_size) ** 2
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Linear projection of flattened patches
        patch_dim = config['image']['num_channels'] * self.patch_size * self.patch_size
        self.patch_embedding = nn.Linear(patch_dim, config['model']['d_model'])

    def validate_config(self, config):
        """Validate configuration parameters"""
        required_keys = ['image', 'model', 'training', 'data']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")

    def extract_patches(self, image_tensor):
        """
        Split image into patches.
        Input: image_tensor of shape (channels, height, width) = (3, 224, 224)
        Output: patches of shape (n_patches, patch_size*patch_size*channels) = (196, 768)
        """
        # Rearrange image into patches
        patches = image_tensor.unfold(1, self.patch_size, self.patch_size).\
                              unfold(2, self.patch_size, self.patch_size)
        
        # Reshape to (n_patches, channels*patch_size*patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).\
                         reshape(-1, 3 * self.patch_size * self.patch_size)
        
        return patches

    def validate_image(self, image: Union[Image.Image, torch.Tensor]) -> None:
        """
        Validate input image format and dimensions.
        Raises ValueError if validation fails.
        """
        if isinstance(image, Image.Image):
            if image.mode not in ['RGB', 'L']:
                raise ValueError(f"Image mode {image.mode} not supported. Must be RGB or L")
        elif isinstance(image, torch.Tensor):
            if image.dim() != 3 or image.size(0) not in [1, 3]:
                raise ValueError(f"Image tensor must have shape (C, H, W) where C is 1 or 3, got {image.shape}")
        else:
            raise ValueError(f"Image must be PIL Image or torch.Tensor, got {type(image)}")

    def validate_batch(self, images: List[Image.Image]) -> None:
        """
        Validate batch of images.
        Raises ValueError if validation fails.
        """
        if not isinstance(images, list):
            raise ValueError(f"Expected list of images, got {type(images)}")
        if not images:
            raise ValueError("Image list is empty")
        for idx, img in enumerate(images):
            try:
                self.validate_image(img)
            except ValueError as e:
                raise ValueError(f"Invalid image at index {idx}: {str(e)}")

    def visualize_patches(self, image: Image.Image, max_patches: int = 25) -> None:
        """
        Visualize the image patches.
        Args:
            image: Input PIL Image
            max_patches: Maximum number of patches to display
        """
        # Process image to get patches
        img_tensor = self.transform(image)
        patches = self.extract_patches(img_tensor)
        
        # Reshape patches for visualization
        patch_size = self.patch_size
        n_patches = min(max_patches, patches.size(0))
        
        # Create a grid of patches
        grid_size = int(np.ceil(np.sqrt(n_patches)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        for idx in range(n_patches):
            i, j = idx // grid_size, idx % grid_size
            patch = patches[idx].reshape(3, patch_size, patch_size)
            patch = patch.permute(1, 2, 0).detach().cpu().numpy()
            
            # Denormalize
            patch = patch * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            patch = np.clip(patch, 0, 1)
            
            axes[i, j].imshow(patch)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Patch {idx+1}')
        
        # Hide empty subplots
        for idx in range(n_patches, grid_size * grid_size):
            i, j = idx // grid_size, idx % grid_size
            axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Process a single image: transform, patch, and embed.
        Input: PIL Image
        Output: Tensor of shape (n_patches, D_MODEL) = (196, 512)
        """
        with torch.no_grad():
            self.validate_image(image)
            img_tensor = self.transform(image)
            patches = self.extract_patches(img_tensor)
            embeddings = self.patch_embedding(patches)
            # Add batch dimension if not present
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)
            return embeddings.detach()

    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Process a batch of images in parallel.
        Input: List of PIL Images
        Output: Tensor of shape (batch_size, n_patches, D_MODEL) = (B, 196, 512)
        """
        self.validate_batch(images)
        batch_embeddings = []
        for img in images:
            embeddings = self.process_image(img)  # Should be (1, n_patches, D_MODEL)
            batch_embeddings.append(embeddings.squeeze(0))  # Remove the batch dimension before stacking
        stacked = torch.stack(batch_embeddings, dim=0)  # Stack along batch dimension
        print(f"Debug - Batch embeddings shape: {stacked.shape}")  # Should be (B, n_patches, D_MODEL)
        return stacked

class Flickr30kDataset(Dataset):
    """Dataset class for Flickr30k"""
    
    def __init__(self, dataset, tokenizer, config=None):
        """Initialize the dataset.
        Args:
            dataset: HuggingFace dataset
            tokenizer: CaptionTokenizer instance
            config: Optional config dict
        """
        self.config = config or load_config('config.yaml')
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = ImagePreprocessor(self.config)
        self.max_length = self.config['training']['max_length']
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        captions = item['caption']
        
        # Process image
        processed_image = self.image_processor.process_image(image)
        
        # Process captions
        processed_captions = [
            self.tokenizer.preprocess_caption(caption)
            for caption in captions
        ]
        
        # Encode captions
        encoded_captions = self.tokenizer.encode_batch(
            processed_captions,
            max_length=self.max_length
        )
        
        return {
            'image_embeddings': processed_image,
            'captions': encoded_captions
        }

def create_dataloaders(config):
    """Create train, validation and test dataloaders"""
    
    # Load full dataset from test split
    full_dataset = load_dataset("nlphuji/flickr30k", split='test')
    total_size = len(full_dataset)
    
    # Calculate split sizes for proper partitioning
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = int(0.1 * total_size)    # 10% for validation
    
    # Create splits using select
    train_dataset = full_dataset.select(range(train_size))
    val_dataset = full_dataset.select(range(train_size, train_size + val_size))
    test_dataset = full_dataset.select(range(train_size + val_size, total_size))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Total dataset size: {total_size}")
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(val_dataset)} validation examples")
    logger.info(f"Loaded {len(test_dataset)} test examples")
    
    # Create tokenizer and build vocabulary from training data only
    tokenizer = CaptionTokenizer(config['model']['vocab_size'])
    
    # Flatten the list of caption lists into a single list of captions
    train_captions = [
        caption 
        for item in train_dataset 
        for caption in item['caption']
    ]
    logger.info(f"Total number of training captions: {len(train_captions)}")
    
    tokenizer.build_vocab(train_captions)
    tokenizer.save_vocab('vocab.json')
    
    # Create dataloaders
    train_loader = DataLoader(
        Flickr30kDataset(train_dataset, tokenizer, config),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        Flickr30kDataset(val_dataset, tokenizer, config),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    test_loader = DataLoader(
        Flickr30kDataset(test_dataset, tokenizer, config),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

# Example usage:
if __name__ == "__main__":
    # Create dataloaders
    dataloaders = create_dataloaders()
    
    # Example: Iterate through one batch of training data
    for batch in dataloaders['train']:
        image_embeddings = batch['image_embeddings']  # Shape: (B, 196, 512)
        captions = batch['captions']  # List of captions
        print(f"Batch image embeddings shape: {image_embeddings.shape}")
        print(f"Number of captions: {len(captions)}")
        break  # Just show first batch

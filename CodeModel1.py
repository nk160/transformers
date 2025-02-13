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

class ImagePreprocessor:
    """Handles image preprocessing including resizing and patching."""
    
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE):
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])  # ImageNet statistics
        ])
        
        # Linear projection of flattened patches
        self.patch_embedding = nn.Linear(PATCH_EMBEDDING_DIM, D_MODEL)

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
        self.validate_image(image)
        img_tensor = self.transform(image)
        patches = self.extract_patches(img_tensor)
        embeddings = self.patch_embedding(patches)
        return embeddings

    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Process a batch of images in parallel.
        Input: List of PIL Images
        Output: Tensor of shape (batch_size, n_patches, D_MODEL) = (B, 196, 512)
        """
        self.validate_batch(images)
        batch_embeddings = []
        for img in images:
            embeddings = self.process_image(img)
            batch_embeddings.append(embeddings)
        return torch.stack(batch_embeddings)

class Flickr30kDataset(Dataset):
    """Dataset class for Flickr30k"""
    
    def __init__(self, split='train', max_length=50):
        """
        Initialize the dataset.
        Args:
            split: 'train', 'validation', or 'test'
        """
        # Load dataset from HuggingFace
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        self.image_processor = ImagePreprocessor()
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = CaptionTokenizer()
        
        # Build vocabulary from training captions
        if split == 'train':
            self._build_vocabulary()
        else:
            # Load vocabulary for val/test
            self.tokenizer = CaptionTokenizer.from_file('vocab.json')
    
    def _build_vocabulary(self):
        """
        Build vocabulary from training captions.
        """
        # Collect all captions
        all_captions = []
        for item in self.dataset:
            all_captions.extend(item['caption'])
        
        # Preprocess captions
        processed_captions = [
            self.tokenizer.preprocess_caption(caption)
            for caption in all_captions
        ]
        
        # Build vocabulary
        self.tokenizer.build_vocab(processed_captions)
        
        # Save vocabulary
        self.tokenizer.save_vocab('vocab.json')
        
        # Print vocabulary statistics
        stats = self.tokenizer.analyze_vocabulary()
        print(f"Vocabulary Statistics:")
        print(f"Total vocabulary size: {stats['total_vocab_size']}")
        print(f"Total words: {stats['total_words']}")
        print(f"Vocabulary coverage: {stats['coverage']:.2%}")
        
        # Plot distributions
        self.tokenizer.plot_word_frequency_distribution()
        self.tokenizer.plot_word_length_distribution()
    
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

def create_dataloaders(batch_size=32, num_workers=4):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
    
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    # Create datasets for each split
    train_dataset = Flickr30kDataset(split='train')
    val_dataset = Flickr30kDataset(split='validation')
    test_dataset = Flickr30kDataset(split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

# Example usage:
if __name__ == "__main__":
    # Create dataloaders
    dataloaders = create_dataloaders(batch_size=32)
    
    # Example: Iterate through one batch of training data
    for batch in dataloaders['train']:
        image_embeddings = batch['image_embeddings']  # Shape: (B, 196, 512)
        captions = batch['captions']  # List of captions
        print(f"Batch image embeddings shape: {image_embeddings.shape}")
        print(f"Number of captions: {len(captions)}")
        break  # Just show first batch

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class ImagePreprocessor:
    """Handles image preprocessing including resizing and patching."""
    
    def __init__(self, img_size=224, patch_size=16):
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

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional
import logging
from pathlib import Path

from CodeModel1 import ImagePreprocessor
from CodeModel2 import Transformer
from tokenizer import CaptionTokenizer
from Background1 import *

class CaptionGenerator:
    """
    Handles image caption generation using the trained transformer model.
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Load tokenizer
        self.tokenizer = CaptionTokenizer.from_file('vocab.json')
        
        # Initialize image processor
        self.image_processor = ImagePreprocessor()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _setup_logging(self):
        """Set up logging for inference"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_model(self, model_path: str) -> Transformer:
        """Load the trained model"""
        try:
            # Initialize model architecture
            model = Transformer(
                vocab_size=self.tokenizer.get_vocab_size(),
                d_model=D_MODEL,
                num_heads=N_HEADS,
                num_encoder_layers=6,
                num_decoder_layers=6
            ).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise 
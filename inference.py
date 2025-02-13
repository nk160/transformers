import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional
import logging
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np

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

    def generate_caption(self, image: Image.Image, max_length: int = 50, 
                        beam_size: int = 5) -> List[str]:
        """
        Generate captions for an image using beam search.
        
        Args:
            image: Input PIL Image
            max_length: Maximum caption length
            beam_size: Beam search width
            
        Returns:
            List of top-k captions, sorted by probability
        """
        # Process image
        image_features = self.image_processor.process_image(image).unsqueeze(0).to(self.device)
        
        # Generate with beam search
        with torch.no_grad():
            return self._beam_search(image_features, beam_size, max_length)
    
    def _beam_search(self, image_features: torch.Tensor, beam_size: int, 
                    max_length: int) -> List[str]:
        """
        Implement beam search for caption generation.
        """
        # Initialize beam
        start_token = torch.tensor([[self.tokenizer.START_token]]).to(self.device)
        beam = [(start_token, 0.0, [])]  # (sequence, score, complete_captions)
        
        # Get encoder output once
        encoder_output = self.model.encoder(image_features)
        
        for _ in range(max_length):
            candidates = []
            
            # Expand each beam
            for seq, score, complete in beam:
                # Skip if sequence is complete
                if complete:
                    candidates.append((seq, score, complete))
                    continue
                
                # Get model predictions
                decoder_output = self.model.decoder(seq, encoder_output)
                logits = self.model.linear(decoder_output[:, -1])
                probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                values, indices = probs[0].topk(beam_size)
                
                for value, token_idx in zip(values, indices):
                    new_seq = torch.cat([seq, token_idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + value.item()
                    
                    # Check if sequence is complete
                    is_complete = token_idx.item() == self.tokenizer.END_token
                    candidates.append((new_seq, new_score, is_complete))
            
            # Select top-k candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]
            
            # Check if all sequences are complete
            if all(complete for _, _, complete in beam):
                break
        
        # Convert sequences to captions
        captions = []
        for seq, score, _ in beam:
            caption = self.tokenizer.decode(seq[0], skip_special_tokens=True)
            captions.append((caption, score))
        
        return [cap for cap, _ in sorted(captions, key=lambda x: x[1], reverse=True)]
    
    def visualize_attention(self, image: Image.Image, caption: str):
        """
        Visualize attention weights for a generated caption.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Process image and prepare inputs
        image_features = self.image_processor.process_image(image).unsqueeze(0).to(self.device)
        caption_tokens = self.tokenizer.encode(caption).unsqueeze(0).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            encoder_output = self.model.encoder(image_features)
            decoder_output, attention_weights = self.model.decoder(
                caption_tokens, 
                encoder_output, 
                return_attention=True
            )
        
        # Plot attention heatmap
        plt.figure(figsize=(15, 10))
        
        # Plot original image
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        # Plot attention heatmap
        plt.subplot(2, 1, 2)
        attention = attention_weights[-1].mean(1)[0].cpu()  # Use last layer's attention
        sns.heatmap(attention, xticklabels=caption.split(), yticklabels=False)
        plt.title("Attention Weights")
        
        plt.tight_layout()
        plt.show()

    def generate_batch(self, images: List[Image.Image], max_length: int = 50, 
                      beam_size: int = 5) -> List[List[str]]:
        """
        Generate captions for a batch of images.
        """
        all_captions = []
        for img in images:
            captions = self.generate_caption(img, max_length, beam_size)
            all_captions.append(captions)
            self.logger.info(f"Generated captions: {captions[0]}")  # Log best caption
        return all_captions 

    def evaluate_caption(self, generated_caption: str, reference_captions: List[str]) -> dict:
        """
        Evaluate generated caption against reference captions.
        """
        # Calculate BLEU score
        bleu1 = sentence_bleu([ref.split() for ref in reference_captions], 
                            generated_caption.split(), weights=(1, 0, 0, 0))
        bleu4 = sentence_bleu([ref.split() for ref in reference_captions], 
                            generated_caption.split(), weights=(0.25, 0.25, 0.25, 0.25))
        
        # Calculate METEOR score
        meteor = np.mean([
            meteor_score([ref.split()], generated_caption.split())
            for ref in reference_captions
        ])
        
        return {
            'bleu1': bleu1,
            'bleu4': bleu4,
            'meteor': meteor
        }
    
    def visualize_generation_process(self, image: Image.Image, max_length: int = 50):
        """
        Visualize the caption generation process step by step.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Process image
        image_features = self.image_processor.process_image(image).unsqueeze(0).to(self.device)
        
        # Initialize generation
        current_tokens = torch.tensor([[self.tokenizer.START_token]]).to(self.device)
        generated_words = []
        attention_maps = []
        
        # Generate word by word
        with torch.no_grad():
            encoder_output = self.model.encoder(image_features)
            
            for _ in range(max_length):
                # Get predictions and attention
                decoder_output, attention = self.model.decoder(
                    current_tokens, 
                    encoder_output,
                    return_attention=True
                )
                logits = self.model.linear(decoder_output[:, -1])
                next_token = logits.argmax(dim=-1)
                
                # Save attention map
                attention_maps.append(attention[-1].mean(1)[0].cpu())
                
                # Convert token to word
                word = self.tokenizer.decode(next_token, skip_special_tokens=True)
                generated_words.append(word)
                
                # Check if generation is complete
                if next_token.item() == self.tokenizer.END_token:
                    break
                    
                # Append token for next iteration
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
        
        # Visualize
        n_steps = len(generated_words)
        fig, axes = plt.subplots(n_steps + 1, 1, figsize=(15, 5 * (n_steps + 1)))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Plot attention maps for each generation step
        for idx, (word, attention) in enumerate(zip(generated_words, attention_maps)):
            sns.heatmap(attention, ax=axes[idx + 1])
            axes[idx + 1].set_title(f"Step {idx + 1}: Generated '{word}'")
        
        plt.tight_layout()
        plt.show()

    def save_results(self, image_path: str, generated_captions: List[str], 
                    reference_captions: Optional[List[str]] = None,
                    output_dir: str = "results"):
        """
        Save generation results including visualizations.
        """
        from datetime import datetime
        import os
        import json
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(output_dir, f"generation_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and save image
        image = Image.open(image_path)
        image.save(os.path.join(save_dir, "input_image.jpg"))
        
        # Save captions and metrics
        results = {
            "generated_captions": generated_captions,
            "reference_captions": reference_captions,
            "metrics": None
        }
        
        if reference_captions:
            results["metrics"] = self.evaluate_caption(
                generated_captions[0],  # Use top caption
                reference_captions
            )
        
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save visualizations
        self.visualize_attention(image, generated_captions[0])
        plt.savefig(os.path.join(save_dir, "attention_visualization.png"))
        plt.close()
        
        self.visualize_generation_process(image)
        plt.savefig(os.path.join(save_dir, "generation_process.png"))
        plt.close()
        
        self.logger.info(f"Results saved to {save_dir}") 
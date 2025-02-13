import argparse
from pathlib import Path
import torch
from PIL import Image
import logging
from inference import CaptionGenerator
from typing import List, Optional
import json

def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/generation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_image(image_path: str) -> Optional[Image.Image]:
    """Load and validate image"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {str(e)}")
        return None

def load_reference_captions(caption_path: str) -> Optional[List[str]]:
    """Load reference captions from JSON file"""
    try:
        with open(caption_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load reference captions: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate image captions using trained transformer')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image_path', required=True, help='Path to input image or directory of images')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum caption length')
    parser.add_argument('--reference_captions', help='Path to reference captions JSON file')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing multiple images')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Initialize generator
    generator = CaptionGenerator(args.model_path, device=args.device)
    
    # Process input path
    input_path = Path(args.image_path)
    if input_path.is_file():
        # Single image
        image = load_image(str(input_path))
        if image is None:
            return
        
        reference_captions = None
        if args.reference_captions:
            reference_captions = load_reference_captions(args.reference_captions)
        
        # Generate and save results
        captions = generator.generate_caption(
            image, 
            max_length=args.max_length,
            beam_size=args.beam_size
        )
        
        generator.save_results(
            str(input_path),
            captions,
            reference_captions,
            args.output_dir
        )
        
    else:
        # Directory of images
        image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        logger.info(f"Found {len(image_paths)} images")
        
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i + args.batch_size]
            batch_images = [load_image(str(p)) for p in batch_paths]
            batch_images = [img for img in batch_images if img is not None]
            
            if not batch_images:
                continue
            
            # Generate captions for batch
            all_captions = generator.generate_batch(
                batch_images,
                max_length=args.max_length,
                beam_size=args.beam_size
            )
            
            # Save results for each image
            for image_path, captions in zip(batch_paths, all_captions):
                generator.save_results(
                    str(image_path),
                    captions,
                    None,
                    args.output_dir
                )

if __name__ == "__main__":
    main() 
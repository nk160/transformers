import torch
from torch.utils.data import DataLoader
from image_dataset import ImageCaptioningDataset, collate_fn
from utils.config import load_config
import logging

def test_dataset(config_path='config.yaml'):
    # Load config
    config = load_config(config_path)
    
    # Initialize dataset
    dataset = ImageCaptioningDataset(
        image_dir=config['data']['image_dir'],
        captions_file='captions.json',
        vocab_file='vocab.json',
        max_length=config['training']['max_length']
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Test a single batch
    for images, captions, lengths in dataloader:
        # Verify dimensions
        assert images.shape[1:] == (3, 224, 224), f"Expected image shape (3, 224, 224), got {images.shape[1:]}"
        assert captions.shape[1] == config['training']['max_length'], f"Expected caption length {config['training']['max_length']}, got {captions.shape[1]}"
        
        print(f"Batch size: {images.shape[0]}")
        print(f"Image shape: {images.shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"Caption lengths: {lengths}")
        
        # Convert caption to text
        idx2word = {v: k for k, v in dataset.word2idx.items()}
        caption = captions[0].tolist()
        text = ' '.join([idx2word[idx] for idx in caption if idx != dataset.word2idx['[PAD]']])
        print(f"\nSample caption: {text}")
        
        # Verify special tokens
        assert caption[0] == dataset.word2idx['[START]'], "Missing [START] token"
        assert dataset.word2idx['[END]'] in caption, "Missing [END] token"
        break

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_dataset() 
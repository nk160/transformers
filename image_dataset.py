import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab_file, transform=None, max_length=50):
        self.image_dir = image_dir
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.max_length = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_data = self.captions[idx]
        img_name = caption_data['image']
        caption = caption_data['caption']
        
        # Load and transform image
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Convert caption to tensor of indices
        tokens = ['[START]'] + caption.lower().split() + ['[END]']
        tokens = tokens[:self.max_length]  # Truncate if too long
        
        # Pad with [PAD] tokens if needed
        while len(tokens) < self.max_length:
            tokens.append('[PAD]')

        # Convert tokens to indices
        caption_indices = [self.word2idx.get(token, self.word2idx['[UNK]']) 
                         for token in tokens]
        caption_tensor = torch.tensor(caption_indices)

        return image, caption_tensor

def collate_fn(batch):
    # Sort batch by caption length (descending)
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)

    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions to same length
    lengths = [len(cap) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths 
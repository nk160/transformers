import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from CodeModel1 import create_dataloaders
from CodeModel2 import Transformer
from Background1 import *
import wandb
from datetime import datetime
import logging
import sys
from pathlib import Path
from tokenizer import CaptionTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')  # Required for METEOR score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from collections import defaultdict
from utils.config import load_config

class DropoutWrapper(nn.Module):
    """Wrapper module that adds dropout to a model"""
    def __init__(self, model, dropout_p=0.2):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        # Copy important attributes from the model
        self.vocab_size = model.vocab_size
        
    def forward(self, src, tgt, tgt_mask=None):
        # Pass through model first
        output = self.model(src=src, tgt=tgt, tgt_mask=tgt_mask)
        # Apply dropout to output
        return self.dropout(output)

class TransformerTrainer:
    """
    Handles the training process for the Transformer model.
    Includes training loop, optimization, and logging.
    """
    def __init__(self, model, vocab_size, learning_rate=None, device='cuda', config=None):
        """Initialize trainer with additional regularization"""
        self.config = config or load_config('config.yaml')
        self.device = torch.device(device)
        
        # Wrap model with dropout
        self.model = DropoutWrapper(
            model,
            dropout_p=0.2
        ).to(self.device)
        
        self.logger = logging.getLogger(__name__)
        
        # Use learning rate from config
        learning_rate = learning_rate or self.config['training']['learning_rate']
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5,
            verbose=True
        )
        
        self.criterion = CrossEntropyLoss(ignore_index=0)  # ignore PAD token
        self.tokenizer = CaptionTokenizer.from_file('vocab.json')
        
        # Verify dimensions
        assert self.tokenizer.get_vocab_size() == model.vocab_size, \
            f"Tokenizer vocab size ({self.tokenizer.get_vocab_size()}) != Model vocab size ({model.vocab_size})"
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Initialize example indices for validation logging
        self.example_indices = list(range(min(5, vocab_size)))  # Log first 5 examples

    def create_masks(self, src, tgt):
        """Create masks for transformer input"""
        batch_size = src.size(0)
        
        # Handle 4D input by removing singleton dimension if needed
        if src.dim() == 4:
            src = src.squeeze(1)  # Remove singleton dimension
        
        # Create source mask (all ones since we want to attend to all patches)
        src_seq_len = src.size(1)
        src_mask = torch.ones((batch_size, 1, 1, src_seq_len), device=self.device)
        
        # Create target mask (causal/triangular mask)
        tgt_seq_len = tgt.size(-1)
        tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len)), diagonal=1).bool()
        tgt_mask = ~tgt_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        tgt_mask = tgt_mask.to(self.device)
        
        return src_mask, tgt_mask

    def setup_wandb(self, project_name="image-captioning-transformer"):
        """
        Initialize Weights & Biases logging.
        """
        self.run = wandb.init(
            project=project_name,
            config={
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "architecture": "Transformer",
                "dataset": "Flickr30k",
                "epochs": self.config['training']['num_epochs'],
                "batch_size": self.config['training']['batch_size'],
                "device": self.device
            }
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data with correct keys
            image_features = batch['image_embeddings'].to(self.device)
            captions = batch['captions'].to(self.device)
            
            # Create target mask of correct size
            seq_len = captions.size(-1)
            tgt_mask = torch.ones(seq_len, seq_len, device=self.device)
            tgt_mask = torch.triu(tgt_mask, diagonal=1).bool()
            tgt_mask = ~tgt_mask
            
            # Forward pass
            outputs = self.model(
                src=image_features,
                tgt=captions,
                tgt_mask=tgt_mask
            )
            
            # Reshape outputs and targets for loss calculation
            B, num_caps, seq_len, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)  # Combine batch and caption dimensions
            targets = captions.reshape(-1)  # Flatten targets to match
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def _log_example_predictions(self, output, target, batch_idx):
        """
        Log example predictions for monitoring.
        
        Args:
            output: Model output tensor (B*num_caps*seq_len, vocab_size) or (B*num_caps, seq_len, vocab_size)
            target: Target tensor (B, num_caps, seq_len)
            batch_idx: Current batch index
        """
        # Get first example from batch
        if output.dim() == 2:
            # Reshape if flattened
            B = output.size(0) // (target.size(1) * target.size(2))
            output = output.reshape(B, target.size(1), target.size(2), -1)
        
        # Take first caption from first example
        pred_tokens = output[0, 0].argmax(dim=-1)  # (seq_len,)
        target_tokens = target[0, 0]  # (seq_len,)
        
        # Decode to text
        pred_caption = self.tokenizer.decode(pred_tokens.tolist(), skip_special_tokens=True)
        target_caption = self.tokenizer.decode(target_tokens.tolist(), skip_special_tokens=True)
        
        # Log to wandb
        wandb.log({
            f"example_prediction_{batch_idx}": wandb.Table(
                columns=["Predicted", "Target"],
                data=[[pred_caption, target_caption]]
            )
        })

    def validate(self, val_loader):
        """Validate the model on the validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Get data
                image_features = batch['image_embeddings'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                # Forward pass
                output = self.model(
                    src=image_features,
                    tgt=captions,
                    tgt_mask=self.create_masks(image_features, captions)[1]
                )
                
                # Calculate loss
                B, num_caps, seq_len, vocab_size = output.shape
                output = output.reshape(-1, vocab_size)
                targets = captions.reshape(-1)
                loss = self.criterion(output, targets)
                total_loss += loss.item()
                
                # Get predictions for logging
                pred_captions = self._get_predictions(output)
                target_captions = self._get_predictions(captions)
                
                # Store predictions and targets
                for i in range(B):
                    idx = batch_idx * B + i
                    if idx < len(self.example_indices):
                        example_idx = self.example_indices[idx]
                        all_predictions.append(pred_captions[example_idx])
                        all_targets.append(target_captions[example_idx])
                
                # Log example predictions
                if batch_idx == 0:
                    self._log_example_predictions(output, captions, batch_idx)
        
        # Log validation metrics
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss

    def _get_predictions(self, output):
        """
        Convert model output to caption predictions.
        
        Args:
            output: Model output tensor (batch_size * num_caps, seq_len, vocab_size)
                    or (batch_size * num_caps * seq_len, vocab_size)
        """
        # Ensure output is 3D
        if output.dim() == 2:
            B = output.size(0) // 50  # Assuming seq_len=50
            output = output.reshape(B, 50, -1)
        
        # Get most likely tokens
        pred_tokens = output.argmax(dim=-1)  # (batch_size * num_caps, seq_len)
        
        # Convert to captions
        pred_captions = []
        for tokens in pred_tokens:
            caption = self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            pred_captions.append(caption)
        
        return pred_captions

    def create_target_mask(self, size):
        """Create mask for decoder self-attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def train(self, train_loader, val_loader, num_epochs, batch_size):
        """
        Main training loop with early stopping and learning rate scheduling.
        """
        try:
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            logger = logging.getLogger(__name__)
            
            # Setup W&B logging
            self.setup_wandb()
            
            # Early stopping parameters
            patience = 5
            no_improve_count = 0
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                try:
                    self.current_epoch = epoch
                    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
                    
                    # Training phase
                    train_loss = self.train_epoch(train_loader)
                    self.train_losses.append(train_loss)
                    
                    # Validation phase
                    val_loss = self.validate(val_loader)
                    self.val_losses.append(val_loss)
                    
                    # Learning rate scheduling
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Log metrics
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr
                    })
                    
                    # Early stopping and model saving logic
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        
                        # Save best model
                        model_path = f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'train_losses': self.train_losses,
                                'val_losses': self.val_losses,
                            }, model_path)
                            wandb.save(model_path)
                            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
                        except Exception as e:
                            logger.error(f"Failed to save model checkpoint: {str(e)}")
                    else:
                        no_improve_count += 1
                    
                    # Early stopping check
                    if no_improve_count >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                    
                    logger.info(f'Epoch {epoch+1}/{num_epochs}:')
                    logger.info(f'Training Loss: {train_loss:.4f}')
                    logger.info(f'Validation Loss: {val_loss:.4f}')
                    logger.info(f'Learning Rate: {current_lr:.6f}')
                    logger.info('-' * 50)
                    
                except Exception as e:
                    logger.error(f"Error in epoch {epoch+1}: {str(e)}")
                    raise
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            wandb.finish()
            logger.info("Training finished")

def setup_logging(log_dir="logs"):
    """
    Set up logging configuration.
    Creates a log file with timestamp and configures console output.
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main training function"""
    # Load configuration
    config = load_config('config.yaml')
    
    # Set up logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info("Starting training setup...")
    
    # Set device
    device = torch.device(config['training']['device'] 
                        if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        dataloaders = create_dataloaders(config)
        
        # Initialize model
        logger.info("Initializing model...")
        model = Transformer(config)
        
        # Initialize trainer
        logger.info("Setting up trainer...")
        trainer = TransformerTrainer(
            model=model,
            vocab_size=config['model']['vocab_size'],
            device=device,
            config=config
        )
        
        # Start training
        logger.info("Starting training process...")
        trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config['training']['num_epochs'],
            batch_size=config['training']['batch_size']
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        logger.info("Training process ended")

if __name__ == "__main__":
    main() 
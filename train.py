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

class TransformerTrainer:
    """
    Handles the training process for the Transformer model.
    Includes training loop, optimization, and logging.
    """
    def __init__(self, model, vocab_size, learning_rate=0.0001, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = CrossEntropyLoss(ignore_index=0)  # ignore PAD token (0)
        self.tokenizer = CaptionTokenizer.from_file('vocab.json')  # Load saved vocabulary
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        
    def create_masks(self, src, tgt):
        """
        Create source and target masks for transformer.
        
        Args:
            src: Source sequence (image features)
            tgt: Target sequence (captions)
            
        Returns:
            src_mask: Mask for source sequence
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
        """
        # Source mask (None for image features as we want to attend to all patches)
        src_mask = None
        
        # Target mask (prevents attending to future tokens)
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
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
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "device": self.device
            }
        )

    def train_epoch(self, train_loader):
        """Modified to handle tokenized captions"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        logger = logging.getLogger(__name__)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get image features and tokenized captions
            image_features = batch['image_embeddings'].to(self.device)
            captions = batch['captions'].to(self.device)  # Now contains encoded captions
            
            # Create masks
            src_mask, tgt_mask = self.create_masks(image_features, captions[:, :-1])
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                src=image_features,
                tgt=captions[:, :-1],  # Input tokens (exclude last)
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            
            # Calculate loss (reshape for CrossEntropyLoss)
            loss = self.criterion(
                output.reshape(-1, self.tokenizer.get_vocab_size()),
                captions[:, 1:].reshape(-1)  # Target tokens (exclude first)
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log example predictions periodically
            if batch_idx % 100 == 0:
                self._log_example_predictions(output[0], captions[0], batch_idx)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": self.current_epoch,
                "batch": batch_idx
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss

    def _log_example_predictions(self, output, target, batch_idx):
        """Log example predictions for monitoring"""
        # Get predicted caption
        pred_tokens = output.argmax(dim=-1)
        pred_caption = self.tokenizer.decode(pred_tokens)
        
        # Get target caption
        target_caption = self.tokenizer.decode(target)
        
        # Log to wandb
        wandb.log({
            f"example_prediction_{batch_idx}": wandb.Table(
                columns=["Predicted", "Target"],
                data=[[pred_caption, target_caption]]
            )
        })

    def validate(self, val_loader):
        """
        Validate the model with caption quality metrics.
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        logger = logging.getLogger(__name__)
        
        # Metrics tracking
        bleu_scores = []
        meteor_scores = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Validation')
            
            for batch in progress_bar:
                # Get image features and captions
                image_features = batch['image_embeddings'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                # Create masks
                src_mask, tgt_mask = self.create_masks(image_features, captions[:, :-1])
                
                # Forward pass
                output = self.model(
                    src=image_features,
                    tgt=captions[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                
                # Calculate loss
                loss = self.criterion(
                    output.reshape(-1, self.tokenizer.get_vocab_size()),
                    captions[:, 1:].reshape(-1)
                )
                
                total_loss += loss.item()
                
                # Calculate caption quality metrics
                pred_captions = self._get_predictions(output)
                target_captions = [
                    self.tokenizer.decode(cap, skip_special_tokens=True)
                    for cap in captions
                ]
                
                # Update metrics
                for pred, target in zip(pred_captions, target_captions):
                    # BLEU score
                    bleu = sentence_bleu([target.split()], pred.split())
                    bleu_scores.append(bleu)
                    
                    # METEOR score
                    meteor = meteor_score([target.split()], pred.split())
                    meteor_scores.append(meteor)
                
                progress_bar.set_postfix({
                    'val_loss': loss.item(),
                    'bleu': np.mean(bleu_scores[-batch.size(0):]),
                    'meteor': np.mean(meteor_scores[-batch.size(0):])
                })
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_bleu = np.mean(bleu_scores)
        avg_meteor = np.mean(meteor_scores)
        
        # Log metrics
        wandb.log({
            "val_loss": avg_loss,
            "val_bleu": avg_bleu,
            "val_meteor": avg_meteor
        })
        
        logger.info(f"Validation Metrics:")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"BLEU: {avg_bleu:.4f}")
        logger.info(f"METEOR: {avg_meteor:.4f}")
        
        return avg_loss
    
    def _get_predictions(self, output):
        """
        Convert model output to caption predictions.
        """
        # Get most likely tokens
        pred_tokens = output.argmax(dim=-1)
        
        # Convert to captions
        pred_captions = [
            self.tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in pred_tokens
        ]
        
        return pred_captions

    def train(self, train_loader, val_loader, num_epochs, batch_size):
        """
        Main training loop with error handling.
        """
        try:
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            logger = logging.getLogger(__name__)
            
            # Setup W&B logging
            self.setup_wandb()
            
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
                    
                    # Log metrics
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model_path = f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'train_losses': self.train_losses,
                                'val_losses': self.val_losses,
                            }, model_path)
                            wandb.save(model_path)
                            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
                        except Exception as e:
                            logger.error(f"Failed to save model checkpoint: {str(e)}")
                    
                    logger.info(f'Epoch {epoch+1}/{num_epochs}:')
                    logger.info(f'Training Loss: {train_loss:.4f}')
                    logger.info(f'Validation Loss: {val_loss:.4f}')
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
    """
    Main function to set up and run the training process.
    """
    try:
        # Set up logging
        logger = setup_logging()
        logger.info("Starting training setup...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create dataloaders
        try:
            logger.info("Creating dataloaders...")
            dataloaders = create_dataloaders(batch_size=32, num_workers=4)
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']
            logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        except Exception as e:
            logger.error(f"Failed to create dataloaders: {str(e)}")
            raise
        
        # Initialize model
        try:
            logger.info("Initializing model...")
            vocab_size = 10000
            model = Transformer(
                vocab_size=vocab_size,
                d_model=D_MODEL,
                num_heads=N_HEADS,
                num_encoder_layers=6,
                num_decoder_layers=6
            )
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
        
        # Initialize trainer
        try:
            logger.info("Setting up trainer...")
            trainer = TransformerTrainer(
                model=model,
                vocab_size=vocab_size,
                learning_rate=0.0001,
                device=device
            )
            logger.info("Trainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {str(e)}")
            raise
        
        # Start training
        logger.info("Starting training process...")
        try:
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=30,
                batch_size=32
            )
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Training setup failed: {str(e)}")
        raise
    finally:
        logger.info("Training process ended")

if __name__ == "__main__":
    main() 
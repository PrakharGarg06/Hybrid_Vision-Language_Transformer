
import torch
from transformers import T5Tokenizer, BertTokenizer, ViTImageProcessor
import os

def get_tokenizers_and_image_processor():
    """
    Returns:
      - T5 tokenizer (for captions)
      - BERT tokenizer (for questions)
      - ViT image processor (for images)
    """
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # --- CHANGE: Use the modern ViTImageProcessor ---
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    return t5_tokenizer, bert_tokenizer, image_processor


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the entire state dictionary to a file."""
    torch.save(state, filename)
    print(f"✅ Saved checkpoint: {filename}")


def load_checkpoint(filename, map_location="cpu"):
    """
    Loads a checkpoint file.
    This is more flexible as it just returns the state dictionary,
    allowing the calling script to load the correct parts into the correct models.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint {filename} not found.")
    
    # --- CHANGE: More flexible loading ---
    checkpoint = torch.load(filename, map_location=map_location)
    print(f"✅ Loaded checkpoint: {filename}")
    return checkpoint




 


import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from dataset_hybrid import SimpleVQADataset
from models_hybrid import VQAHybrid
from utils_hybrid import get_tokenizers_and_image_processor, save_checkpoint, load_checkpoint


# =============================
# Global variables for multiprocessing
# =============================
BERT_TOKENIZER = None
IMAGE_PROCESSOR = None
ANSWER_MAP_LOADED = None


def collate_vqa(batch):
    """Collates VQA samples into tensors."""
    global BERT_TOKENIZER, IMAGE_PROCESSOR, ANSWER_MAP_LOADED
    images, questions, answers = zip(*batch)

    pixel_values = IMAGE_PROCESSOR(images=list(images), return_tensors='pt')['pixel_values']
    q_enc = BERT_TOKENIZER(list(questions), padding='longest', truncation=True, max_length=32, return_tensors='pt')

    input_ids = q_enc['input_ids']
    attention_mask = q_enc['attention_mask']
    ans_idx = [ANSWER_MAP_LOADED.get(a.lower(), 0) for a in answers]
    ans_idx = torch.tensor(ans_idx, dtype=torch.long)

    return pixel_values, input_ids, attention_mask, ans_idx


def train(args):
    """Train the VQA hybrid model."""
    if not os.path.exists(args.vqa_json):
        raise FileNotFoundError(f"❌ VQA JSON not found: {args.vqa_json}")
    if not os.path.exists(args.answer_map):
        raise FileNotFoundError(f"❌ Answer map not found: {args.answer_map}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processors and answer map
    global BERT_TOKENIZER, IMAGE_PROCESSOR, ANSWER_MAP_LOADED
    _, BERT_TOKENIZER, IMAGE_PROCESSOR = get_tokenizers_and_image_processor()
    with open(args.answer_map, "r") as f:
        ANSWER_MAP_LOADED = json.load(f)

    print(f"✅ Loaded {len(ANSWER_MAP_LOADED)} answers from {args.answer_map}")

    # Auto-detect enriched answer map
    if "enriched" in args.answer_map.lower():
        print("✨ Enriched Answer Map detected! Model will learn descriptive words (nouns + verbs).")
    else:
        print("⚙️ Base Answer Map detected. Training on limited categorical answers.")

    dataset = SimpleVQADataset(args.vqa_json)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_vqa,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    model = VQAHybrid(
        vit_out_dim=args.img_dim,
        bert_model_name=args.bert_name,
        hidden_dim=args.hidden_dim,
        num_answers=len(ANSWER_MAP_LOADED),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"🔄 Resuming training from checkpoint: {args.resume_from}")
        checkpoint = load_checkpoint(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        try:
            start_epoch = int(os.path.basename(args.resume_from).split("_")[-1].split(".")[0]) + 1
        except:
            start_epoch = 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"🚀 Starting VQA Training (from epoch {start_epoch})...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        last_loss = None

        for pixel_values, input_ids, attention_mask, ans_idx in loop:
            pixel_values, input_ids, attention_mask, ans_idx = (
                pixel_values.to(device),
                input_ids.to(device),
                attention_mask.to(device),
                ans_idx.to(device),
            )

            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, ans_idx)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            loop.set_postfix(loss=last_loss)

        # Save checkpoint
        save_path = os.path.join(args.ckpt_dir, f"vqa_epoch_{epoch}.pt")
        save_checkpoint({"model": model.state_dict(), "opt": optimizer.state_dict()}, save_path)
        print(f"✅ Epoch {epoch} finished | Loss: {last_loss:.4f} | Saved → {save_path}")

    print("🎯 VQA training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid VQA model with auto-detection of enriched answer maps")

    # Paths
    parser.add_argument("--vqa_json", type=str, required=True, help="Path to VQA JSON (patched for local)")
    parser.add_argument("--answer_map", type=str, required=True, help="Path to answer_map.json or answer_map_enriched.json")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume checkpoint")

    # Model config
    parser.add_argument("--img_dim", type=int, default=768)
    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--hidden_dim", type=int, default=1024)

    # Training config
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    train(args)
 


import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import matplotlib.pyplot as plt

from dataset_hybrid import SimpleCaptionDataset
from models_hybrid import VisionEncoderViT, CaptionDecoder
from utils_hybrid import get_tokenizers_and_image_processor, save_checkpoint, load_checkpoint

# Globals
T5_TOKENIZER = None
IMAGE_PROCESSOR = None


def collate_caption(batch, max_len=30):
    """Prepare batch of (images, captions)."""
    global T5_TOKENIZER, IMAGE_PROCESSOR
    images, captions = zip(*batch)
    inputs = IMAGE_PROCESSOR(images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    tokenized = T5_TOKENIZER(list(captions), padding="longest", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    return pixel_values, input_ids, list(captions)


def train(args):
    """Train captioning model and collect metrics"""
    if not os.path.exists(args.captions_json):
        raise FileNotFoundError(f"❌ Missing captions file: {args.captions_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global T5_TOKENIZER, IMAGE_PROCESSOR
    T5_TOKENIZER, _, IMAGE_PROCESSOR = get_tokenizers_and_image_processor()

    dataset = SimpleCaptionDataset(args.captions_json)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_caption,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    vision_encoder = VisionEncoderViT(out_dim=768).to(device)
    caption_decoder = CaptionDecoder(vocab_size=T5_TOKENIZER.vocab_size, d_model=768, num_layers=6).to(device)

    optimizer = torch.optim.Adam(list(vision_encoder.parameters()) + list(caption_decoder.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=T5_TOKENIZER.pad_token_id)

    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"🔁 Resuming from checkpoint: {args.resume_from}")
        ckpt = load_checkpoint(args.resume_from, map_location=device)
        vision_encoder.load_state_dict(ckpt["vision"])
        caption_decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["opt"])
        try:
            start_epoch = int(os.path.basename(args.resume_from).split("_")[-1].split(".")[0]) + 1
        except:
            start_epoch = 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "caption_metrics_log.json")
    metrics_log = []

    print(f"🚀 Starting Captioning Training from epoch {start_epoch}...")

    for epoch in range(start_epoch, args.epochs):
        vision_encoder.train()
        caption_decoder.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")

        epoch_losses, bleu_scores = [], []
        smooth_fn = SmoothingFunction().method1

        for pixel_values, input_ids, true_captions in loop:
            pixel_values, input_ids = pixel_values.to(device), input_ids.to(device)
            optimizer.zero_grad()

            # Forward
            _, img_seq = vision_encoder(pixel_values)
            tgt_input = input_ids[:, :-1]
            tgt_output = input_ids[:, 1:]

            logits = caption_decoder(tgt_input, img_seq)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Generate captions for BLEU evaluation (greedy)
            with torch.no_grad():
                _, img_seq_eval = vision_encoder(pixel_values)
                token_ids = caption_decoder.generate_greedy(img_seq_eval, T5_TOKENIZER, max_len=25, device=device)
                pred_caption = T5_TOKENIZER.decode(token_ids, skip_special_tokens=True)
                # Compare with first reference in batch
                bleu = sentence_bleu([true_captions[0].split()], pred_caption.split(), smoothing_function=smooth_fn)
                bleu_scores.append(bleu)

            loop.set_postfix(loss=loss.item())

        avg_loss = np.mean(epoch_losses)
        avg_bleu = np.mean(bleu_scores)

        print(f"\n📊 Epoch {epoch} | Loss: {avg_loss:.4f} | BLEU: {avg_bleu:.4f}")

        # Save checkpoint
        save_path = os.path.join(args.ckpt_dir, f"caption_epoch_{epoch}.pt")
        save_checkpoint({
            "vision": vision_encoder.state_dict(),
            "decoder": caption_decoder.state_dict(),
            "opt": optimizer.state_dict()
        }, save_path)

        # Save logs
        metrics_log.append({
            "epoch": epoch,
            "loss": avg_loss,
            "bleu": avg_bleu
        })
        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

        print(f"✅ Saved checkpoint → {save_path}")

    print("🎉 Captioning Training Completed | Metrics saved at:", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid image captioning model with EDA")
    parser.add_argument("--captions_json", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    train(args)
 

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
import os # Ensure os is imported again if needed (already imported, but checking environment relies on it)

from dataset import PIIDataset, collate_batch
from labels import LABELS, ID2LABEL
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def get_loss_weights(id2label: dict, device: torch.device) -> torch.Tensor:
    weights = torch.ones(len(id2label))
    O_WEIGHT = 0.5
    HIGH_RISK_WEIGHT = 5.0
    HIGH_RISK_PREFIXES = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"]
    for label_id, label_name in id2label.items():
        if label_name == 'O':
            weights[label_id] = O_WEIGHT
        elif any(prefix in label_name for prefix in HIGH_RISK_PREFIXES):
            weights[label_id] = HIGH_RISK_WEIGHT
    return weights.to(device)


# === FINAL TRAIN FUNCTION WITH CONDITIONAL WANDB INITIALIZATION ===
def train(config=None):
    # Determine execution context
    is_sweep_run = config is not None
    
    # 1. Handle configuration (Priority: Sweep Config > Command Line Args)
    if is_sweep_run:
        wandb.init(config=config)
        args = argparse.Namespace(**wandb.config)
    else:
        # Standard run: parse args, but DO NOT initialize wandb unconditionally
        args = parse_args()
    
    # --- Check for WANDB_DISABLED environment variable manually ---
    # This ensures that even if WANDB_DISABLED=true, the code runs without error.
    is_wandb_logging_active = is_sweep_run and (os.getenv('WANDB_DISABLED') != 'true')

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Training Setup ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True) 

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()
    
    # --- Loss Weighting Implementation ---
    loss_weights = get_loss_weights(ID2LABEL, args.device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            model_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
            )
            
            loss = model_outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # --- Conditional WandB Logging ---
        if is_wandb_logging_active:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})


    # --- Saving Model (Custom CRF Saving) ---
    torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))
    model.distilbert.config.save_pretrained(args.out_dir) 
    tokenizer.save_pretrained(args.out_dir)

    print(f"Saved custom model weights and tokenizer to {args.out_dir}")
    
    # --- Conditional WandB Finalization ---
    if is_wandb_logging_active:
        wandb.finish()


if __name__ == "__main__":
    train()
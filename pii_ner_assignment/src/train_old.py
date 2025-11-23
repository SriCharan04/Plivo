import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS, ID2LABEL # Need ID2LABEL for saving logic
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl") # Use large data
    ap.add_argument("--dev", default="data/dev.jsonl")     # Use large data
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3) # Set to a reasonable value for training
    ap.add_argument("--lr", type=float, default=5e-5) # Reduced LR for stability (was 2.5e-4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )
    
    # Optional: Gradient Clipping implementation (for stability)
    MAX_GRAD_NORM = 1.0 

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            # --- Loss calculation for CUSTOM CRF Model ---
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"] # Corrected access method
            # ---------------------------------------------

            optimizer.zero_grad()
            loss.backward()
            
            # Optional: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # --- FIX: Custom saving for DistilBertCrfForTokenClassification ---
    # Save the custom model's state dictionary
    torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))
    
    # Save the configuration (essential for loading the model structure later)
    model.distilbert.config.save_pretrained(args.out_dir) 
    
    # Save the tokenizer (essential for prediction)
    tokenizer.save_pretrained(args.out_dir)
    # --- End FIX ---
    
    print(f"Saved custom model weights and tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
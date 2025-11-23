import wandb
import argparse
import os

# --- CRITICAL CHANGE: The 'train' function is now imported ---
# Ensure this file is saved in the root, and the path below is correct:
try:
    from src.train import train
except ImportError:
    # If the environment setup is complex, this fallback is needed.
    # For now, we assume the user fixes the environment path.
    print("FATAL: Cannot import 'train' from 'src.train'. Please check environment.")
    raise


# --- Define the Sweep Configuration (Based on YAML content) ---
sweep_config = {
    'project': 'pii-precision-optimization', 
    'program': 'src/train.py',
    'method': 'bayes',
    'name': 'pii-ner-tuning',
    
    'metric': {
      'name': 'train_loss', 
      'goal': 'minimize'
    },
    
    'parameters': {
      'model_name': {'value': 'distilbert-base-uncased'},
      'train': {'value': 'data/train_unbiased.jsonl'},
      'dev': {'value': 'data/dev_unbiased.jsonl'},
      'out_dir': {'value': 'out/sweep_run'},
      'epochs': {'value': 10}, # Using 10 epochs for faster sweep, not 15
      
      'batch_size': {
        'values': [8, 16, 32, 64] 
      },
      'lr': {
        'min': 1e-5,
        'max': 5e-4, # Corrected max LR to the safer 5e-4 range for fine-tuning
        'distribution': 'uniform'
      },
      'max_length': {
        'values': [64, 128, 256] 
      }
    }
}


def main():
    # 1. Create the Sweep (Generates the ID)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])

    print(f"WandB Sweep Created with ID: {sweep_id}")
    print("Starting agent to run trials. Check the WandB dashboard for progress.")

    # 2. Start the Agent (Executes the 'train' function for each trial)
    # count=10 limits the total number of runs in this agent process
    wandb.agent(sweep_id, function=train, count=13)


if __name__ == "__main__":
    main()
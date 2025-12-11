import json
import matplotlib.pyplot as plt
import os
import glob
import re

# Base path for LoRA outputs
base_path = "/raid/home/nlp4/workspace/projects/workspace/outputs"
output_dir = "/raid/home/nlp4/workspace/legal_mvp_2/static"

def get_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Extract numbers and find the max
    def extract_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else -1
    
    latest_checkpoint = max(checkpoints, key=extract_step)
    return latest_checkpoint

def plot_lora_curves():
    # Find all qlora directories
    model_dirs = glob.glob(os.path.join(base_path, "qlora-sft-*"))
    model_dirs.sort()
    
    if not model_dirs:
        print("No LoRA model directories found.")
        return

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"Processing {model_name}...")
        
        checkpoint_dir = get_latest_checkpoint(model_dir)
        if not checkpoint_dir:
            print(f"  No checkpoints found for {model_name}")
            continue
            
        log_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.exists(log_path):
            print(f"  No trainer_state.json found for {model_name}")
            continue

        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            log_history = data.get("log_history", [])
            
            train_steps = []
            train_loss = []
            eval_steps = []
            eval_loss = []
            
            for entry in log_history:
                if "loss" in entry and "step" in entry:
                    train_steps.append(entry["step"])
                    train_loss.append(entry["loss"])
                if "eval_loss" in entry and "step" in entry:
                    eval_steps.append(entry["step"])
                    eval_loss.append(entry["eval_loss"])
            
            # Create a new figure for each model
            plt.figure(figsize=(10, 6))
            plt.plot(train_steps, train_loss, label='Training Loss')
            if eval_loss:
                plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
            
            plt.title(f'Learning Curve: {model_name}')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save individual plot
            save_filename = f"learning_curve_{model_name}.png"
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path)
            plt.close() # Close the figure to free memory
            
            print(f"  Saved to {save_path}")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_lora_curves()

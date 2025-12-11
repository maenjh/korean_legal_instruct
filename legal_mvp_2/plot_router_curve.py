import json
import matplotlib.pyplot as plt
import os

# Path to the trainer_state.json
log_path = "/raid/home/nlp4/workspace/models/domain_router_classifier/checkpoint-2700/trainer_state.json"
output_path = "/raid/home/nlp4/workspace/legal_mvp_2/static/router_learning_curve.png"

def plot_learning_curve():
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    
    train_steps = []
    train_loss = []
    
    eval_steps = []
    eval_loss = []
    eval_acc = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                eval_acc.append(entry["eval_accuracy"])

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label='Training Loss')
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    if eval_acc:
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, eval_acc, label='Validation Accuracy', color='orange', marker='o')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Learning curve saved to {output_path}")

if __name__ == "__main__":
    plot_learning_curve()

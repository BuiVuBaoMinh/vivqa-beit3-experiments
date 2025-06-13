import json
import argparse
import sys
import matplotlib.pyplot as plt
import os

def get_args():
    parser = argparse.ArgumentParser('Plot max accuracy from log txt.', add_help=False)

    parser.add_argument('--log_path', required=True, type=str,
                        help="Path to log file (e.g log.txt)")
    
    return parser.parse_args()

def plot_beit3_train_max_val(args):
    log_path = args.log_path
    val_scores = []
    train_scores = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line.strip()
            if not line or not line.startswith("{"):
                raise Exception("Found an empty of malformed line!")
            try:
                item = json.loads(line)
                assert 'val_score' in item
                val_scores.append(item['val_score'])
                assert 'train_score' in item
                train_scores.append(item['train_score'])
            except json.JSONDecodeError:
                raise json.JSONDecodeError
    
    if val_scores and train_scores:
        # Plotting
        epochs = list(range(len(val_scores)))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_scores, marker='o', linestyle='-', color='blue', label='Validation Accuracy')
        plt.plot(epochs, train_scores, marker='o', linestyle='-', color='orange', label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = log_path.replace("log.txt", "accuracy.png")
        plt.savefig(plot_path)
        plt.show()

    else:
        print("No valid val_score entries found.")


def plot_beit3_loss(args):
    log_path = args.log_path
    val_losses = []
    train_losses = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line.strip()
            if not line or not line.startswith("{"):
                raise Exception("Found an empty of malformed line!")
            try:
                item = json.loads(line)
                assert 'val_loss' in item
                val_losses.append(item['val_loss'])
                assert 'train_loss' in item
                train_losses.append(item['train_loss'])
            except json.JSONDecodeError:
                raise json.JSONDecodeError
    
    if val_losses and train_losses:
        # Plotting
        epochs = list(range(len(val_losses)))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_losses, marker='o', linestyle='-', color='blue', label='Validation Loss')
        plt.plot(epochs, train_losses, marker='o', linestyle='-', color='orange', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_path = log_path.replace("log.txt", "loss.png")
        plt.savefig(plot_path)
        
        plt.show()
    else:
        print("No valid val_score entries found.")


def main(args):
    plot_beit3_train_max_val(args)
    plot_beit3_loss(args)
    return

if __name__ == "__main__":
    opts = get_args()
    main(opts)
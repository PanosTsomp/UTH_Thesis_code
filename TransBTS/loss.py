import os
import re
import numpy as np
import matplotlib.pyplot as plt


log_file_path = "...."


if not os.path.exists(log_file_path):
    raise FileNotFoundError(f"Log file not found: {log_file_path}")

# Lists to store average metrics per epoch
epoch_losses = []
epoch_dices = []
epoch_sensitivities = []
epoch_precisions = []
epoch_count = 0


current_epoch_losses = []
current_epoch_dices = []
current_epoch_sensitivities = []
current_epoch_precisions = []

def moving_average(data, window_size):
    """Compute the moving average of a list using a sliding window."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Regular expression to match lines with iteration metrics.
# Expected format:
# "Epoch: 0 | Iter: 0 | loss: 2.92404 || 1:0.0760 | 2:0.0000 | 3:0.0000 | Sensitivity: 0.3068 | Precision: 0.2476"
pattern = re.compile(
    r"Epoch:\s*(\d+)\s*\|\s*Iter:\s*(\d+)\s*\|\s*loss:\s*([\d\.]+)\s*\|\|\s*1:([\d\.]+)\s*\|\s*2:([\d\.]+)\s*\|\s*3:([\d\.]+)\s*\|\s*Sensitivity:\s*([\d\.]+)\s*\|\s*Precision:\s*([\d\.]+)"
)

# Parse the log file line by line
with open(log_file_path, "r") as log_file:
    for line in log_file:
        match = pattern.search(line)
        if match is not None:
            try:
                loss = float(match.group(3))
                dice1 = float(match.group(4))
                dice2 = float(match.group(5))
                dice3 = float(match.group(6))
                # Compute the average DICE over the three reported values
                avg_dice = (dice1 + dice2 + dice3) / 3.0
                sensitivity = float(match.group(7))
                precision = float(match.group(8))
            except ValueError:
                continue
            # Append values to current epoch lists
            current_epoch_losses.append(loss)
            current_epoch_dices.append(avg_dice)
            current_epoch_sensitivities.append(sensitivity)
            current_epoch_precisions.append(precision)
        # Look for the line indicating that an epoch has finished.
        elif "Epoch" in line and "finished" in line:
            if current_epoch_losses:
                epoch_losses.append(np.mean(current_epoch_losses))
                epoch_dices.append(np.mean(current_epoch_dices))
                epoch_sensitivities.append(np.mean(current_epoch_sensitivities))
                epoch_precisions.append(np.mean(current_epoch_precisions))
                # Reset temporary lists for the next epoch
                current_epoch_losses = []
                current_epoch_dices = []
                current_epoch_sensitivities = []
                current_epoch_precisions = []
                epoch_count += 1

if epoch_count == 0:
    raise ValueError("No epoch metrics were parsed from the log file.")

# Define the moving average window size
window_size = 20  # Adjust as needed

# ----------------------------------------------------------------
# Function to plot raw and smoothed metrics in one plot
def plot_metric(metric, metric_name, color, filename):
    """Plot the raw and smoothed metric values, then save the plot."""
    plt.figure(figsize=(8, 6))
    # Plot the raw metric values in the given color.
    plt.plot(range(len(metric)), metric, label=f"Avg {metric_name}", color=color)
    if len(metric) >= window_size:
        smoothed = moving_average(metric, window_size)
        # Plot the smoothed line in black.
        plt.plot(range(len(smoothed)), smoothed, label=f"Smoothed (window={window_size})", linestyle='--', color='black')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training {metric_name} per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")

# ----------------------------------------------------------------
# Function to plot only the smoothed metrics in a separate plot
def plot_metric_smoothed(metric, metric_name, color, filename):
    """Plot only the smoothed metric values (in the provided color) and save the plot."""
    plt.figure(figsize=(8, 6))
    if len(metric) >= window_size:
        smoothed = moving_average(metric, window_size)
        plt.plot(range(len(smoothed)), smoothed, label=f"Smoothed (window={window_size})", color=color)
    else:
        # If there's not enough data to compute a moving average,
        # plot the raw data in the provided color with a note.
        plt.plot(range(len(metric)), metric, label=f"Raw {metric_name} (not enough data for smoothing)", color=color)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training {metric_name} per Epoch (Smoothed Only)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")

# ----------------------------------------------------------------
# Create the original plots with both raw and smoothed lines
plot_metric(epoch_losses, "Loss", "blue", "training_loss.png")
plot_metric(epoch_dices, "DICE", "green", "training_dice.png")
plot_metric(epoch_sensitivities, "Sensitivity", "orange", "training_sensitivity.png")
plot_metric(epoch_precisions, "Precision", "red", "training_precision.png")

# Create additional plots with only the smoothed lines
plot_metric_smoothed(epoch_losses, "Loss", "blue", "training_loss_smoothed_only.png")
plot_metric_smoothed(epoch_dices, "DICE","green", "training_dice_smoothed_only.png")
plot_metric_smoothed(epoch_sensitivities, "Sensitivity","orange", "training_sensitivity_smoothed_only.png")
plot_metric_smoothed(epoch_precisions, "Precision","red", "training_precision_smoothed_only.png")
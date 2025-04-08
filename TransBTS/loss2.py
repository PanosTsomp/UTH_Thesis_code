import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


log_file_path = "....."

# Ensure the log file exists
if not os.path.exists(log_file_path):
    raise FileNotFoundError(f"Log file not found: {log_file_path}")

# Lists to store average metrics per epoch
epoch_losses = []
epoch_dices = []
epoch_sensitivities = []
epoch_precisions = []
epoch_count = 0

# Temporary lists to collect iteration metrics for the current epoch
current_epoch_losses = []
current_epoch_dices = []
current_epoch_sensitivities = []
current_epoch_precisions = []

def moving_average(data, window_size):
    """Compute the moving average of a list using a sliding window."""
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return np.concatenate((np.full(window_size - 1, np.nan), smoothed))  # Pad to match original length

# Regular expression to match lines with iteration metrics.
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
                avg_dice = (dice1 + dice2 + dice3) / 3.0
                sensitivity = float(match.group(7))
                precision = float(match.group(8))
            except ValueError:
                continue
            current_epoch_losses.append(loss)
            current_epoch_dices.append(avg_dice)
            current_epoch_sensitivities.append(sensitivity)
            current_epoch_precisions.append(precision)
        elif "Epoch" in line and "finished" in line:
            if current_epoch_losses:
                epoch_losses.append(np.mean(current_epoch_losses))
                epoch_dices.append(np.mean(current_epoch_dices))
                epoch_sensitivities.append(np.mean(current_epoch_sensitivities))
                epoch_precisions.append(np.mean(current_epoch_precisions))
                current_epoch_losses = []
                current_epoch_dices = []
                current_epoch_sensitivities = []
                current_epoch_precisions = []
                epoch_count += 1

if epoch_count == 0:
    raise ValueError("No epoch metrics were parsed from the log file.")

# Define the moving average window size
window_size = 20

# Compute smoothed values
smoothed_losses = moving_average(epoch_losses, window_size)
smoothed_dices = moving_average(epoch_dices, window_size)
smoothed_sensitivities = moving_average(epoch_sensitivities, window_size)
smoothed_precisions = moving_average(epoch_precisions, window_size)

# Save data to CSV
csv_filename = "training_metrics.csv"
df = pd.DataFrame({
    "Epoch": range(len(epoch_losses)),
    "Loss": epoch_losses,
    "Smoothed_Loss": smoothed_losses,
    "DICE": epoch_dices,
    "Smoothed_DICE": smoothed_dices,
    "Sensitivity": epoch_sensitivities,
    "Smoothed_Sensitivity": smoothed_sensitivities,
    "Precision": epoch_precisions,
    "Smoothed_Precision": smoothed_precisions
})
df.to_csv(csv_filename, index=False)
print(f"Metrics saved to {csv_filename}")

def plot_metric(metric, metric_name, color, filename):
    """Plot the raw and smoothed metric values, then save the plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(metric)), metric, label=f"Avg {metric_name}", color=color)
    if len(metric) >= window_size:
        smoothed = moving_average(metric, window_size)
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

plot_metric(epoch_losses, "Loss", "blue", "training_loss.png")
plot_metric(epoch_dices, "DICE", "green", "training_dice.png")
plot_metric(epoch_sensitivities, "Sensitivity", "orange", "training_sensitivity.png")
plot_metric(epoch_precisions, "Precision", "red", "training_precision.png")
import pandas as pd
import matplotlib.pyplot as plt

# Path to your progress file
file_path = 'data/PPO_MBK_DG/PPO_MBK_DG_s0/progress.txt'

# Read the data
data = pd.read_csv(file_path, sep='\s+')

# Display the first few rows to understand the structure
print(data.head())


# Function to plot with smoothing and a vertical line
def plot_with_smoothing_and_marker(ax, x, y, title, xlabel, ylabel, window=10, marker_epoch=500):
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    ax.plot(x, y, label='Original')
    ax.plot(x, y_smooth, label='Smoothed', color='red')
    # ax.axvline(marker_epoch, color='green', linestyle='--', label='Learning Rate Change')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


# Plotting the data
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot AverageEpRet
plot_with_smoothing_and_marker(axs[0, 0], data['Epoch'], data['AverageEpRet'], 'Average Episode Return', 'Epoch',
                               'AverageEpRet')

# Plot StdEpRet
plot_with_smoothing_and_marker(axs[0, 1], data['Epoch'], data['StdEpRet'], 'Standard Deviation of Episode Return',
                               'Epoch', 'StdEpRet')

# Plot MaxEpRet
plot_with_smoothing_and_marker(axs[1, 0], data['Epoch'], data['MaxEpRet'], 'Maximum Episode Return', 'Epoch',
                               'MaxEpRet')

# Plot MinEpRet
plot_with_smoothing_and_marker(axs[1, 1], data['Epoch'], data['MinEpRet'], 'Minimum Episode Return', 'Epoch',
                               'MinEpRet')

# Plot LossPi
plot_with_smoothing_and_marker(axs[2, 0], data['Epoch'], data['LossPi'], 'Policy Loss', 'Epoch', 'LossPi')

# Plot LossV
plot_with_smoothing_and_marker(axs[2, 1], data['Epoch'], data['LossV'], 'Value Loss', 'Epoch', 'LossV')

# Adjust layout
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plotting the data for second player
plot_with_smoothing_and_marker(axs[0, 0], data['Epoch'], data['AverageEpRet_2'], 'Average Episode Return', 'Epoch',
                               'AverageEpRet')

# Plot StdEpRet
plot_with_smoothing_and_marker(axs[0, 1], data['Epoch'], data['StdEpRet_2'], 'Standard Deviation of Episode Return',
                               'Epoch', 'StdEpRet')

# Plot MaxEpRet
plot_with_smoothing_and_marker(axs[1, 0], data['Epoch'], data['MaxEpRet_2'], 'Maximum Episode Return', 'Epoch',
                               'MaxEpRet')

# Plot MinEpRet
plot_with_smoothing_and_marker(axs[1, 1], data['Epoch'], data['MinEpRet_2'], 'Minimum Episode Return', 'Epoch',
                               'MinEpRet')

# Plot LossPi
plot_with_smoothing_and_marker(axs[2, 0], data['Epoch'], data['LossPi_2'], 'Policy Loss', 'Epoch', 'LossPi')

# Plot LossV
plot_with_smoothing_and_marker(axs[2, 1], data['Epoch'], data['LossV_2'], 'Value Loss', 'Epoch', 'LossV')

# Adjust layout
plt.tight_layout()
plt.show()

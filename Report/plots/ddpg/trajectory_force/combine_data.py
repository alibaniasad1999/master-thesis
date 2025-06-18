import pandas as pd
import numpy as np

# Load CSV files
state = pd.read_csv('../../../../Code/Python/TBP/DDPG/results/state.csv')
state.to_csv('state.csv', index=False)
action = pd.read_csv('../../../../Code/Python/TBP/DDPG/results/action.csv')

# Load CSV files
state_zs = pd.read_csv('../../../../Code/Python/TBP/DDPG/DG/results/state.csv')
state_zs.to_csv('state_zs.csv', index=False)
action_zs = pd.read_csv('../../../../Code/Python/TBP/DDPG/DG/results/action.csv')

# Create combined dataframe
combined = pd.DataFrame({
    'x': state['x'],
    'y': state['y'],
    'u': action['ax'],  # Using u/v for quiver plot compatibility
    'v': action['ay']
})

combined_zs = pd.DataFrame({
    'x': state_zs['x'],
    'y': state_zs['y'],
    'u': action_zs['ax'],  # Using u/v for quiver plot compatibility
    'v': action_zs['ay']
})

# Sample evenly across the entire dataset if it's more than 1000 rows
max_samples = 1000
if len(combined) > max_samples:
    # Calculate the step size to get an evenly distributed sample
    step = len(combined) // max_samples
    # Use numpy's linspace to create evenly spaced indices
    indices = np.linspace(0, len(combined)-1, max_samples, dtype=int)
    combined = combined.iloc[indices]

if len(combined_zs) > max_samples:
    # Calculate the step size to get an evenly distributed sample
    step_zs = len(combined_zs) // max_samples
    # Use numpy's linspace to create evenly spaced indices
    indices_zs = np.linspace(0, len(combined_zs)-1, max_samples, dtype=int)
    combined_zs = combined_zs.iloc[indices_zs]

# Save to file
combined.to_csv('combined_data.csv', index=False)
combined_zs.to_csv('combined_data_zs.csv', index=False)

print(f"Created combined_data.csv with {len(combined)} points")
print(f"Created combined_data_zs.csv with {len(combined_zs)} points")
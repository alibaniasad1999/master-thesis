import os
import pandas as pd
import re
import numpy as np


def data4latex(num):
    for i in os.listdir():
        data_dir = (i + '/' + 'evaluation_returns.csv')
        if os.path.isfile(data_dir):
            df = pd.read_csv(data_dir)
            df_100 = df.sample(n=num, random_state=42)
            df = df_100

            # 2) Clean column names to remove non-alphanumeric chars, replace with underscores
            df.columns = [re.sub(r"\W+", "_", col) for col in df.columns]
            data = df

            # 3) Normalize the data (Min-Max scaling to [0,1])
            # Subtract each column's mean and divide by its standard deviation
            df_standardized = (data - np.min(data.mean())) / np.max(data.std())
            print(np.min(data.mean()), np.max(data.std()))

            # print(df_standardized.head())

            # 4) Write the normalized data to a .dat file, using tabs
            save_dir = "../../../../Report/plots/ddpg/violin_plot/" + i + ".dat"
            print(save_dir)
            df_standardized.to_csv(save_dir, sep="\t", index=False)

            print(f"Normalized data written to {i} with cleaned headers.")


if __name__ == "__main__":
    data4latex(10)
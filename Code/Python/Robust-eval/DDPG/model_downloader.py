import os
import subprocess
import ssl
import logging
import urllib
import pandas as pd
import numpy as np

# Disable all Python logging warnings
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set environment variables before any MPI initialization occurs.
os.environ["TMPDIR"] = "/tmp"  # Force the system to use /tmp for temporary files.
os.environ["OMPI_MCA_shmem_mmap_backing_file_base_dir"] = "/tmp"
os.environ["OMPI_MCA_shmem"] = "posix"  # Try forcing a different shmem mechanism.
# Set default HTTPS context to bypass certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(filename_: str):
    trajectory_file = "trajectory.csv"
    trajectory_url = "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/main/Code/Python/TBP/SAC/trajectory.csv"
    # Download trajectory.csv if it doesn't exist

    if not os.path.isfile(trajectory_file):
        print(f"Downloading {trajectory_file} ...")
        urllib.request.urlretrieve(trajectory_url, trajectory_file)
        print(f"{trajectory_file} downloaded.")
    else:
        print(f"{trajectory_file} already exists.")

    df = pd.read_csv('trajectory.csv')
    df.head()
    # df to numpy array
    data = df.to_numpy()
    print(data.shape)
    trajectory = np.delete(data, 2, 1)
    trajectory = np.delete(trajectory, -1, 1)

    # Environment downloader
    env_file = 'TBP.py'
    env_url = "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/main/Code/Python/Environment/TBP.py"

    if not os.path.isfile(env_file):
        print(f"Downloading {env_file} ...")
        subprocess.run(
            ['wget', '-q', '-O', env_file, env_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"{env_file} downloaded.")
    else:
        print(f"{env_file} already exists.")





    if filename_.lower() == 'ddpg':
        # First download group: files for the 'model_zs' directory
        model_dir = 'model_zs'
        urls = [
            'https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/actor_cuda.pth',
            'https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/q_cuda.pth',
            'https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/q_2_cuda.pth',
            'https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/actor_2_cuda.pth'
        ]
        # Create directory if it doesn't exist
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
            print(f"Directory '{model_dir}' created.")
        else:
            if not os.listdir(model_dir):
                print(f"Directory '{model_dir}' exists but is empty.")
            else:
                print(f"Directory '{model_dir}' already exists and is not empty.")

        # Download each file into model_zs using quiet wget
        for url in urls:
            file_path = os.path.join(model_dir, os.path.basename(url))
            if not os.path.isfile(file_path):
                print(f"Downloading {url} to {file_path}...")
                subprocess.run(
                    ['wget', '-q', '-P', model_dir, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                print(f"File {file_path} already exists; skipping download.")

        # Download Zero_Sum_DDPG.py quietly
        zero_sum_script = "Zero_Sum_DDPG.py"
        if not os.path.isfile(zero_sum_script):
            print(f"Downloading {zero_sum_script}...")
            subprocess.run(
                ['wget', '-q', 'https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/Code/Python/Algorithms/DDPG/Zero_Sum_DDPG.py'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print(f"{zero_sum_script} already exists.")

        # Second download group: files for the 'model' directory
        model_dir = 'model'
        urls = [
            'https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/model/actor_cuda.pth',
            'https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/model/q_cuda.pth'
        ]
        # Create directory if it doesn't exist
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
            print(f"Directory '{model_dir}' created.")
        else:
            if not os.listdir(model_dir):
                print(f"Directory '{model_dir}' exists but is empty.")
            else:
                print(f"Directory '{model_dir}' already exists and is not empty.")

        # Download each file into model quietly
        for url in urls:
            file_path = os.path.join(model_dir, os.path.basename(url))
            if not os.path.isfile(file_path):
                print(f"Downloading {url} to {file_path}...")
                subprocess.run(
                    ['wget', '-q', '-P', model_dir, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                print(f"File {file_path} already exists; skipping download.")

        # Download DDPG.py quietly
        ddpg_script = "DDPG.py"
        if not os.path.isfile(ddpg_script):
            print(f"Downloading {ddpg_script}...")
            subprocess.run(
                ['wget', '-q', 'https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/Code/Python/Algorithms/DDPG/DDPG.py'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print(f"{ddpg_script} already exists.")



    return trajectory

# Example call to download the files if needed:
# download_file('ddpg')

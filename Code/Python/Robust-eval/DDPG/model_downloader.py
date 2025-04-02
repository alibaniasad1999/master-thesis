import os
import subprocess
import ssl
import logging
import urllib.request
import pandas as pd
import numpy as np

# Disable all Python logging warnings
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set environment variables before any MPI initialization occurs.
os.environ["TMPDIR"] = "/tmp"  # Force the system to use /tmp for temporary files.
os.environ["OMPI_MCA_shmem_mmap_backing_file_base_dir"] = "/tmp"
os.environ["OMPI_MCA_shmem"] = "posix"  # Force a different shmem mechanism.
# Bypass HTTPS certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


def download_file(url: str, dest: str, use_wget: bool = False) -> None:
    """
    Download a file from a URL to a destination path.
    Uses wget if use_wget is True; otherwise, uses urllib.
    """
    if not os.path.isfile(dest):
        print(f"Downloading {dest} ...")
        if use_wget:
            subprocess.run(
                ['wget', '-q', '-O', dest, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            urllib.request.urlretrieve(url, dest)
        print(f"{dest} downloaded.")
    else:
        print(f"{dest} already exists.")


def download_tbp() -> np.ndarray:
    """
    Downloads the trajectory file and TBP.py.
    Processes trajectory.csv into a NumPy array by removing the 3rd and last columns.
    Returns the processed trajectory array.
    """
    # Download trajectory.csv
    trajectory_file = "trajectory.csv"
    trajectory_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/"
        "main/Code/Python/TBP/SAC/trajectory.csv"
    )
    download_file(trajectory_url, trajectory_file, use_wget=False)

    # Process trajectory.csv
    df = pd.read_csv(trajectory_file)
    print("Trajectory head:")
    print(df.head())
    data = df.to_numpy()
    print("Data shape:", data.shape)
    # Remove the 3rd column (index 2) and the last column
    trajectory = np.delete(data, 2, axis=1)
    trajectory = np.delete(trajectory, -1, axis=1)

    # Download TBP.py
    tbp_file = "TBP.py"
    tbp_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/"
        "main/Code/Python/Environment/TBP.py"
    )
    download_file(tbp_url, tbp_file, use_wget=True)

    return trajectory


def download_ddpg_script() -> None:
    """
    Downloads only the DDPG.py script.
    """
    ddpg_script = "DDPG.py"
    ddpg_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/DDPG/DDPG.py"
    )
    download_file(ddpg_url, ddpg_script, use_wget=True)


def download_zs_ddpg_script() -> None:
    """
    Downloads only the Zero_Sum_DDPG.py script.
    """
    zero_sum_script = "Zero_Sum_DDPG.py"
    zero_sum_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/DDPG/Zero_Sum_DDPG.py"
    )
    download_file(zero_sum_url, zero_sum_script, use_wget=True)


def download_models() -> None:
    """
    Downloads only the standard model files (from the 'model' directory).
    """
    model_dir = "model"
    urls_model = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/model/actor_cuda.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/model/q_cuda.pth"
    ]
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        print(f"Directory '{model_dir}' created.")
    else:
        if not os.listdir(model_dir):
            print(f"Directory '{model_dir}' exists but is empty.")
        else:
            print(f"Directory '{model_dir}' already exists and is not empty.")
    for url in urls_model:
        file_path = os.path.join(model_dir, os.path.basename(url))
        download_file(url, file_path, use_wget=True)


def download_zs_models() -> None:
    """
    Downloads only the zero‑sum model files (from the 'model_zs' directory).
    """
    model_zs_dir = "model_zs"
    urls_model_zs = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/actor_cuda.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/q_cuda.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/q_2_cuda.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/DDPG/DG/model/actor_2_cuda.pth"
    ]
    if not os.path.isdir(model_zs_dir):
        os.makedirs(model_zs_dir)
        print(f"Directory '{model_zs_dir}' created.")
    else:
        if not os.listdir(model_zs_dir):
            print(f"Directory '{model_zs_dir}' exists but is empty.")
        else:
            print(f"Directory '{model_zs_dir}' already exists and is not empty.")
    for url in urls_model_zs:
        file_path = os.path.join(model_zs_dir, os.path.basename(url))
        download_file(url, file_path, use_wget=True)


def download_everything(input_keywords) -> np.ndarray:
    """
    Downloads files based on the provided keyword(s). Accepted keywords (case-insensitive):
      - "TBP": Downloads the trajectory and TBP.py.
      - "DDPG": Downloads only the DDPG.py script.
      - "zs_DDPG": Downloads only the Zero_Sum_DDPG.py script.
      - "MODELS": Downloads only the standard model files (from the 'model' directory).
      - "zs_MODELS": Downloads only the zero‑sum model files (from the 'model_zs' directory).

    You can provide a comma-separated string (e.g., "DDPG, TBP") or a list of keywords.

    Returns the processed trajectory array if TBP files are downloaded; otherwise, returns None.
    """
    # Normalize input keywords to a list of uppercase strings.
    if isinstance(input_keywords, str):
        keywords = [kw.strip().upper() for kw in input_keywords.split(",")]
    else:
        keywords = [str(kw).upper() for kw in input_keywords]

    trajectory = None
    if "TBP" in keywords:
        trajectory = download_tbp()
    if "DDPG" in keywords:
        download_ddpg_script()
    if "ZS_DDPG" in keywords:
        download_zs_ddpg_script()
    if "MODELS" in keywords:
        download_models()
    if "ZS_MODELS" in keywords:
        download_zs_models()

    return trajectory


if __name__ == "__main__":
    # Example usage:
    # "DDPG, TBP" downloads only TBP files and the standard DDPG.py script.
    # "zs_DDPG" downloads only the Zero_Sum_DDPG.py script.
    # "MODELS" or "zs_MODELS" downloads only the respective model files.
    user_input = input(
        "Enter download keyword(s) (e.g., 'TBP', 'DDPG, TBP', 'zs_DDPG', 'MODELS', 'zs_MODELS'): ").strip()
    result = download_everything(user_input)
    if result is not None:
        print("Processed trajectory array shape:", result.shape)

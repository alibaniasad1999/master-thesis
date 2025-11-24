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
        "main/Code/Python/TBP/SAC/legacy/trajectory.csv"
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


def download_ddpg_models(folder_name=None) -> None:
    """
    Downloads only the standard model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_dir = folder_name if folder_name else "model"
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

def download_ppo_models(folder_name=None) -> None:
    """
    Downloads only the standard model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_dir = folder_name if folder_name else "model"
    urls_model = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/PPO/Standard/model/actor_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/PPO/Standard/model/v_cpu.pth"
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

def download_ddpg_zs_models(folder_name=None) -> None:
    """
    Downloads only the zero‑sum model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_zs_dir = folder_name if folder_name else "model_zs"
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

def download_ppo_zs_models(folder_name=None) -> None:
    """
    Downloads only the zero‑sum model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_zs_dir = folder_name if folder_name else "model_zs"
    urls_model_zs = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/PPO/ZeroSum/model/actor_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/PPO/ZeroSum/model/actor_1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/PPO/ZeroSum/model/v_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/PPO/ZeroSum/model/v_1_cpu.pth"
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

def download_ppo_script() -> None:
    """
    Downloads only the PPO.py script.
    """
    ppo_script = "PPO.py"
    ppo_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
    "Code/Python/Algorithms/PPO/PPO.py"
    )
    download_file(ppo_url, ppo_script, use_wget=True)

def download_sac_script() -> None:
    """
    Downloads only the SAC.py script.
    """
    sac_script = "SAC.py"
    sac_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/SAC/SAC.py"
    )
    download_file(sac_url, sac_script, use_wget=True)

def download_zs_ppo_script() -> None:
    """
    Downloads only the Zero_Sum_PPO.py script.
    """
    zero_sum_script = "Zero_Sum_PPO.py"
    zero_sum_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/PPO/Zero_Sum_PPO.py"
    )
    download_file(zero_sum_url, zero_sum_script, use_wget=True)



def download_sac_models(folder_name=None) -> None:
    """
    Downloads only the standard model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_dir = folder_name if folder_name else "model"
    urls_model = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/Standard/model/actor_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/Standard/model/q1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/Standard/model/q2_cpu.pth"
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


def download_zs_sac_script() -> None:
    """
    Downloads only the Zero_Sum_SAC.py script.
    """
    zero_sum_script = "Zero_Sum_SAC.py"
    zero_sum_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/SAC/Zero_Sum_SAC.py"
    )
    download_file(zero_sum_url, zero_sum_script, use_wget=True)

def download_sac_zs_models(folder_name=None) -> None:
    """
    Downloads only the zero‑sum model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_zs_dir = folder_name if folder_name else "model_zs"
    urls_model_zs = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/ZeroSum/model/actor_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/ZeroSum/model/actor_1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/ZeroSum/model/q1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/ZeroSum/model/q1_1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/ZeroSum/model/q2_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/SAC/ZeroSum/model/q2_1_cpu.pth"
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

def download_td3_script() -> None:
    """
    Downloads only the TD3.py script.
    """
    td3_script = "TD3.py"
    td3_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/TD3/TD3.py"
    )
    download_file(td3_url, td3_script, use_wget=True)

def download_td3_models(folder_name=None) -> None:
    """
    Downloads only the standard model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_dir = folder_name if folder_name else "model"
    urls_model = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/Standard/model/actor_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/Standard/model/q1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/Standard/model/q2_cpu.pth"
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

def download_zs_td3_script() -> None:
    """
    Downloads only the Zero_Sum_TD3.py script.
    """
    zero_sum_script = "Zero_Sum_TD3.py"
    zero_sum_url = (
        "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/refs/heads/main/"
        "Code/Python/Algorithms/TD3/Zero_Sum_TD3.py"
    )
    download_file(zero_sum_url, zero_sum_script, use_wget=True)

def download_zs_td3_models(folder_name=None) -> None:
    """
    Downloads only the zero‑sum model files.

    Args:
        folder_name: Optional custom folder name to save models in
    """
    model_zs_dir = folder_name if folder_name else "model_zs"
    urls_model_zs = [
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/ZeroSum/model/actor_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/ZeroSum/model/actor_1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/ZeroSum/model/q1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/ZeroSum/model/q1_1_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/ZeroSum/model/q2_cpu.pth",
        "https://github.com/alibaniasad1999/master-thesis/raw/main/Code/Python/TBP/TD3/ZeroSum/model/q2_1_cpu.pth"
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

def download_everything(input_keywords, folder_name=None) -> np.ndarray:
    """
    Downloads files based on the provided keyword(s). Accepted keywords (case-insensitive):
      - "TBP": Downloads the trajectory and TBP.py.
      - "DDPG": Downloads only the DDPG.py script.
      - "zs_DDPG": Downloads only the Zero_Sum_DDPG.py script.
      - "MODELS": Downloads only the standard model files.
      - "zs_MODELS": Downloads only the zero‑sum model files.

    You can provide a comma-separated string (e.g., "DDPG, TBP") or a list of keywords.

    Args:
        input_keywords: Keywords specifying what to download
        folder_name: If True, adds algorithm name to model folder; if False or None, uses default folder names

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

    # Handle folder names for model downloads
    ddpg_folder = "model_DDPG" if folder_name is True else folder_name
    ddpg_zs_folder = "model_zs_DDPG" if folder_name is True else folder_name
    ppo_folder = "model_PPO" if folder_name is True else folder_name
    ppo_zs_folder = "model_zs_PPO" if folder_name is True else folder_name
    sac_folder = "model_SAC" if folder_name is True else folder_name
    sac_zs_folder = "model_zs_SAC" if folder_name is True else folder_name
    td3_folder = "model_TD3" if folder_name is True else folder_name
    td3_zs_folder = "model_zs_TD3" if folder_name is True else folder_name

    if "MODELS" in keywords:
        download_ddpg_models(ddpg_folder)
    if "DDPG_MODELS" in keywords:
        download_ddpg_models(ddpg_folder)
    if "ZS_MODELS" in keywords:
        download_ddpg_zs_models(ddpg_zs_folder)
    if "PPO" in keywords:
        download_ppo_script()
    if "PPO_MODELS" in keywords:
        download_ppo_models(ppo_folder)
    if "ZS_PPO" in keywords:
        download_zs_ppo_script()
    if "ZS_PPO_MODELS" in keywords:
        download_ppo_zs_models(ppo_zs_folder)
    if "SAC" in keywords:
        download_sac_script()
    if "SAC_MODELS" in keywords:
        download_sac_models(sac_folder)
    if "ZS_SAC" in keywords:
        download_zs_sac_script()
    if "ZS_SAC_MODELS" in keywords:
        download_sac_zs_models(sac_zs_folder)
    if "TD3" in keywords:
        download_td3_script()
    if "TD3_MODELS" in keywords:
        download_td3_models(td3_folder)
    if "ZS_TD3" in keywords:
        download_zs_td3_script()
    if "ZS_TD3_MODELS" in keywords:
        download_zs_td3_models(td3_zs_folder)

    return trajectory


if __name__ == "__main__":
    # Example usage:
    # "DDPG, TBP" downloads only TBP files and the standard DDPG.py script.
    # "zs_DDPG" downloads only the Zero_Sum_DDPG.py script.
    # "MODELS" or "zs_MODELS" downloads only the respective model files.
    user_input = input(
        "Enter download keyword(s) (e.g., 'TBP', 'DDPG, TBP', 'zs_DDPG', 'MODELS', 'zs_MODELS'): ").strip()

    # Ask for custom folder naming
    folder_input = input("Use algorithm-specific folder names? (yes/no): ").strip().lower()
    custom_folder = True if folder_input in ['yes', 'y', 'true', '1'] else None

    result = download_everything(user_input, folder_name=custom_folder)
    if result is not None:
        print("Processed trajectory array shape:", result.shape)

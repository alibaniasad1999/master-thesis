<div align="center">

# Robust Reinforcement Learning Differential Game Guidance in Low-Thrust, Multi-Body Dynamical Environments

### Master's Thesis

**Ali Bani Asad**

Department of Aerospace Engineering  
Sharif University of Technology

Supervised by: **Dr. Hadi Nobahari**  

*September 2025*

---

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Abstract

This repository contains the complete implementation and documentation of a **zero-sum multi-agent reinforcement learning (MARL)** framework for robust spacecraft guidance in the challenging Earth-Moon three-body dynamical system. The research addresses the critical problem of low-thrust spacecraft guidance under significant environmental uncertainties through a novel differential game formulation.

**Key Contributions:**
- 🎮 **Zero-Sum Game Formulation**: Spacecraft guidance cast as a two-player differential game between a guidance agent (spacecraft) and a disturbance agent (uncertainties)
- 🤖 **Multi-Agent RL Algorithms**: Extended implementations of DDPG, TD3, SAC, and PPO to their zero-sum multi-agent variants (MA-DDPG, MA-TD3, MA-SAC, MA-PPO)
- 🛡️ **Robustness Analysis**: Comprehensive evaluation under diverse uncertainty scenarios including sensor noise, actuator disturbances, time delays, model mismatch, and initial condition variations
- 🚀 **Hardware Integration**: ROS2-based implementation with C++ inference for real-time deployment
- 📊 **Benchmark Comparison**: Rigorous comparison against classical control methods and standard single-agent RL approaches

**Results**: The zero-sum MARL approach demonstrates superior robustness, with MA-TD3 achieving the best performance in trajectory tracking and fuel efficiency while maintaining stability in highly perturbed environments.

---

## 🏗️ Repository Structure

```
master-thesis/
├── 📚 Report/                      # LaTeX thesis document
│   ├── thesis.tex                  # Main thesis file
│   ├── Chapters/                   # 8 chapters (Introduction → Conclusion)
│   ├── bibs/                       # Bibliography
│   └── plots/                      # Result plots and figures
│
├── 📜 Paper/                       # Conference paper (IEEE format)
│
├── 💻 Code/
│   ├── Python/
│   │   ├── Algorithms/             # DDPG, TD3, SAC, PPO implementations
│   │   ├── Environment/            # Three-body problem dynamics (TBP.py)
│   │   ├── TBP/                    # Single-agent training (Classic, DDPG, TD3, SAC, PPO)
│   │   ├── MBK/                    # Multi-body Kepler experiments
│   │   ├── Robust_eval/            # Robustness testing (Standard & ZeroSum variants)
│   │   ├── Benchmark/              # OpenAI Gym environments
│   │   └── utils/                  # Utility functions
│   │
│   ├── C/                          # C++ real-time inference (PyTorch models)
│   ├── ROS2/                       # ROS2 packages for hardware integration
│   ├── Simulink/                   # MATLAB Simulink models
│   └── ros_legacy/                 # Legacy ROS1 implementation
│
├── 🖼️ Figure/                      # Visualizations (TBP, HIL)
├── 🎓 Presentation/                # Defense slides (Beamer)
└── 📖 Proposal/                    # Research proposal
```

**Key Directories:**
- `Code/Python/Algorithms/`: Core RL algorithm implementations
- `Code/Python/TBP/`: Training notebooks for single-agent baseline
- `Code/Python/Robust_eval/`: Comprehensive robustness evaluation scripts
- `Report/`: Complete thesis document with LaTeX source

---

## 🔬 Research Methodology

### Problem Formulation

The spacecraft guidance problem in the **Circular Restricted Three-Body Problem (CR3BP)** is formulated as a **zero-sum differential game**:

- **Player 1 (Guidance Agent)**: Minimizes trajectory deviation and fuel consumption
- **Player 2 (Disturbance Agent)**: Maximizes trajectory deviation (models worst-case uncertainties)

This formulation enables the development of inherently robust control policies that perform well under adversarial conditions.

### Multi-Agent RL Algorithms

Four state-of-the-art continuous control algorithms are extended to their zero-sum multi-agent variants:

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **MA-DDPG** | Off-policy, Deterministic | Simple, efficient, good baseline |
| **MA-TD3** | Off-policy, Deterministic | Target policy smoothing, delayed updates, clipped double Q-learning |
| **MA-SAC** | Off-policy, Stochastic | Maximum entropy, automatic temperature tuning |
| **MA-PPO** | On-policy, Stochastic | Trust region optimization, robust training |

### Training Strategy

- **Centralized Training, Decentralized Execution (CTDE)**: Both agents observe full state during training but can act independently during deployment
- **Alternating Optimization**: Sequential training of guidance and disturbance agents
- **Full Information Setting**: Complete state observation for optimal policy learning

### Robustness Evaluation

The trained policies are rigorously tested under six uncertainty scenarios:

1. 🎲 **Initial Condition Variations**: Random perturbations in initial state
2. ⚡ **Actuator Disturbances**: Thrust vector perturbations
3. 📡 **Sensor Noise**: Gaussian noise in state measurements
4. ⏱️ **Time Delays**: Communication and actuation delays
5. 🔧 **Model Mismatch**: Errors in system dynamics model
6. 🌪️ **Combined Uncertainties**: All scenarios simultaneously

---

## 🚀 Getting Started

### Prerequisites

**Software Requirements:**
- Python 3.8 or higher
- PyTorch 2.2.2
- CUDA 11.8+ (optional, for GPU acceleration)
- ROS2 Humble (for hardware deployment)
- CMake 3.16+ (for C++ implementation)
- LaTeX distribution (for compiling thesis document)

**Hardware Requirements:**
- 16+ GB RAM (recommended for training)
- NVIDIA GPU with 6+ GB VRAM (optional, speeds up training significantly)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/alibaniasad1999/master-thesis.git
cd master-thesis
```

#### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 3. Verify Installation

```bash
python -c "import torch; import gymnasium; import numpy; print('✓ All packages installed successfully')"
```

---

## 💡 Usage Guide

### Training RL Agents

#### Single-Agent Training (Baseline)

```bash
cd Code/Python/TBP/SAC
jupyter notebook SAC_TBP.ipynb
```

Follow the notebook to:
1. Configure environment parameters
2. Set hyperparameters
3. Train the agent
4. Evaluate performance
5. Save trained models

#### Zero-Sum Multi-Agent Training

```bash
cd Code/Python/TBP/SAC/ZeroSum
jupyter notebook Zero_Sum_SAC_TBP.ipynb
```

The notebook demonstrates:
1. Zero-sum game setup
2. Alternating training procedure
3. Nash equilibrium convergence
4. Robustness evaluation

### Robustness Evaluation

```bash
cd Code/Python/Robust_eval/ZeroSum/sensor_noise
jupyter notebook sensor_noise.ipynb
```

This evaluates trained policies under sensor noise perturbations and generates comparison plots.

### C++ Inference (Real-Time Deployment)

```bash
cd Code/C
mkdir build && cd build
cmake ..
make
./main
```

The C++ implementation loads PyTorch traced models for fast inference.

### ROS2 Integration

```bash
cd Code/ROS2
colcon build
source install/setup.bash
ros2 launch tbp_rl_controler tbp_system.launch.py
```

This launches:
- Three-body dynamics simulator node
- RL controller node
- Data logging node

### Model Download Utility

```bash
cd Code/Python/utils
python model_downloader.py
```

Downloads pre-trained models from the GitHub repository.

---

## 📊 Key Results

### Performance Comparison

| Algorithm | Trajectory Error (m) | Fuel Consumption (m/s) | Success Rate (%) | Robustness Score |
|-----------|---------------------|------------------------|------------------|------------------|
| PID Control | 8,432 ± 2,156 | 45.2 ± 8.3 | 72.4 | ⭐⭐ |
| DDPG | 1,234 ± 892 | 28.7 ± 5.2 | 84.6 | ⭐⭐⭐ |
| TD3 | 967 ± 654 | 26.4 ± 4.1 | 88.2 | ⭐⭐⭐⭐ |
| SAC | 1,045 ± 721 | 27.8 ± 4.8 | 86.9 | ⭐⭐⭐⭐ |
| PPO | 1,398 ± 978 | 31.2 ± 6.3 | 81.5 | ⭐⭐⭐ |
| **MA-DDPG** | 892 ± 423 | 25.1 ± 3.2 | 91.7 | ⭐⭐⭐⭐ |
| **MA-TD3** | **687 ± 312** | **23.4 ± 2.8** | **95.3** | ⭐⭐⭐⭐⭐ |
| **MA-SAC** | 734 ± 367 | 24.2 ± 3.1 | 93.8 | ⭐⭐⭐⭐⭐ |
| **MA-PPO** | 856 ± 445 | 26.7 ± 3.9 | 90.4 | ⭐⭐⭐⭐ |

*Results averaged over 1,000 test episodes with combined uncertainty scenarios.*

### Trajectory Tracking Performance: TD3

<div align="center">

#### TD3: Standard vs Zero-Sum MA-TD3 Comparison

**Trajectory Tracking**

<table>
<tr>
<td align="center"><b>Standard TD3 Trajectory</b></td>
<td align="center"><b>Zero-Sum MA-TD3 Trajectory</b></td>
</tr>
<tr>
<td><img src="Report/plots/td3/trajectory_force/plot_trajectory.png" width="400" alt="Standard TD3 Trajectory"/></td>
<td><img src="Report/plots/td3/trajectory_force/plot_trajectory_zs.png" width="400" alt="Zero-Sum MA-TD3 Trajectory"/></td>
</tr>
</table>

**Trajectory with Control Forces**

<table>
<tr>
<td align="center"><b>Standard TD3</b></td>
<td align="center"><b>Zero-Sum MA-TD3</b></td>
</tr>
<tr>
<td><img src="Report/plots/td3/trajectory_force/plot_trajectory_force.png" width="400" alt="Standard TD3 with Forces"/></td>
<td><img src="Report/plots/td3/trajectory_force/plot_trajectory_force_zs.png" width="400" alt="Zero-Sum MA-TD3 with Forces"/></td>
</tr>
</table>

*MA-TD3 demonstrates superior trajectory tracking with reduced deviation and more efficient control force usage.*

</div>

---

### Robustness Analysis Under Uncertainty

#### Comparative Performance: All Four Algorithms

The violin plots below show the performance distribution of all four RL algorithms (DDPG, TD3, SAC, PPO) under various uncertainty scenarios. Each plot compares **Standard (single-agent)** vs **Zero-Sum (multi-agent)** variants.

<div align="center">

**Zero-Sum Multi-Agent RL - All Algorithms Combined**

<table>
<tr>
<td align="center"><b>Actuator Disturbance</b></td>
<td align="center"><b>Sensor Noise</b></td>
</tr>
<tr>
<td><img src="Report/plots/ZeroSum/violin_plot/actuator_disturbance.png" width="400" alt="Actuator Disturbance - ZS"/></td>
<td><img src="Report/plots/ZeroSum/violin_plot/sensor_noise.png" width="400" alt="Sensor Noise - ZS"/></td>
</tr>
<tr>
<td align="center"><b>Initial Condition Shift</b></td>
<td align="center"><b>Time Delay</b></td>
</tr>
<tr>
<td><img src="Report/plots/ZeroSum/violin_plot/initial_condition_shift.png" width="400" alt="Initial Condition - ZS"/></td>
<td><img src="Report/plots/ZeroSum/violin_plot/time_delay.png" width="400" alt="Time Delay - ZS"/></td>
</tr>
<tr>
<td align="center"><b>Model Mismatch</b></td>
<td align="center"><b>Partial Observation</b></td>
</tr>
<tr>
<td><img src="Report/plots/ZeroSum/violin_plot/model_mismatch.png" width="400" alt="Model Mismatch - ZS"/></td>
<td><img src="Report/plots/ZeroSum/violin_plot/partial_observation.png" width="400" alt="Partial Observation - ZS"/></td>
</tr>
</table>

---

**Standard Single-Agent RL - All Algorithms Combined**

<table>
<tr>
<td align="center"><b>Actuator Disturbance</b></td>
<td align="center"><b>Sensor Noise</b></td>
</tr>
<tr>
<td><img src="Report/plots/standard/violin_plot/actuator_disturbance.png" width="400" alt="Actuator Disturbance - Standard"/></td>
<td><img src="Report/plots/standard/violin_plot/sensor_noise.png" width="400" alt="Sensor Noise - Standard"/></td>
</tr>
<tr>
<td align="center"><b>Initial Condition Shift</b></td>
<td align="center"><b>Time Delay</b></td>
</tr>
<tr>
<td><img src="Report/plots/standard/violin_plot/initial_condition_shift.png" width="400" alt="Initial Condition - Standard"/></td>
<td><img src="Report/plots/standard/violin_plot/time_delay.png" width="400" alt="Time Delay - Standard"/></td>
</tr>
<tr>
<td align="center"><b>Model Mismatch</b></td>
<td align="center"><b>Partial Observation</b></td>
</tr>
<tr>
<td><img src="Report/plots/standard/violin_plot/model_mismatch.png" width="400" alt="Model Mismatch - Standard"/></td>
<td><img src="Report/plots/standard/violin_plot/partial_observation.png" width="400" alt="Partial Observation - Standard"/></td>
</tr>
</table>

</div>

<details>
<summary><b>📊 Click to view individual algorithm robustness (TD3)</b></summary>

#### TD3 Robustness Evaluation

<table>
<tr>
<td align="center"><b>Actuator Disturbance</b></td>
<td align="center"><b>Sensor Noise</b></td>
<td align="center"><b>Initial Condition Shift</b></td>
</tr>
<tr>
<td><img src="Report/plots/td3/violin_plot/actuator_disturbance.png" width="250" alt="TD3 Actuator"/></td>
<td><img src="Report/plots/td3/violin_plot/sensor_noise.png" width="250" alt="TD3 Sensor"/></td>
<td><img src="Report/plots/td3/violin_plot/initial_condition_shift.png" width="250" alt="TD3 Initial"/></td>
</tr>
<tr>
<td align="center"><b>Time Delay</b></td>
<td align="center"><b>Model Mismatch</b></td>
<td align="center"><b>Partial Observation</b></td>
</tr>
<tr>
<td><img src="Report/plots/td3/violin_plot/time_delay.png" width="250" alt="TD3 Delay"/></td>
<td><img src="Report/plots/td3/violin_plot/model_mismatch.png" width="250" alt="TD3 Mismatch"/></td>
<td><img src="Report/plots/td3/violin_plot/partial_observation.png" width="250" alt="TD3 Partial"/></td>
</tr>
</table>

</details>

<details>
<summary><b>📊 Click to view individual algorithm robustness (DDPG)</b></summary>

#### DDPG Robustness Evaluation

<table>
<tr>
<td align="center"><b>Actuator Disturbance</b></td>
<td align="center"><b>Sensor Noise</b></td>
<td align="center"><b>Initial Condition Shift</b></td>
</tr>
<tr>
<td><img src="Report/plots/ddpg/violin_plot/actuator_disturbance.png" width="250" alt="DDPG Actuator"/></td>
<td><img src="Report/plots/ddpg/violin_plot/sensor_noise.png" width="250" alt="DDPG Sensor"/></td>
<td><img src="Report/plots/ddpg/violin_plot/initial_condition_shift.png" width="250" alt="DDPG Initial"/></td>
</tr>
<tr>
<td align="center"><b>Time Delay</b></td>
<td align="center"><b>Model Mismatch</b></td>
<td align="center"><b>Partial Observation</b></td>
</tr>
<tr>
<td><img src="Report/plots/ddpg/violin_plot/time_delay.png" width="250" alt="DDPG Delay"/></td>
<td><img src="Report/plots/ddpg/violin_plot/model_mismatch.png" width="250" alt="DDPG Mismatch"/></td>
<td><img src="Report/plots/ddpg/violin_plot/partial_observation.png" width="250" alt="DDPG Partial"/></td>
</tr>
</table>

</details>

<details>
<summary><b>📊 Click to view individual algorithm robustness (SAC)</b></summary>

#### SAC Robustness Evaluation

<table>
<tr>
<td align="center"><b>Actuator Disturbance</b></td>
<td align="center"><b>Sensor Noise</b></td>
<td align="center"><b>Initial Condition Shift</b></td>
</tr>
<tr>
<td><img src="Report/plots/sac/violin_plot/actuator_disturbance.png" width="250" alt="SAC Actuator"/></td>
<td><img src="Report/plots/sac/violin_plot/sensor_noise.png" width="250" alt="SAC Sensor"/></td>
<td><img src="Report/plots/sac/violin_plot/initial_condition_shift.png" width="250" alt="SAC Initial"/></td>
</tr>
<tr>
<td align="center"><b>Time Delay</b></td>
<td align="center"><b>Model Mismatch</b></td>
<td align="center"><b>Partial Observation</b></td>
</tr>
<tr>
<td><img src="Report/plots/sac/violin_plot/time_delay.png" width="250" alt="SAC Delay"/></td>
<td><img src="Report/plots/sac/violin_plot/model_mismatch.png" width="250" alt="SAC Mismatch"/></td>
<td><img src="Report/plots/sac/violin_plot/partial_observation.png" width="250" alt="SAC Partial"/></td>
</tr>
</table>

</details>

<details>
<summary><b>📊 Click to view individual algorithm robustness (PPO)</b></summary>

#### PPO Robustness Evaluation

<table>
<tr>
<td align="center"><b>Actuator Disturbance</b></td>
<td align="center"><b>Sensor Noise</b></td>
<td align="center"><b>Initial Condition Shift</b></td>
</tr>
<tr>
<td><img src="Report/plots/ppo/violin_plot/actuator_disturbance.png" width="250" alt="PPO Actuator"/></td>
<td><img src="Report/plots/ppo/violin_plot/sensor_noise.png" width="250" alt="PPO Sensor"/></td>
<td><img src="Report/plots/ppo/violin_plot/initial_condition_shift.png" width="250" alt="PPO Initial"/></td>
</tr>
<tr>
<td align="center"><b>Time Delay</b></td>
<td align="center"><b>Model Mismatch</b></td>
<td align="center"><b>Partial Observation</b></td>
</tr>
<tr>
<td><img src="Report/plots/ppo/violin_plot/time_delay.png" width="250" alt="PPO Delay"/></td>
<td><img src="Report/plots/ppo/violin_plot/model_mismatch.png" width="250" alt="PPO Mismatch"/></td>
<td><img src="Report/plots/ppo/violin_plot/partial_observation.png" width="250" alt="PPO Partial"/></td>
</tr>
</table>

</details>

---

### Key Findings

✅ **Zero-sum MARL outperforms single-agent RL** across all metrics  
✅ **MA-TD3 achieves best overall performance** with 30% error reduction vs. TD3  
✅ **Robustness significantly improved** under all uncertainty scenarios  
✅ **Tighter performance distributions** in zero-sum variants (visible in violin plots)  
✅ **Stable performance** in highly perturbed environments  
✅ **Real-time capable** C++ implementation achieves <5ms inference time

---

## 📖 Documentation

### Thesis Document

The complete thesis is available in the `Report/` directory:

```bash
cd Report
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

Or use `latexmk` for automatic compilation:

```bash
latexmk -pdf thesis.tex
```

### Chapter Overview

1. **Introduction**: Motivation, problem statement, and research objectives
2. **Literature Review**: Survey of RL, MARL, differential games, and spacecraft guidance
3. **Simulation**: Three-body problem dynamics and environment setup
4. **Reinforcement Learning**: Single-agent RL algorithms (DDPG, TD3, SAC, PPO)
5. **Agent Simulation**: Training procedures and baseline results
6. **Multi-Agent RL**: Zero-sum game formulation and MARL algorithms
7. **Results**: Comprehensive evaluation and comparison
8. **Conclusion**: Summary, contributions, and future work

### API Documentation

Key classes and functions are documented in the code:

- `Environment/TBP.py`: Three-body problem environment class
- `Algorithms/*/Zero_Sum_*.py`: Zero-sum MARL implementations
- `utils/model_downloader.py`: Pre-trained model utilities

---

## 🎯 Reproducibility

### Reproduce Training Results

```bash
# Train MA-TD3 agent
cd Code/Python/TBP/TD3/ZeroSum
jupyter notebook Zero_Sum_TD3_TBP.ipynb
# Execute all cells
```

### Reproduce Evaluation Results

```bash
# Run robustness evaluation
cd Code/Python/Robust_eval/ZeroSum/All_in_one/actuator_disturbance
jupyter notebook all_in_one.ipynb
```

### Random Seeds

All experiments use fixed random seeds for reproducibility:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- Gymnasium: `env.seed(42)`

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{baniasad2025robust,
  author       = {Ali Bani Asad},
  title        = {Robust Reinforcement Learning Differential Game Guidance 
                  in Low-Thrust, Multi-Body Dynamical Environments},
  school       = {Sharif University of Technology},
  year         = {2025},
  address      = {Tehran, Iran},
  month        = {September},
  type         = {Master's Thesis},
  note         = {Department of Aerospace Engineering}
}
```

### Related Publications

- **Conference Paper**: "Robustness on Demand: Transformer-Directed Switching in Multi-Agent RL" (in preparation)

---

## 🤝 Contributing

This is an academic research repository. While it's primarily for archival and reference purposes, suggestions and discussions are welcome:

1. Open an issue to discuss proposed changes
2. Fork the repository
3. Create a feature branch
4. Submit a pull request with detailed description

---

## 📧 Contact

**Ali Bani Asad**  
Department of Aerospace Engineering  
Sharif University of Technology  
📧 Email: ali_baniasad@ae.sharif.edu  
🔗 GitHub: [@alibaniasad1999](https://github.com/alibaniasad1999)

**Supervisor: Dr. Hadi Nobahari**  
📧 Email: nobahari@sharif.edu

---

## 🙏 Acknowledgments

This research was conducted at the **Sharif University of Technology**, Department of Aerospace Engineering, under the supervision of **Dr. Hadi Nobahari** and the advisory of **Dr. Seyed Ali Emami Khooansari**.

Special thanks to:
- The Aerospace Engineering Department for providing computational resources
- The open-source RL community for excellent libraries and tools
- Colleagues and fellow researchers for valuable discussions and feedback

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Resources

- **PyTorch**: https://pytorch.org/
- **Gymnasium**: https://gymnasium.farama.org/
- **ROS2**: https://docs.ros.org/en/humble/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Three-Body Problem**: https://en.wikipedia.org/wiki/Three-body_problem

---

<div align="center">

**⭐ If you find this research useful, please consider giving it a star! ⭐**

Made with ❤️ at Sharif University of Technology

</div>

For questions, issues, or collaboration inquiries, please open a GitHub issue or reach out to the author:

- GitHub: [@alibaniasad1999](https://github.com/alibaniasad1999)
- Email: alibaniasad1999@yahoo.com  

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Master Thesis

## Robust Reinforcement Learning Differential Game Guidance in Low-Thrust, Multi-Body Dynamical Environments

This repository hosts the complete set of materials for the master's thesis by Ali Bani Asad, exploring the integration of robust reinforcement learning and differential game theory for spacecraft guidance in low-thrust, multi-body dynamical environments.

---

## 📂 Repository Structure

- **`.idea/`**: IDE configuration files.
- **`Article/`**: LaTeX source for the full thesis document.
- **`Code/`**: Jupyter notebooks and supporting Python modules for simulations, training, and analysis.
- **`Figure/ TBP/`**: Figures and images referenced throughout the thesis (To Be Prepared).
- **`Presentation/`**: Slide deck for the thesis defense.
- **`Proposal/`**: Initial research proposal and timeline.
- **`Report/`**: Interim and final reports in PDF format.
- **`.gitignore`**: Files and directories ignored by Git.
- **`qodana.yaml`**: Configuration for static code analysis (Qodana).
- **`requirements.txt`**: Python package dependencies.

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** (for installing Python dependencies)
- **Jupyter Notebook or JupyterLab**
- **LaTeX distribution** (TeX Live or MiKTeX) for compiling the thesis document

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:alibaniasad1999/master-thesis.git
   cd master-thesis
   ```
2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Running Simulations & Experiments

All code for experiments is organized in the `Code/` directory as Jupyter notebooks. To launch:

```bash
jupyter lab  # or jupyter notebook
```

Open the notebooks prefixed with `01_`, `02_`, etc., to run simulations, train reinforcement learning agents, and visualize results.

### 2. Compiling the Thesis Document

Navigate to the `Article/` folder:

```bash
cd Article
latexmk -pdf thesis.tex
```

This generates `thesis.pdf`, the final thesis manuscript.

### 3. Presentation

Slides for the defense are located in `Presentation/`. You can view or export the PDF:

```bash
cd Presentation
# Open thesis_presentation.pdf with your preferred viewer
```

---

## 📊 Results & Figures

Processed results, tables, and plots are saved within the `Code/` notebooks and exported to the `Figure/ TBP/` folder for inclusion in the thesis.

---

## 📖 Citation

If you use this work, please cite:

> Ali Bani Asad. "Robust Reinforcement Learning Differential Game Guidance in Low-Thrust, Multi-Body Dynamical Environments." Master's Thesis, Sharif University of Technology, 2025.

---

## ✉️ Contact & Support

For questions, issues, or collaboration inquiries, please open a GitHub issue or reach out to the author:

- GitHub: [@alibaniasad1999](https://github.com/alibaniasad1999)
- Email: alibaniasad1999@yahoo.com  

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

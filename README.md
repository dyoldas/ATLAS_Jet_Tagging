# Jet Tagging Project

**Author:** Deniz Yoldas  
**Course:** Physics Applications of AI  
**Project Topic:** Jet Tagging

---

## Overview

This project applies machine learning techniques to the jet tagging problem in high-energy physics. The goal is to classify jets into physical categories by employing modern neural network architectures. Binary classification is performed between QCD jets and a combined class of top + W/Z jets. Multiclass classification is also supported. This modular codebase is inspired by (`projecttemplate-master`), allowing for flexible model development and reproducibility.

---

## Directory Structure

Structure of the jet_tagging_dyoldas project:
```
jet_tagging_dyoldas/
│   README.md
│   pyproject.toml
│   config_default.yaml
│   train.py
│   evaluate.py
│   data_analysis.ipynb
│
└── data/
│
└── jet_tagging/
│   ├── __init__.py
│   ├── data.py
│   └── networks/
│       ├── __init__.py
│       └── mlp.py
│
└── outputs/
│
└── writeup/
    └── jet_tagging_analysis_dyoldas_copy.pdf
    └── example_outputs/
```

---

## Getting Started

1. **Install dependencies:**

Some dependencies were downgraded/changed from `projecttemplate-master` while developing the project, below you will see some contents of `pyproject.toml`:
```
requires-python = ">=3.12"
dependencies = [
    "h5py>=3.13.0",
    "matplotlib>=3.10.0",
    "numpy<2",
    "pandas>=2.2.3",
    "dacite>=1.9.2",
    "pyyaml>=6.0.2",
    "scikit-learn>=1.6.1",
    "simple-parsing>=0.1.7",
    "tables>=3.10.2",
    "torch",
    "torchvision",
]
```
So please make sure you start by installing the dependencies to prevent any bugs. To install, run the following line in the terminal:
```bash
pip install -e .
```

2. **Check that following data files exist under jet_tagging_dyoldas/data/:**
    1. `qcd_const.h5`
    2. `top_const.h5`
    3. `wz_const.h5`
    4. `qcd_small.h5`
    5. `top_small.h5`
    6. `wz_small.h5`

    ***Note:** if any data file is missing, download and add the missing file to the jet_tagging_dyoldas/data/ directory from [here](https://drive.switch.ch/index.php/s/H7I57It4auYxJmb).*

3. **Configure training:**
    Edit `config_default.yaml` to adjust hyperparameters, model settings, etc (more details are given below).

4. **Train model:** To start training, run the following line in the terminal:
```bash
python train.py --run_name "replace_with_the_name_you_want"
```
Or, for debugging, where you don't want to edit the config and press enter all the time, you can use the optional --no_config_edit boolean flag like below:
```bash
python train.py --run_name "name" --no_config_edit True
```

5. **Evaluate model:** To evaluate given training, run the following line in the terminal:
```bash
python evaluate.py --run_name "replace_with_the_name_you_want"
```

---

## Description of Key Files and Folders

- **README.md:** Project overview, instructions, and documentation.
- **pyproject.toml:** Lists required packages/dependencies.
- **config_default.yaml:** Stores experiment hyperparameters and configuration.
- **train.py:** Main script for training models.
- **evaluate.py:** Script for evaluating trained models.
- **data/:** Place for all raw jet dataset files.
- **jet_tagging/:** Main Python package for this project.
    - **data.py:** Handles all data preparation.
    - **__init__.py:** Marks jet_tagging folder as importable module, also contains necessary training functionalities.
    - **networks/:** Contains neural network architecture code.
        - **mlp.py:** MLP (multi-layer perceptron) for jet classification, custom weigh.
        - **__init__.py:** Marks jet_tagging/networks folder as importable module, also containt functionality for network construction
- **data_analysis.ipynb:** Jupyter notebook to visualize data preparation.
***Note: This notebook is not a core module; it visualizes the preprocessing steps from data.py and helps interpret input features.***
- **outputs/:** Training and evaluation outputs will be saved here (more details are given below).
- **writeup/jet_tagging_analysis_dyoldas_copy.pdf:** PDF copy of the PRL style writeup for the project.
- **writeup/example_outputs/:** Some example output files that I generated.

---

## Training Configuration

If you open the `config_default.yaml` file, you will see the following structure:
```
training:
  dataset_type: ConstituentMultiWeightedJets   # Options: BinaryJets, BinaryWeightedJets, FeatureWeightedJets, MultiJets, MultiWeightedJets, ConstituentBinaryJets, ConstituentMultiJets, ConstituentMultiWeightedJets

  # === Paths to HDF5 files ===
  qcd_path: data/qcd_small.h5
  top_path: data/top_small.h5
  wz_path: data/wz_small.h5

  qcd_const_path: data/qcd_const.h5
  top_const_path: data/top_const.h5
  wz_const_path: data/wz_const.h5

  # === Dataset setup ===
  batch_size: 128
  frac_train: 0.75
  frac_val: 0.15
  seed: 42
  augment: True

  # === BinaryWeightedJets-specific ===
  four_vector_cut: False

  # === Model architecture ===
  architecture: "deep"   # Options: standard, deep, downsized
  hidden_dim: 128
  use_dropout: False
  use_batchnorm: True

  # === Optimization ===
  learning_rate: 0.002
  epoch_count: 10
  weight_regularization: True
  weight_decay: 0.05

  # === Evaluation threshold ===
  threshold: 0.5
```
Detailed explanation follows below:
- **dataset_type**
    - **BinaryJets:** Base dataset for binary classification task. Preprocesses and splits the data and returns appropriate DataLoader and Tensor objects for binary classification. QCD jets are labeled as class 0, and Top + WZ jets are labeled as class 1.
    - **BinaryWeightedJets:** For binary classification, applies class based weighing to do eliminate class imbalances. **Optional four_vector_cut argument to exclude ['pt', 'eta', 'phi', 'mass'] and focus on substructures.**
    - **FeatureWeightedJets:** For binary classification, applies per-sample feature-based weighing to eliminate kinematics bias and focus on substruces & shapes.
    - **MultiJets:** Base dataset for multiclassification task between QCD, Top, & WZ jets. Extended logic of binary classification.
    - **MultiWeightedJets:** For multi classification, applies class based weighing, **also optional four_vector_cut argument.**
    - **ConstituentBinaryJets:** Base dataset for binary classification task using jet constituents. Employs DeepSet module automatically.
    - **ConstituentBinaryWeightedJets:** For binary classification using constituents, applies class based weighing.
    - **ConstituentMultiJets:** Base dataset for multi classification using jet constituents.
    - **ConstituentMultiWeightedJets:** For multi classification, applies class based weighing.

- **Dataset Setup**
    - **batch_size:** Adjust batch size for mini batching.
    - **frac_train/frac_val:** Adjust training, validation, test split.
    - **seed:** Very important for reproductibility of dataset shuffles in train.py & evaluate.py files.
    - **four_vector_cut:** Cuts the four-vector features to focus on substructures (not applicable for constituents)
    - **augment** Augments constituent data by adding extra energy feature and centering phi & eta.

- **Model Architecture**
    - **architecture:** Different MLP architecture styles, look into mlp.py file for more detail, **also reflected on DeepSet submodules.**
    - **hidden_dim:** Dimension of hidden layers.
    - **use_dropout:** Use dropout method in the modules.
    - **use_batchnorm:** Use batch normalization method in the modules.

- **Optimization**
    - **learning_rate:** Learning step size for optimizer.
    - **epoch_count:** Number of training iterations.
    - **weight_regularization:** Apply weight regularization technique.
    - **weight_decay:** Weight decay rate for AdamW optimizer.
    - **threshold:** Not used/important, threshold for binary classification accuracy.

Adjust the parameters above in `config_default.yaml` to set different training configurations.

---

## Outputs

In the `writeup/example_outputs/` directory, I provided example training & evaluation results that I generated from my tests which I used in writing the analysis paper.

`outputs/your_test_run_name/` directory will be created if you run `train.py`. You will see the following within that directory:
- **classifier/:**
    - **parameters.pth:** Saved best model during training.
    - **hyperparameters.yaml:** Saved model metadata.
- **config.yaml:** Configurations used for that training.
- **plots/:**
    - **training_summary.png:** Basic training summary plot.

If you run `evaluate.py` for a given test run, you will get more plots under `plots/` and a log file summarizing evaluation.

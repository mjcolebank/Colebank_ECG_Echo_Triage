# Colebank_ECG_Echo_Triage
Code for reproducing results in "Uncertainty-aware classification and triage of structural heart disease using electrocardiography and echocardiography metrics"

# Uncertainty-aware classification and triage of structural heart disease using electrocardiography and echocardiography metrics

This repository contains code associated with the paper:

**Uncertainty-aware classification and triage of structural heart disease using electrocardiography and echocardiography metrics**

The code in this repository can be used to train neural-network models, perform uncertainty-aware classification, evaluate model performance, and generate plots for analysis and visualization. The repository is intended to support reproducibility of the modeling and plotting workflow described in the paper.

## Data availability

The data are **not included** in this GitHub repository.

To run the training and evaluation scripts, users must separately download the EchoNext dataset from PhysioNet:

**EchoNext: A Dataset for Detecting Echocardiogram-Confirmed Structural Heart Disease from ECGs**  
PhysioNet, version 1.1.1  
https://physionet.org/content/echonext/

Please follow the PhysioNet data-use requirements and cite the EchoNext dataset appropriately when using these data.

The EchoNext PhysioNet resource contains de-identified ECG data paired with echocardiogram-confirmed structural heart disease labels, along with demographic and ECG-specific metadata. The code in this repository assumes that the required data files have been downloaded separately and placed in the expected local data directory.

## Repository contents

This repository provides code for:

- Training deterministic neural-network classifiers using PyTorch
- Training Bayesian or uncertainty-aware neural-network models using NumPyro/JAX
- Performing classification of structural heart disease labels
- Computing performance metrics including ROC curves, AUC, confusion matrices, and classification reports
- Generating plots and figures used for model evaluation and interpretation

## Installation

We recommend using a clean Python environment.

```bash
conda create -n shd-triage python=3.10
conda activate shd-triage

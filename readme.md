# Endogenous Augmentation for NER

Welcome to the **Endogenous Augmentation** project! This repository contains code for training and testing a Named Entity Recognition (NER) model with endogenous augmentation techniques. The project includes scripts for training (`train.py`) and testing (`test.py`), with a pre-trained model saved in the `Models` directory.

---

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Model Storage](#model-storage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project implements a training and testing pipeline for an NER model enhanced with endogenous augmentation. The training script (`train.py`) allows customization of hyperparameters like sample size, regularization strength, and training epochs, while the testing script (`test.py`) provides an option for quick inference on a smaller sample.

---

## Dependencies
To run the code, you'll need the following Python packages:
- `torch` - PyTorch for deep learning
- `datasets` - Hugging Face Datasets library
- `scikit-learn` - Machine learning utilities
- `transformers` - Hugging Face Transformers library

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:
    ```bash
    pip install torch datasets scikit-learn transformers
    ```
## Usage

### Training
The `train.py` script trains the NER model with endogenous augmentation. You can customize the training process using command-line arguments.

#### Command
```bash
python train.py --sample_size 100 --alpha 0.01 --max_epochs 500 --patience 8```
Arguments
Argument	Type	Default	Description
--sample_size	int	100	Number of samples to use for training
--alpha	float	0.01	Alpha value for regularization
--max_epochs	int	500	Maximum number of epochs to train
--patience	int	8	Patience for early stopping

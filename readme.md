# Exogenous and Endogenous Augmentation for NER
### Exogenous Augmentation -> /Exogenous

# Endogenous Augmentation

Welcome to the *Endogenous Augmentation* project! This repository contains code for training and testing a Named Entity Recognition (NER) model with endogenous augmentation techniques. The project includes scripts for training (train.py) and testing (test.py), with a pre-trained model saved in the Models directory.

---

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)

---

## Overview
This project implements a training and testing pipeline for an NER model enhanced with endogenous augmentation. The training script (train.py) allows customization of hyperparameters like sample size, regularization strength, and training epochs, while the testing script (test.py) provides an option for quick inference on a smaller sample.

---

## Overall Architecture
![image](https://github.com/user-attachments/assets/d09047ee-089d-4633-8ab4-844d972c17af)

# Architectural Flow

## 1. Exogenous Data Augmentation (Text-Level Diversity)

### LLM-Driven Generation
Leverages large language models (e.g., ChatGPT) to synthesize new training samples through:

- **Instruction Constraints**: Explicit prompts requiring LLMs to generate texts with entities that are highly dissimilar to original low-resource data (e.g., varying syntactic structures or semantic contexts).
- **Self-Refinement**: Automatically verifies generated samples via LLM-based rechecking to filter out context-entity mismatches or labeling errors.
- **In-Context Learning**: (Novelty)

**Outcome**: Expands the original "anchor" dataset into diverse, high-quality synthetic data (D).

---

## 2. Endogenous Data Augmentation (Semantic Space Exploration)

### Intra-Class Semantic Variations
Operates in the feature space by:

#### **Directional Feature Perturbation**
Identifies latent semantic directions via covariance matrices derived from entity mentions and contexts. For a token *i* in class *c*:

```math
\text{Augmented Feature} = e_i^r + \beta \cdot N(0, \Sigma_e^c) + \gamma \cdot N(0, \Sigma_t^c)
```

Where:
- \( \Sigma_e^c \) and \( \Sigma_t^c \) are covariance matrices for entity and context features.
- \( \beta, \gamma \) control augmentation strength.

#### **Moment Generating Function (MGF)**
Strengthens augmentation upper bounds by combining entity-context perturbations multiplicatively.

**Outcome**: Enables infinite semantic variations without explicit text generation, mitigating noise.

---

## 3. Training Pipeline

### **Phase 1:** Train on exogenous corpus \( D \) to learn robust initial representations.

### **Phase 2:** Fine-tune on original low-resource data using the combined loss:

```math
L = L_{base} + \alpha (L_{ENDA-sp} + L_{ENDA-tp})
```

Where:
- \( L_{base} \) is the standard NER loss.
- \( L_{ENDA} \) terms enforce semantic consistency from endogenous augmentations.



---

## Dependencies
To run the code, you'll need the following Python packages:
- torch - PyTorch for deep learning
- datasets - Hugging Face Datasets library
- scikit-learn - Machine learning utilities
- transformers - Hugging Face Transformers library

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/JayeshK008/ENEX.git
   cd your-repo-name
   ```
   

2. Install the required dependencies:
    ```bash
    pip install torch datasets scikit-learn transformers
    ```
    
## Usage

### Training
The train.py script trains the NER model with endogenous augmentation. You can customize the training process using command-line arguments.

#### Command
bash
python train.py --sample_size 100 --alpha 0.01 --max_epochs 500 --patience 8

## Arguments

| Argument       | Type  | Default | Description                              |
|---------------|------|---------|------------------------------------------|
| --sample_size | int  | 100     | Number of samples to use for training  |
| --alpha      | float | 0.01    | Alpha value for regularization         |
| --max_epochs | int  | 500     | Maximum number of epochs to train      |
| --patience   | int  | 8       | Patience for early stopping            |

### Testing
The test.py script tests the train model against the testing data.
for this you will need to download the model from the below link and place it in Models folder.
[Model Link](https://drive.google.com/file/d/1vRmGc-VwND0Fce-zNW8PiAYeZKWMNmrJ/view)

#### Command
bash
python test.py 


### Note
We have explicitly built exogenous and endogenous methods and obtained the results.For Endogenous Results in low resource setting you can visit the Endogenous_exp file. In the final code for phase 3, we will optimize for higher accuracy and F1 score while working on a required subset of data in a low-resource setting and tuning hyperparameters. The workflow is as follows: first, we generate data augmentation using the exogenous method, then use both the original and augmented data for the subsequent endogenous processing.

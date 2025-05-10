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
  - [General](#general)
  - [Training](#training)
  - [Testing](#testing)
- [Results](#results)

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
   cd ENEX
   ```
   

2. Install the required dependencies:
    ```bash
    pip install torch datasets scikit-learn transformers
    ```
    
## Usage

### General
We evaluate our method on the [MultiCoNER 2022 dataset](https://multiconer.github.io/) across 10 languages, using only 100 training samples per language.

## 🌍 Languages Supported
- English (`En`)
- Bengali (`Bn`)
- Hindi (`Hi`)
- German (`De`)
- Spanish (`Es`)
- Korean (`Ko`)
- Dutch (`Nl`)
- Russian (`Ru`)
- Turkish (`Tr`)
- Chinese (`Zh`)

---

### Directory Structure

```

├── Augmented/               # Contains augmented data (exogenous + endogenous) for each language
├── Model/                  # The user should create this Directory for saving the models
├── Multiconer2022/         # Contains raw MultiCoNER 2022 dataset
├── exogenous/              # Code for generating exogenous augmentations
├── train.py                # Training script
├── test.py                 # Evaluation script

```

---

### Training
To train the NER model with endogenous and optionally exogenous augmentation:

python train.py --lang En --sample_size 100 --woExo False --alpha 0.01 --max_epochs 500 --patience 8
Arguments:
--lang: Language code (default: En)
--sample_size: Number of training samples (default: 100)
--woExo: Whether to disable exogenous augmentation (default: True)
--alpha: Regularization coefficient for orthogonality loss
--max_epochs: Maximum training epochs (default: 500)
--patience: Early stopping patience (default: 8)

### Testing
Evaluate a trained model using:

python test.py --modelPath path/to/model.pth --lang En
Arguments:
--modelPath: Path to the saved model file
--lang: Language code (default: En)


### Results

**Table 1: NER performance (micro-F1, %) on 10 languages under low-resource (100-sample) setting**

| Method    | En        | Bn        | Hi        | De        | Es        | Ko        | Nl        | Ru        | Tr        | Zh        | Avg       |
| --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Gold-only | 29.36     | 14.49     | 18.80     | 37.04     | 36.30     | 12.76     | 38.78     | 23.89     | 24.13     | 14.18     | 24.97     |
| LwTR      | 48.60     | 20.25     | 29.95     | 48.38     | 44.08     | 35.09     | 43.00     | 39.22     | 30.58     | 27.70     | 36.68     |
| DAGA      | 16.24     | 5.87      | 10.40     | 32.44     | 27.78     | 19.28     | 15.44     | 11.14     | 16.17     | 10.33     | 16.51     |
| MELM      | 40.12     | 6.22      | 27.84     | 43.94     | 37.45     | 34.10     | 37.82     | 32.38     | 20.13     | 25.11     | 30.51     |
| ACLM      | 48.76     | 23.09     | 33.53     | 48.80     | 44.14     | 38.35     | 46.22     | 39.48     | 37.20     | 35.12     | 39.47     |
| E2DA      | 56.69     | 35.07     | 44.32     | 56.02     | 52.83     | 47.77     | 53.24     | 44.36     | 40.57     | 42.26     | 47.31     |
| **Ours**  | **66.67** | **41.42** | **57.88** | **61.06** | **61.26** | **51.53** | **62.20** | **62.03** | **50.52** | **50.03** | **56.46** |

Our approach shows significant improvements across all languages and sets a new state-of-the-art in low-resource complex NER.

---

# Exogenous Augmentation for NER

**Exogenous Augmentation** project! This repository contains code for training and testing a Named Entity Recognition (NER) model with exogenous augmentation techniques. The project includes scripts for training (`train.py`) and testing (`test.py`).

---

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Exogenous Data](#exogenous-data)
  - [Training](#training)
  - [Testing](#testing)
- [Model Storage](#model-storage)


---

## Overview
This project implements a training and testing pipeline for an NER model using XLM-RoBERTa-large, enhanced with exogenous augmentation. The training script (train.py) allows customization of hyperparameters, including sample size (500 selected original samples + augmented samples), learning rate, batch size, and training epochs. The testing script (test.py) evaluates the trained model on 5% of the test dataset, providing an F1 score and classification report for performance analysis.

---

## Dependencies
To run the code, you'll need the following Python packages:
- `torch` - PyTorch for deep learning
- `json` - Handling Json Files
- `datasets` - Hugging Face Datasets library
- `scikit-learn` - Machine learning utilities
- `transformers` - Hugging Face Transformers library
- `seqeval` - Sequence evaluation metrics for named entity recognition (NER)
- `nltk` - Natural Language Toolkit for text preprocessing and tokenization

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:
    ```bash
    pip install torch datasets scikit-learn transformers seqeval nltk
    ```
## Usage

### Exogenous Data
Data Augmentation
The exogenous_data.py script performs data augmentation on the training set.

#### Command
```bash
python exogenous_data.py --num_augmentations 3
```
### Arguments
| Argument       | Type  | Default | Description                              |
|---------------|------|---------|------------------------------------------|
| `--num_augmentations` | int  | 3     | Number of augmented samples to generate per original sample  |

### Training
The `train.py` script trains the **NER model** with **exogenous augmentation**.  
You can customize the training process using **command-line arguments**.  

#### Command
```bash
python train.py --sample_size 500 --learning_rate 5e-6 --batch_size 4 --epochs 5
```
### Arguments

| Argument       | Type  | Default | Description                              |
|---------------|------|---------|------------------------------------------|
| `--sample_size` | int  | 500     | Number of original samples used for training  |
| `--learning rate`      | float | 5e-6    | Learning Rate for Finetuning         |
| `--batch_size` | int  | 4     | Batch size for training and evaluation      |
| `--epochs`   | int  | 10       | Training epochs            |


### Testing
The `test.py` script evaluates the trained **NER model** on 5% of the test data and computes the **F1 score**.

#### Command
```bash
python test.py --test_sample_size 5
```
### Arguments

| Argument       | Type  | Default | Description                              |
|---------------|------|---------|------------------------------------------|
| `--test_sample_size` | int  | 5     | % of test samples used to evaluate  |


We have introduced In-Context Learning (ICL) data augmentation alongside the original data augmentation technique prescribed in the paper: Exogenous and Endogenous Data Augmentation for Low-Resource Complex Named Entity Recognition (specifically in the Exogenous part).

In a low-resource setting, it is crucial to obtain diverse yet meaningful data that varies both semantically and syntactically. Large Language Models (LLMs) can be leveraged to generate high-quality augmented data that preserves coherency while introducing diversity.

LLMs generate augmented examples by restructuring content, ensuring variation while maintaining entity integrity. Additionally, In-Context Learning (ICL) enhances augmentation by learning patterns from provided examples, allowing the model to generate contextually appropriate and structurally diverse samples.

By incorporating these augmented examples, we effectively expand the dataset, mitigating the limitations of low-resource settings. This approach improves generalization, allowing the model to better recognize complex named entities with limited original training data.


## Model
Since we introduced In-Context Learning (ICL) augmentation with a limited number of samples and did not extensively tune hyperparameters, the current implementation serves as a baseline experiment rather than a fully optimized model.

To maintain flexibility and avoid unnecessary storage usage, we did not save the trained model. Instead, the model can be retrained and evaluated by running the provided scripts. This allows further experimentation with different hyperparameters, additional data augmentation techniques, and fine-tuning strategies to enhance performance.

### Note:

For output and results, refer to the /notebooks directory. This folder contains Python notebooks where you can analyze the model's performance, visualization of predictions, and evaluation metrics. These notebooks provide insights into training loss, validation loss, F1 scores, and entity recognition performance across different test samples.


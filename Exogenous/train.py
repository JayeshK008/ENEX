import json
import random
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load JSON files
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Load datasets
train_data = load_json("train_preprocessed.json")
val_data = load_json("validation_preprocessed.json")
test_data = load_json("test_preprocessed.json")
augmented_data = load_json("augmented_data_exogenous.json")

# Extract sentences and entity labels
def extract_data(data):
    sentences, labels = [], []
    for entry in data:
        tokens = entry.get("tokens", [])
        entities = entry.get("ner_tags", [])
        if tokens and entities and len(tokens) == len(entities):
            sentences.append(tokens)
            labels.append(entities)
    return sentences, labels

# Process datasets
train_sentences, train_labels = extract_data(train_data)
val_sentences, val_labels = extract_data(val_data)
test_sentences, test_labels = extract_data(test_data)

# Randomly select 500 samples from train_preprocessed.json
selected_indices = random.sample(range(len(train_sentences)), 500)
train_sentences = [train_sentences[i] for i in selected_indices]
train_labels = [train_labels[i] for i in selected_indices]

# Convert original train data into tuples
original_train_data = list(zip(train_sentences, train_labels))

# Clean augmented data
def clean_augmented_data(data):
    cleaned_data = []
    for entry in data:
        tokens = entry.get("tokens", [])
        ner_tags = entry.get("ner_tags", [])
        if not tokens:
            continue
        if len(ner_tags) > len(tokens):
            ner_tags = ner_tags[:len(tokens)]
        elif len(ner_tags) < len(tokens):
            ner_tags.extend(["O"] * (len(tokens) - len(ner_tags)))
        cleaned_data.append((tokens, ner_tags))
    return cleaned_data

cleaned_augmented_data = clean_augmented_data(augmented_data)

# Select first 30% of test data
correct_test_size = int(0.3 * len(test_data))
test_sentences = test_sentences[:correct_test_size]
test_labels = test_labels[:correct_test_size]

# Extract all unique labels
all_labels = set(label for labels in train_labels for label in labels) | \
             set(label for labels in val_labels for label in labels) | \
             set(label for labels in test_labels for label in labels)

# Create label mappings
label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
id2label = {idx: label for label, idx in label2id.items()}

# Merge original and augmented data
final_train_data = original_train_data + cleaned_augmented_data

# Convert labels from text to corresponding integers
final_train_data = [(tokens, [label2id[label] for label in labels]) for tokens, labels in final_train_data]
val_data_pairs = [(tokens, [label2id[label] for label in labels]) for tokens, labels in zip(val_sentences, val_labels)]
test_data_pairs = [(tokens, [label2id[label] for label in labels]) for tokens, labels in zip(test_sentences, test_labels)]

# Load tokenizer
XLM_MODEL_NAME = "xlm-roberta-large"
xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_NAME)

# Function to tokenize data & align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = xlm_tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id >= len(label):
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Convert data to Hugging Face dataset format
def convert_to_dataset(data):
    return Dataset.from_dict({"tokens": [x[0] for x in data], "ner_tags": [x[1] for x in data]})

# Convert and tokenize datasets
train_dataset = convert_to_dataset(final_train_data).map(tokenize_and_align_labels, batched=True)
val_dataset = convert_to_dataset(val_data_pairs).map(tokenize_and_align_labels, batched=True)
test_dataset = convert_to_dataset(test_data_pairs).map(tokenize_and_align_labels, batched=True)

# Load model with correct number of labels
model = AutoModelForTokenClassification.from_pretrained(
    XLM_MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=xlm_tokenizer,
)

# Start training
trainer.train()

# Save the model
model_save_path = "./ner_model_final"
trainer.save_model(model_save_path)
print(f"Model saved to {model_save_path}")

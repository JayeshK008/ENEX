import json
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import classification_report

# Load JSON files
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Paths to dataset files
test_file = ("test_preprocessed.json")

# Load test dataset
test_data = load_json(test_file)

# Extract sentences and entity labels
def extract_data(data):
    sentences, labels = [], []
    for entry in data:
        tokens = entry.get("tokens", [])
        entities = entry.get("ner_tags", [])
        if tokens and entities:
            sentences.append(tokens)
            labels.append(entities)
    return sentences, labels

# Process test dataset
test_sentences, test_labels = extract_data(test_data)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Select 30% of test data
correct_test_size = int(0.3 * len(test_data))
test_sentences = test_sentences[:correct_test_size]
test_labels = test_labels[:correct_test_size]

# Load the saved model and tokenizer
model_path = "./ner_model_final"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to the appropriate device
model.to(device)

# Get label mappings from the model
label2id = model.config.label2id
id2label = model.config.id2label

# Function to tokenize and align labels
def tokenize_and_align_labels(tokens, labels):
    tokenized_inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=128)
    word_ids = tokenized_inputs.word_ids()
    
    label_ids = []
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        else:
            label_ids.append(label2id[labels[word_id]])
    
    return tokenized_inputs, torch.tensor(label_ids)

# Function to get predictions
def get_predictions(model, tokens, labels):
    model.eval()
    with torch.no_grad():
        inputs, label_ids = tokenize_and_align_labels(tokens, labels)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    return predictions.cpu().numpy(), label_ids.numpy()

# Evaluate on test set
all_true_labels = []
all_pred_labels = []

for tokens, labels in zip(test_sentences, test_labels):
    predictions, true_labels = get_predictions(model, tokens, labels)
    
    # Filter out padding (-100)
    valid_indices = true_labels != -100
    filtered_preds = predictions[valid_indices]
    filtered_true = true_labels[valid_indices]
    
    all_pred_labels.extend([id2label[pred] for pred in filtered_preds])
    all_true_labels.extend([id2label[true] for true in filtered_true])

# Compute and print classification report
report = classification_report(all_true_labels, all_pred_labels, digits=4)
print("\n🔹 Test Set Performance (Classification Report):\n")
print(report)

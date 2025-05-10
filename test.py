import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Subset
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
import random
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for NER model with endogenous augmentation")
    parser.add_argument('--modelPath', type=str, default="NA", help='Path where the model is stored')
    parser.add_argument('--lang',type=str,default="En",help='Can Choose from languages En, Bn, Hi, De, Es, Ko, Nl, Ru, Tr, Zh')
    return parser.parse_args()

args = parse_args()

# Use parsed arguments
modelPath = args.modelPath
lang = args.lang


#labels set
entity_label_set = ['O','CORP', 'CW', 'GRP', 'LOC', 'PER', 'PROD']

#loading data
def read_conll_file(file_path):
    sentences = []
    ner_tags = []
    tokens = []
    tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    sentences.append(tokens)
                    ner_tags.append(tags)
                    tokens = []
                    tags = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 4:
                token, _, _, tag = parts
                tokens.append(token)
                tags.append(tag)

    if tokens:
        sentences.append(tokens)
        ner_tags.append(tags)

    return sentences, ner_tags

def load_conll_dataset_from_dir(data_dir):
    all_sentences = []
    all_tags = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".conll"):
            path = os.path.join(data_dir, filename)
            sents, tags = read_conll_file(path)
            all_sentences.extend(sents)
            all_tags.extend(tags)

    return all_sentences, all_tags

def create_hf_dataset(sentences, tags):
    data = [{"tokens": s, "ner_tags": t} for s, t in zip(sentences, tags)]
    return Dataset.from_list(data)


test_data_dir= f"multiconer2022/{lang}/test"

#test
sentences1, ner_tags1 = load_conll_dataset_from_dir(test_data_dir)
test_dataset = create_hf_dataset(sentences1, ner_tags1)

tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')
encoder = XLMRobertaModel.from_pretrained('xlm-roberta-large')

span_label_set = ['B', 'I', 'O']
span2id = {label: idx for idx, label in enumerate(span_label_set)}
entity2id = {label: idx for idx, label in enumerate(entity_label_set)}

def split_ner_tags(ner_tags):
    span_labels = []
    entity_labels = []
    for tag in ner_tags:
        if tag == "O":
            span_labels.append("O")
            entity_labels.append("O")
        else:
            bio, entity = tag.split("-", 1)
            span_labels.append(bio)
            entity_labels.append(entity)
    return span_labels, entity_labels

def collate_fn(batch):
    batch_tokens = [item['tokens'] for item in batch]
    batch_ner_tags = [item['ner_tags'] for item in batch]

    batch_span_labels = []
    batch_entity_labels = []
    for ner_tags in batch_ner_tags:
        span_labels, entity_labels = split_ner_tags(ner_tags)
        batch_span_labels.append([span2id[label] for label in span_labels])
        batch_entity_labels.append([entity2id[label] for label in entity_labels])

    encodings = tokenizer(batch_tokens, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    max_len = input_ids.size(1)
    padded_span_labels = []
    padded_entity_labels = []
    for idx, (span_labels, entity_labels) in enumerate(zip(batch_span_labels, batch_entity_labels)):
        word_ids = encodings.word_ids(batch_index=idx)
        aligned_span = []
        aligned_entity = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_span.append(-100)
                aligned_entity.append(-100)
            elif word_id != prev_word_id:
                aligned_span.append(span_labels[word_id])
                aligned_entity.append(entity_labels[word_id])
            else:
                aligned_span.append(-100)
                aligned_entity.append(-100)
            prev_word_id = word_id
        padded_span_labels.append(aligned_span)
        padded_entity_labels.append(aligned_entity)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'span_labels': torch.tensor(padded_span_labels),
        'entity_labels': torch.tensor(padded_entity_labels)
    }

# DataLoaders
batch_size = 8
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model Definition
class E2DAEndogenous(nn.Module):
    def __init__(self, hidden_dim=1024, num_span_labels=len(span_label_set), num_entity_labels=len(entity_label_set)):
        super(E2DAEndogenous, self).__init__()
        self.encoder = encoder
        self.shared_extractor = nn.Linear(hidden_dim, hidden_dim)
        self.span_extractor = nn.Linear(hidden_dim, hidden_dim)
        self.type_extractor = nn.Linear(hidden_dim, hidden_dim)
        self.span_head = nn.Linear(hidden_dim, num_span_labels)
        self.type_head = nn.Linear(hidden_dim, num_entity_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state

        H_share = self.shared_extractor(H)
        H_st = H - H_share
        H_span = self.span_extractor(H_st)
        H_type = self.type_extractor(H_st)
        H_sp = H_span + H_share
        H_tp = H_type + H_share

        span_logits = self.span_head(self.dropout(H_sp))
        type_logits = self.type_head(self.dropout(H_tp))
        return span_logits, type_logits, H_span, H_type, H_share

# Endogenous Augmentation Loss
def compute_covariance_matrix(features, labels, num_classes):
    batch_size, seq_len, dim = features.size()
    device = features.device  # Get the device of the input features
    features_flat = features.view(-1, dim)
    labels_flat = labels.view(-1)
    cov_matrices = []
    for c in range(num_classes):
        class_features = features_flat[labels_flat == c]
        if class_features.numel() > 0 and class_features.size(0) > 1:
            cov = torch.cov(class_features.T)
            cov_matrices.append(cov + torch.eye(dim, device=device) * 1e-6)
        else:
            cov_matrices.append(torch.eye(dim, device=device) * 1e-6)
    return torch.stack(cov_matrices)

def endogenous_loss(logits, labels, features, head_weights, head_bias, lambda_):
    batch_size, seq_len, num_classes = logits.size()
    cov_matrices = compute_covariance_matrix(features, labels, num_classes)

    loss = 0
    valid_tokens = 0
    for i in range(batch_size):
        for j in range(seq_len):
            if labels[i, j] != -100:
                c_i = labels[i, j]
                h_i = features[i, j]
                log_sum_exp = 0
                for c_j in range(num_classes):
                    if c_j != c_i:
                        delta_w = head_weights[c_j] - head_weights[c_i]
                        delta_b = head_bias[c_j] - head_bias[c_i]
                        mean_term = delta_w @ h_i + delta_b
                        var_term = (lambda_ / 2) * delta_w @ cov_matrices[c_i] @ delta_w
                        log_sum_exp += torch.exp(mean_term + var_term)
                loss += torch.log(1 + log_sum_exp)
                valid_tokens += 1
    return loss / valid_tokens if valid_tokens > 0 else torch.tensor(0.0, device=logits.device)

# Orthogonality Loss
def orthogonality_loss(H_span, H_type, H_share):
    dot1 = (H_span * H_share).sum(dim=-1).pow(2).mean()
    dot2 = (H_type * H_share).sum(dim=-1).pow(2).mean()
    dot3 = (H_span * H_type).sum(dim=-1).pow(2).mean()
    return dot1 + dot2 + dot3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = E2DAEndogenous().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
model.load_state_dict(torch.load(modelPath))
model.to(device)
model.eval()
test_preds, test_labels = [],[]

test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)

with torch.no_grad():
    for batch in test_loader_tqdm:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        span_labels = batch['span_labels'].to(device)
        entity_labels = batch['entity_labels'].to(device)

        span_logits, type_logits, _, _, _ = model(input_ids, attention_mask)
        span_preds = torch.argmax(span_logits, dim=-1)
        type_preds = torch.argmax(type_logits, dim=-1)

        for i in range(span_preds.size(0)):
            for j in range(span_preds.size(1)):
                if span_labels[i, j] != -100:
                    pred_label = f"{span_label_set[span_preds[i, j]]}-{entity_label_set[type_preds[i, j]]}" if span_preds[i, j] != 2 else "O"
                    true_label = f"{span_label_set[span_labels[i, j]]}-{entity_label_set[entity_labels[i, j]]}" if span_labels[i, j] != 2 else "O"

                    # Only append non-"O" labels to the lists
                    if pred_label != "O" and true_label != "O":
                        test_preds.append(pred_label)
                        test_labels.append(true_label)

# Compute Micro-F1 Score for non-"O" predictions
micro_f1 = f1_score(test_labels, test_preds, average='micro')
print(f"Test Micro-F1 (non-'O'): {micro_f1:.4f}")


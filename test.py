print("start")
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
import random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
print("import completed")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for  endogenous augmentation")
    parser.add_argument('--sample_test', type=bool, default=False, help='Testing on 20 percent sample for fast inference')
    return parser.parse_args()

args = parse_args()

# Use parsed arguments
sample_test = args.sample_test



torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

dataset = load_dataset('MultiCoNER/multiconer_v2', 'English (EN)')
test_dataset = dataset['test']

tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')
encoder = XLMRobertaModel.from_pretrained('xlm-roberta-large')
entity_label_set = ['O','Disease', 'SportsManager', 'Software', 'WrittenWork', 'Food', 'Scientist', 'OtherLOC', 'Cleric', 'Medication/Vaccine', 'PublicCorp', 'VisualWork', 'OtherPER', 'Artist', 'Symptom', 'SportsGRP', 'MedicalProcedure', 'Athlete', 'PrivateCorp', 'ORG', 'Politician', 'ArtWork', 'Drink', 'Vehicle', 'MusicalGRP', 'AnatomicalStructure', 'HumanSettlement', 'CarManufacturer', 'Facility', 'AerospaceManufacturer', 'OtherPROD', 'Clothing', 'MusicalWork', 'Station']

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
print("loader ready")
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = E2DAEndogenous().to(device)
model.load_state_dict(torch.load("Model/model_endo.pth"))
model.to(device)
model.eval()
from torch.utils.data import random_split

train_size = int(0.80 * len(test_dataset))
val_size = len(test_dataset) - train_size

train_test_dataset, val_test_dataset = random_split(test_dataset, [train_size, val_size])

train_test_loader = DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_test_loader = DataLoader(val_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model.eval()
test_preds, test_labels = [],[]
if sample_test==True:
    test_loader_tqdm = tqdm(val_test_loader, desc="Testing", leave=False)

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

                        if pred_label != "O" and true_label != "O":
                            test_preds.append(pred_label)
                            test_labels.append(true_label)

    micro_f1 = f1_score(test_labels, test_preds, average='micro')
    print(f"Test Micro-F1 (sample): {micro_f1:.4f}")
else:
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

                        if pred_label != "O" and true_label != "O":
                            test_preds.append(pred_label)
                            test_labels.append(true_label)

    micro_f1 = f1_score(test_labels, test_preds, average='micro')
    print(f"Test Micro-F1 (sample): {micro_f1:.4f}")

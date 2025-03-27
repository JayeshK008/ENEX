import json
import random
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import word_tokenize

import os
import json
from datasets import load_dataset

# Load the MultiCoNER v2 dataset for English
ds = load_dataset("MultiCoNER/multiconer_v2", "English (EN)")

# Define the save directory
save_dir = 'project_directory'
os.makedirs(save_dir, exist_ok=True)

# Preprocess the dataset
preprocessed_data = {}
for split in ['train', 'validation', 'test']:
    preprocessed_data[split] = []
    for example in ds[split]:
        # Extract tokens and ner_tags
        preprocessed_data[split].append({
            'tokens': example['tokens'],
            'ner_tags': example['ner_tags']
        })

# Save preprocessed data locally as JSON files
for split, data in preprocessed_data.items():
    file_path = os.path.join(save_dir, f'{split}_preprocessed.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved {split} data to {file_path}")



# Load JSON files
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

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

# Paths to dataset files
train_file = ("train_preprocessed.json")
val_file = ("validation_preprocessed.json")
test_file = ("test_preprocessed.json")

# Load and process datasets
train_data = load_json(train_file)
train_sentences, train_labels = extract_data(train_data)

# Randomly select 500 samples for training
random.seed(42)
train_samples = random.sample(list(zip(train_sentences, train_labels)), 500)

# Set up device and models
device = "cuda" if torch.cuda.is_available() else "cpu"
XLM_MODEL_NAME = "xlm-roberta-large"
GPTJ_MODEL_NAME = "EleutherAI/gpt-j-6B"

xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_NAME)
xlm_model = AutoModel.from_pretrained(XLM_MODEL_NAME).to(device).half()

gptj_tokenizer = AutoTokenizer.from_pretrained(GPTJ_MODEL_NAME)
gptj_model = AutoModelForCausalLM.from_pretrained(
    GPTJ_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

def get_xlm_embeddings(sentence):
    inputs = xlm_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = xlm_model(**inputs)
    return outputs.last_hidden_state

def get_few_shot_examples(data, num_examples=3):
    sampled_data = random.sample(data, min(num_examples, len(data)))
    return "\n".join([f"Sentence: {' '.join(entry[0])}\nEntities: {entry[1]}" for entry in sampled_data])

def generate_augmented_data(sentence, labels, few_shot_data, num_augmentations=3, use_incontext=False):
    if use_incontext:
        prompt = f"""You are an AI trained to generate diverse Named Entity Recognition (NER) data.
Your task is to rewrite the given sentence while preserving entity structure but changing context.
Ensure that the new sentences are significantly different from the original in wording.

Here are examples:
{few_shot_data}

Now, generate {num_augmentations} variations for this:
Sentence: {" ".join(sentence)}
Entities: {labels}

Output each sentence on a new line, without additional comments.
"""
    else:
        prompt = f"""Generate {num_augmentations} variations of the following sentence while maintaining named entity structure.

Original Sentence: {" ".join(sentence)}
Entity Tags: {labels}

Output each sentence on a new line, without additional comments.
"""

    input_ids = gptj_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = gptj_model.generate(
            input_ids,
            max_new_tokens=64,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    generated_text = gptj_tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return generated_text.strip().split("\n")

def is_diverse(original, generated):
    original_tokens = set(word_tokenize(" ".join(original)))
    generated_tokens = set(word_tokenize(generated))
    overlap = len(original_tokens & generated_tokens) / len(original_tokens)
    return overlap < 0.5

# Perform augmentation
augmented_data = []
num_augmentations = 3
num_samples_for_llm = 10
llm_samples = random.sample(train_samples, num_samples_for_llm)

for sentence, labels in llm_samples:
    few_shot_examples = get_few_shot_examples(train_samples)
    augmented_sentences_1 = generate_augmented_data(sentence, labels, few_shot_examples, num_augmentations, use_incontext=False)
    augmented_sentences_2 = generate_augmented_data(sentence, labels, few_shot_examples, num_augmentations, use_incontext=True)

    for aug_sentence in augmented_sentences_1 + augmented_sentences_2:
        augmented_data.append({"tokens": aug_sentence.split(), "ner_tags": labels})

# Save augmented data
with open("augmented_data_exogenous.json", "w", encoding="utf-8") as f:
    json.dump(augmented_data, f, indent=4)

print(f"Augmentation complete! {len(augmented_data)} new sentences generated.")

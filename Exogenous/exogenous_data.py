import numpy as np
import torch
from torch.utils.data import Subset
import random
import requests
from transformers import AutoModel, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import os

def read_conll_file(file_path):
    sentences, ner_tags = [], []
    tokens, tags = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    ner_tags.append(tags)
                    tokens, tags = [], []
                continue
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 4:
                tokens.append(parts[0])
                tags.append(parts[-1])

    if tokens:
        sentences.append(tokens)
        ner_tags.append(tags)
    return sentences, ner_tags

def load_conll_dataset(data_dir):
    all_sents, all_tags = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(".conll"):
            sents, tags = read_conll_file(os.path.join(data_dir, filename))
            all_sents.extend(sents)
            all_tags.extend(tags)
    return all_sents, all_tags

def prepare_datasets(data_dir, low_resource_samples=100, seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sentences, ner_tags = load_conll_dataset(data_dir)

    # Create Dataset
    dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": ner_tags})

    # 100 Random Samples
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)  # Seed-controlled shuffle
    train_indices = indices[:low_resource_samples]

    train_samples = [
        (dataset[i]['tokens'], dataset[i]['ner_tags'])
        for i in train_indices
    ]

    return train_samples

import os
import random
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Subset

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "replace\file\path\for\data\directory"
NUM_LLM_SAMPLES = 25       # samples taken to generate augmented data

train_samples = prepare_datasets(DATA_DIR, seed=SEED)

random.seed(SEED)
llm_samples = random.sample(train_samples, NUM_LLM_SAMPLES)

def get_few_shot_examples(data, num_examples=3):
    import random
    sampled_data = random.sample(data, min(num_examples, len(data)))
    formatted_examples = "\n".join([
        f"Sentence: {' '.join(tokens)}\nEntities: {labels}"
        for tokens, labels in sampled_data
    ])
    return formatted_examples

sample_idx = 2
tokens, labels = train_samples[sample_idx]

print("📦 ORIGINAL FORMAT:")
print("Tokens:", tokens)
print("NER Tags:", labels)

print("\n📝 FEW-SHOT FORMAT (from get_few_shot_examples):")
formatted = get_few_shot_examples([train_samples[sample_idx]], num_examples=1)
print(formatted)

def generate_prompt_deepseek(original_sentence, labels, use_incontext=False, few_shot_data=None, num_outputs=5):
    entity_guide = """
Entity tagging follows:
- B-<EntityType>: Beginning of entity
- I-<EntityType>: Inside of entity
- O: Outside any entity

Maintain the order and type of entities as in the examples.
"""

    if use_incontext:
        examples = few_shot_data
        prompt = f"""
You are an expert AI designed for generating high-quality training data for Named Entity Recognition (NER).

Task: Create {num_outputs} sentence variations. For each variation:
- Maintain the same number, order, and type of named entities as shown.
- For each generarted - Use the format:
  Sentence: <sentence>
  Entities: <list of entity tags>
- The new sentence (same meaning but rephrased) and The token-level NER tags using the B/I/O scheme
- Give examples without explanations commas or full stops

{entity_guide}

Examples:
{examples}

Input:
Sentence: {original_sentence}
Entity Tags: {labels}

Output:
"""
    else:
          prompt = f"""
Generate {num_outputs} diverse paraphrases of the sentence below, ensuring entity labels remain consistent.

Instructions:
- Maintain the same number, order, and type of named entities as shown.
- Radically alter sentence structure (e.g., voice, clauses, grammar).
- Change non-entity context significantly (e.g., synonyms, temporal/spatial shifts).
- Maintain entity order and types using B/I/O tags.
- Give examples without explanations commas or full stops
- use format
  Sentence: <sentence>
  Entities: <list of entity tags>

{entity_guide}

Original Sentence: {original_sentence}
Original Entity Tags: {labels}

Output:
"""

    return prompt.strip().split("\n"), prompt.strip()

# Send request to DeepSeek via OpenRouter
def send_prompt_to_deepseek(prompt, model="deepseek/deepseek-chat-v3-0324:free", max_tokens=512, temperature=0.7):
    url = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = "replace-api-key"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    print("\n📤 Sending Prompt to DeepSeek via OpenRouter...")
    print("📝 Prompt Preview:\n", prompt)

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()["choices"][0]["message"]["content"]
        print("✅ Response received.")
        return output.strip().split("\n")
    else:
        print(f"❌ Failed to generate. Status Code: {response.status_code}")
        print(response.text)
        return []

def parse_llm_output(llm_lines):
    results = []
    current_sentence = ""
    current_tags = []

    for line in llm_lines:
        line = line.strip()
        if line.lower().startswith("sentence:"):
            current_sentence = line[len("Sentence:"):].strip()
        elif line.lower().startswith("entities:"):
            tag_line = line[len("Entities:"):].strip()
            try:

                tags = eval(tag_line) if isinstance(tag_line, str) else tag_line
                results.append((current_sentence, tags))
            except:
                continue
        elif line.lower().startswith("entity tags:"):
            tag_line = line[len("Entities:"):].strip()
            try:

                tags = eval(tag_line) if isinstance(tag_line, str) else tag_line
                results.append((current_sentence, tags))
            except:
                continue
    return results

def generate_augmented_data(original_sentence, labels, use_incontext=False, few_shot_data=None, num_augmentations=5, token_limit=20, verbose=True):
    lines, prompt = generate_prompt_deepseek(
        original_sentence, labels,
        use_incontext=use_incontext,
        few_shot_data=few_shot_data,
        num_outputs=num_augmentations
    )

    if verbose:
        print("\n📝 Prompt Preview:\n", prompt)
        print("\n📤 Sending Prompt to DeepSeek via OpenRouter...")

    try:
        responses = send_prompt_to_deepseek(prompt)
    except Exception as e:
        print("❌ API Call Failed:", str(e))
        return []

    if verbose:
        print("\n✅ Raw Output:\n", "\n".join(responses))

    parsed_results = parse_llm_output(responses)

    for i, (sent, ents) in enumerate(parsed_results):
        print(f"\n✅ [{i+1}] Sentence: {sent}")
        print(f"   Tags: {ents}")

    return parsed_results

import time

NUM_SAMPLES_LLM = len(llm_samples)
augmented_data = []

for sample_idx in range(NUM_SAMPLES_LLM):
    tokens, labels = llm_samples[sample_idx]
    sentence = " ".join(tokens)

    print(f"Sample {sample_idx + 1}/{NUM_SAMPLES_LLM}:")

    style1_data = generate_augmented_data(
        original_sentence=sentence,
        labels=labels,
        use_incontext=False,
        num_augmentations=5
    )
    time.sleep(5)

    style2_data = generate_augmented_data(
        original_sentence=sentence,
        labels=labels,
        use_incontext=True,
        few_shot_data=get_few_shot_examples(llm_samples, 3),
        num_augmentations=5
    )
    time.sleep(5)

    augmented_data.extend(style1_data + style2_data)

def to_conll_format(data):
    """Convert (sentence, tags) tuples to CONLL format lines"""
    conll_lines = []
    for sentence, tags in data:
        tokens = sentence.split()
        for token, tag in zip(tokens, tags):
            # Original CONLL format: Token _ _ NER_Tag
            conll_lines.append(f"{token}\t_\t_\t{tag}")
        conll_lines.append("")
    return conll_lines

# Convert original 100 samples + augmented data
original_conll = []
for tokens, tags in train_samples:  # original 100 samples
    for token, tag in zip(tokens, tags):
        original_conll.append(f"{token}\t_\t_\t{tag}")
    original_conll.append("")

augmented_conll = to_conll_format(augmented_data)

output_path = "replace\file\path\for\saving\augmented_data"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(final_conll))

print(f"✅ Saved {len(final_conll)//15} sentences to {output_path}")


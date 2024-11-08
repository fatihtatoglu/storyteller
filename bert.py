import requests
import os
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch


def clean_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Identify the start and end indices
    start_idx = text.find("ACT I")
    end_idx = text.find("*** END OF THE PROJECT GUTENBERG")

    # Extract the main content
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    # Clean up extra spaces and newlines
    text = text.replace("\n\n", "\n").strip()

    return text


def split_text(text, max_length=512):
    sentences = text.split("\n")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) > max_length:
            chunks.append(chunk)
            chunk = ""
        chunk += sentence + "\n"

    if chunk:
        chunks.append(chunk)

    return chunks


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)


##### MAIN #####

# Download and save the text
url = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
response = requests.get(url)

with open("romeo_and_juliet.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

# Clean the text
cleaned_text = clean_text("romeo_and_juliet.txt")
with open("cleaned_romeo_and_juliet.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# Split the text into chunks
text_chunks = split_text(cleaned_text)

# Create a Dataset
dataset = Dataset.from_dict({"text": text_chunks})

# Initialize a BertTokenizerFast and train it on your data
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Train tokenizer on your dataset
tokenizer = tokenizer.train_new_from_iterator(dataset["text"], vocab_size=30522)

# Save the tokenizer
save_dir = "_bert_tokenizer"
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_pretrained(save_dir)

# Update the config to match the tokenizer vocab size
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)

# Initialize the model
model = BertForMaskedLM(config)

# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Create Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./_bert_romeo_and_juliet",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    report_to="none",  # Disable logging to Wandb or other services
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./_bert_romeo_and_juliet_model")
tokenizer.save_pretrained("./_bert_romeo_and_juliet_tokenizer")

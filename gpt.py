import requests
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch  # PyTorch için


def clean_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Başlangıç ve bitiş kısımlarını tanımlayın
    start_idx = text.find("ACT I")
    end_idx = text.find("*** END OF THE PROJECT GUTENBERG")

    # İçeriği ayırın
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    # Fazladan boşlukları ve satır başlarını temizleyelim
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


# Tokenize edilmiş metinler için işlev tanımlama
def tokenize_function_gpt(examples):
    return tokenizer_gpt(examples["text"], truncation=True, max_length=128)


##### MAIN #####

# Romeo and Juliet metnini indirip kaydedelim
url = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
response = requests.get(url)

with open("romeo_and_juliet.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

# Temizlenmiş metni kaydetme
cleaned_text = clean_text("romeo_and_juliet.txt")
with open("cleaned_romeo_and_juliet.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# Bölünmüş metni elde etme
text_chunks = split_text(cleaned_text)
# print(f"Bölünmüş metin örneği: {text_chunks[:2]}")

# Dataset oluşturma
dataset = Dataset.from_dict({"text": text_chunks})

# Sıfırdan GPT modeli tanımlama
config_gpt = GPT2Config(
    vocab_size=50257,  # Kelime haznesi boyutu
    n_positions=1024,  # Maksimum konum
    n_ctx=1024,  # Maksimum giriş uzunluğu
    n_embd=768,  # Gömme boyutu
    n_layer=12,  # Katman sayısı
    n_head=12,  # Çoklu başlık sayısı
)

model_gpt = GPT2LMHeadModel(config_gpt)

# Tokenizer'ı eğitme
tokenizer_gpt = ByteLevelBPETokenizer()
tokenizer_gpt.train(
    files="cleaned_romeo_and_juliet.txt",
    vocab_size=50257,
    min_frequency=2,
    special_tokens=[
        "<|endoftext|>",
    ],
)

# Tokenizer'ı kaydetme
save_dir = "_gpt_tokenizer"
os.makedirs(save_dir, exist_ok=True)
tokenizer_gpt.save_model("_gpt_tokenizer")

# `ByteLevelBPETokenizer`'dan `GPT2TokenizerFast`'e yükleme
tokenizer_gpt = GPT2TokenizerFast.from_pretrained(save_dir)

# PAD token ekleyin
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

# Dataset ve tokenize etme
tokenized_dataset_gpt = dataset.map(tokenize_function_gpt, batched=True)

# GPT için data collator
data_collator_gpt = DataCollatorForLanguageModeling(tokenizer=tokenizer_gpt, mlm=False)

training_args_gpt = TrainingArguments(
    output_dir="./_gpt_romeo_and_juliet",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer oluşturma
trainer_gpt = Trainer(
    model=model_gpt,
    args=training_args_gpt,
    data_collator=data_collator_gpt,
    train_dataset=tokenized_dataset_gpt,
)

# Modeli eğitme
trainer_gpt.train()

# GPT modelini ve tokenizer'ı kaydetme
model_gpt.save_pretrained("./_gpt_romeo_and_juliet_model")
tokenizer_gpt.save_pretrained("./_gpt_romeo_and_juliet_tokenizer")

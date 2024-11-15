import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def generate_text(prompt, max_length=200, num_return_sequences=1, temperature=1.2, top_k=50, top_p=0.9, no_repeat_ngram_size=2):
    # Girdi metnini tokenize edin ve attention_mask'i alın
    inputs = tokenizer.encode_plus(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Metin üretimi
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=5,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )

    # Üretilen metni çözümleyin
    generated_texts = [
        tokenizer.decode(out, skip_special_tokens=True) for out in output
    ]

    return generated_texts

##### MAIN #####
# Model ve tokenizer'ı yükleyin
model_path = "./_gpt_combined_books_model"
tokenizer_path = "./_gpt_combined_books_tokenizer"

tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Cihaz seçimi (GPU varsa kullanılır, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pad token ekleme (eğer yoksa)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model.config.pad_token_id = tokenizer.pad_token_id

print("\n\n\n\n\n\n\n\n")
user_prompt_o1 = "Walking alone in the dark forest, the young man listened to the sounds coming from around him."
result_o1 = generate_text(user_prompt_o1)
print("User: " + user_prompt_o1)
print("Model: ")
for idx, text in enumerate(result_o1, 1):
        print(f"\nStory {idx}:\n{text}\n")


print("\n\n\n\n\n\n\n\n")
user_prompt_o2 = "Once upon a time"
result_o2 = generate_text(user_prompt_o2)
print("User: " + user_prompt_o2)
print("Model: ")
for idx, text in enumerate(result_o2, 1):
        print(f"\nStory {idx}:\n{text}\n")


print("\n\n\n\n\n\n\n\n")
user_prompt_o3 = "Magic forest, lost treasure, brave knight"
result_o3 = generate_text(user_prompt_o3)
print("User: " + user_prompt_o3)
print("Model: ")
for idx, text in enumerate(result_o3, 1):
        print(f"\nStory {idx}:\n{text}\n")

from huggingface_hub import login
# Changed the secret key to a more common name. Make sure you have a secret named 'HF_TOKEN'
# in your Colab secrets with your Hugging Face token as the value.
print("enter token")
token = input()

login(token)
print("Successfully logged in to Hugging Face!")


from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import torch

# 1. Load model in 4-bit
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # prevent pad token warnings

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model = prepare_model_for_kbit_training(model)

# 2. Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# 3. Load your dataset from WhatsApp-cleaned data (replace with your own)
import json

with open("cleaned_chats.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list([
    {
        "instruction": sample["role"],   # e.g., "Ayush Daga"
        "response": sample["text"]       # e.g., "Kumbalgarh"
    }
    for sample in data
])

# 4. Tokenize
def tokenize(example):
    prompt = f"User: {example['instruction']}\nFriend: {example['response']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=False)

# 5. Training arguments optimized for P100
args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,  # P100: safe with batch_size=1
    gradient_accumulation_steps=8,  # Effective batch size = 8
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",  # needed for 4-bit
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.model.save_pretrained("/root/home/")
tokenizer.save_pretrained("/root/home/")

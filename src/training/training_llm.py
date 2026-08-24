from data.training.llm_training_data import generation_training_data
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

SYSTEM_PROMPT = (
    "You are a technical documentation assistant. "
    "Answer only using the supplied context. "
    "Do not invent information. "
    "If the context is insufficient, say that the documentation "
    "does not contain enough information to answer the question."
)

records = []

for item in generation_training_data:
    records.append({
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n\n{item['context']}\n\n"
                    f"Question:\n{item['query']}"
                ),
            },
            {
                "role": "assistant",
                "content": item["answer"],
            },
        ]
    })

dataset = Dataset.from_list(records)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42,
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

training_args = SFTConfig(
    output_dir="models/qwen-rag-lora",

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",

    fp16=True,
    bf16=False,

    gradient_checkpointing=True,

    max_length=2048,

    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    quantization_config=bnb_config,
)

trainer.train()
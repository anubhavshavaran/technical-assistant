from src.ingestion.chunk_document import generate_chunks, find_chunk_by_id
from src.ingestion.load_documents import load_documents
from data.training.training_data import queries
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import TripletLoss
from datasets import Dataset

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

docs = load_documents()
chunks = generate_chunks(docs)

triplets = []

for query in queries:
    for positive in query["positives"]:
        for negative in query["negatives"]:
            triplets.append(
                {
                    "query": query["query"],
                    "positive": find_chunk_by_id(chunks, positive)["text"],
                    "negative": find_chunk_by_id(chunks, negative)["text"],
                }
            )

train_dataset = Dataset.from_dict({
    "anchor": [t["query"] for t in triplets],
    "positive": [t["positive"] for t in triplets],
    "negative": [t["negative"] for t in triplets],
})

loss = TripletLoss(
    model=model,
    triplet_margin=0.2,
)

args = SentenceTransformerTrainingArguments(
    output_dir="models/go-retriever-hard-negative",
    num_train_epochs=1,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()
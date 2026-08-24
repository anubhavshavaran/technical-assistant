from src.ingestion.chunk_document import generate_chunks
from src.ingestion.load_documents import load_documents
from data.training.training_data import queries
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
loss = MultipleNegativesRankingLoss(model)

docs = load_documents()
chunks = generate_chunks(docs)

training_data = []

for query in queries:
    for positive in query["positives"]:
        training_data.append(
            InputExample(
                texts=[query["query"], positive]
            )
        )

train_dataloader = DataLoader(
    training_data,
    shuffle=True,
    batch_size=4
)

model.fit(
    train_objectives=[
        (train_dataloader, loss)
    ],
    epochs=1,
    warmup_steps=10,
    output_path="models/go-retriever",
    show_progress_bar=True
)
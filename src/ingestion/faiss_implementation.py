import faiss
from embedding import embeddings, bi_encoder, chunks

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings)

index.add(embeddings.astype("float32"))

print("Number of vectors:", index.ntotal)

def faiss_search(query, index, bi_encoder, chunks, top_k=5):
    query_embedding = bi_encoder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")
    
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k)
    print(scores, indices)
    return [
        {
            "score": float(scores[0][i]),
            "chunk": chunks[indices[0][i]]
        }
        for i in range(top_k)
    ]
    
results = faiss_search(
    "What is net/http?",
    index,
    bi_encoder,
    chunks,
    top_k=5
)

for result in results:
    print("=" * 80)
    print("Score:", result["score"])
    print("Chunk:", result["chunk"]["id"])
    print(result["chunk"]["text"][:300])
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.ingestion.chunk_document import generate_chunks
from src.ingestion.load_documents import load_documents
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss

bi_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

docs = load_documents()
chunks = generate_chunks(docs)

embeddings = bi_encoder.encode([chunk["text"] for chunk in chunks], show_progress_bar=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings.astype("float32"))

query = "How to use Firebase?"

def retrieve(
    query,
    embeddings,
    chunks,
    bi_encoder,
    cross_encoder,
    retrieval_k=50,
    final_k=5
):
    query_embedding = bi_encoder.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:retrieval_k]

    candidates = [
        {
            "score": float(similarities[i]),
            "chunk": chunks[i],
        }
        for i in top_indices
    ]

    pairs = [
        (query, result["chunk"]["text"])
        for result in candidates
    ]

    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = [
        {
            "score": float(score),
            "chunk": result["chunk"],
        }
        for result, score in reranked[:final_k]
    ]

    return candidates, final_results

def retrieve_faiss(
    query,
    chunks=chunks,
    bi_encoder=bi_encoder,
    cross_encoder=cross_encoder,
    retrieval_k=50,
    final_k=5
):
    query_embedding = bi_encoder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")
    
    faiss.normalize_L2(query_embedding)

    scores, top_indices = index.search(query_embedding, retrieval_k)

    scores = scores[0]
    top_indices = top_indices[0]

    candidates = [
        {
            "score": float(scores[i]),
            "chunk": chunks[idx],
        }
        for i, idx in enumerate(top_indices)
    ]

    pairs = [
        (query, result["chunk"]["text"])
        for result in candidates
    ]

    rerank_scores  = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = [
        {
            "score": float(score),
            "chunk": result["chunk"],
        }
        for result, score in reranked[:final_k]
    ]

    return candidates, final_results
from src.ingestion.embedding import retrieve_faiss, chunks, bi_encoder, cross_encoder
import numpy as np

def recall_at_k(retrieved_results, relevant_ids, k):
    if not relevant_ids:
        return None

    retrieved_ids = [
        result["chunk"]["id"]
        for result in retrieved_results[:k]
    ]

    relevant_set = set(relevant_ids)

    hits = sum(
        chunk_id in relevant_set
        for chunk_id in retrieved_ids
    )

    return hits / len(relevant_set)

def get_mrr(final_results, relevant_chunk_ids):
    for rank, final_result in enumerate(final_results, start=1):
        if final_result['chunk']['id'] in relevant_chunk_ids:
            return 1 / rank
        
    return 0.0

evaluation_data = [
    {
        "query": "What is net/http?",
        "relevant_chunks": [
            "docs_0_chunk_166",
            "docs_0_chunk_148",
        ],
    },
    {
        "query": "How does middleware work in Go?",
        "relevant_chunks": [
            "docs_2_chunk_7",
            "docs_0_chunk_167",
        ],
    },
    {
        "query": "How to use Firebase?",
        "relevant_chunks": [
            "docs_2_chunk_10",
            "docs_2_chunk_18",
            "docs_2_chunk_20",
        ],
    },
    {
        "query": "How to prevent SQL Injection?",
        "relevant_chunks": [],
    },
    {
        "query": "What are the cryptographic algorithms used in Go?",
        "relevant_chunks": [
            "docs_1_chunk_18",
            "docs_1_chunk_19",
        ],
    },
    {
        "query": "How to do socket level programming?",
        "relevant_chunks": [
            "docs_0_chunk_144",
            "docs_0_chunk_27",
        ],
    },
    {
        "query": "How to setup logging using Go?",
        "relevant_chunks": [
            "docs_0_chunk_252",
            "docs_0_chunk_258",
            "docs_0_chunk_271",
        ],
    },
    {
        "query": "How to create parameterized queries in Go?",
        "relevant_chunks": [],
    },
    {
        "query": "What is file handling?",
        "relevant_chunks": [
            "docs_1_chunk_3",
        ],
    },
    {
        "query": "Build Domain-Driven Design (DDD) applications in Go",
        "relevant_chunks": [
            "docs_2_chunk_25",
            "docs_2_chunk_30",
            "docs_2_chunk_26",
        ],
    },
]

metrics = []
for eval_item in evaluation_data:
    query = eval_item["query"]
    relevant_chunks = eval_item["relevant_chunks"]

    if not relevant_chunks:
        continue

    candidates, final_results = retrieve_faiss(
        query,
        chunks,
        bi_encoder,
        cross_encoder,
        retrieval_k=50,
        final_k=5
    )

    bi_recall_5 = recall_at_k(
        candidates,
        relevant_chunks,
        5
    )

    bi_recall_50 = recall_at_k(
        candidates,
        relevant_chunks,
        50
    )

    reranked_recall_5 = recall_at_k(
        final_results,
        relevant_chunks,
        5
    )
    
    mrr = get_mrr(final_results, eval_item["relevant_chunks"]) 

    metrics.append({
        "query": query,
        "bi_recall_5": bi_recall_5,
        "bi_recall_50": bi_recall_50,
        "reranked_recall_5": reranked_recall_5,
        "mrr": mrr,
    })

    print("=" * 80)
    print("Query:", query)
    print("Bi-encoder Recall@5:", bi_recall_5)
    print("Bi-encoder Recall@50:", bi_recall_50)
    print("Reranked Recall@5:", reranked_recall_5)
    print("MRR:", mrr)

print("\nAVERAGES")
print(
    "Bi-encoder Recall@5:",
    np.mean([m["bi_recall_5"] for m in metrics])
)

print(
    "Bi-encoder Recall@50:",
    np.mean([m["bi_recall_50"] for m in metrics])
)

print(
    "Reranked Recall@5:",
    np.mean([m["reranked_recall_5"] for m in metrics])
)

print(
    "Average MRR:",
    np.mean([m["mrr"] for m in metrics])
)
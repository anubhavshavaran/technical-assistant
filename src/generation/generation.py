from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.ingestion.embedding import retrieve_faiss
from src.ingestion.load_documents import load_documents
from src.ingestion.chunk_document import generate_chunks

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BI_ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
CR_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

bi_encoder = SentenceTransformer(BI_ENCODER_NAME)
cr_encoder = CrossEncoder(CR_ENCODER_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

docs = load_documents()
chunks = generate_chunks(docs)

query = "Build Domain-Driven Design (DDD) applications in Go"

def build_context(results, max_chars_per_chunk=1500):
    contexts = []
    
    for i, result in enumerate(results, start=1):
        chunk = result['chunk']
        text = chunk["text"][:max_chars_per_chunk]
        
        contexts.append(
            f"[Source {i}]\n"
            f"Document: {chunk['metadata']['source']}\n"
            f"Chunk ID: {chunk['id']}\n"
            f"{text}"
        )
    
    return "\n\n".join(contexts)

def build_messages(context, query):
    return [
        {
            "role": "user",
            "content": (
                "You are a technical documentation assistant. "
                "Answer the user's question using only the provided context. "
                "Do not invent information that is not supported by the context. "
                "Cite the source using [Source N]. "
                "If the context does not contain enough information to answer "
                "the question, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n\n"
                f"{context}\n\n"
                f"Question:\n{query}"
            ),
        }
    ]

def generate_answer(query):
    _, retrieved_results = retrieve_faiss(
        query,
        chunks,
        bi_encoder,
        cr_encoder,
        retrieval_k=50,
        final_k=1
    )
    
    context = build_context(retrieved_results, 2000)

    messages = build_messages(context, query)
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return {
        "query": query,
        "answer": answer,
        "retrieved_results": retrieved_results,
        "context": context
    }

answer = generate_answer(query)
print(answer)
def chunk_document(doc):
    words = doc["text"].split()
    index = 0
    head = 0
    tail = 500

    chunks = []
    while head < len(words):
        chunks.append({
            "id": f"{doc['id']}_chunk_{index}",
            "text": " ".join(words[head:tail]),
            "metadata": {
                "source": doc["metadata"]["source"],
                "chunk_index": index
            }
        })

        head += 400
        tail += 500
        index += 1

    return chunks

def generate_chunks(data):
    chunks = []
    for doc in data:
        chunk = chunk_document(doc)
        chunks.extend(chunk)

    return chunks
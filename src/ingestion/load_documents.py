import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./../../data/raw/")

def load_document(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {path}")


def load_documents():
    docs = [f"docs_{i}.txt" for i in range(3)]

    data = []
    for doc in docs:
        content = load_document(os.path.join(BASE_DIR, doc))
        data.append({
            "id": doc.split(".")[0],
            "text": content,
            "metadata": {
                "source": doc,
            }
        })
    return data
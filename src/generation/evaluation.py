from data.evaluation.rag_eval_data import evaluation_data
from src.generation.generation import generate_answer
import numpy as np

def fact_coverage(fact_scores):
    if not fact_scores:
        return 0.0

    return sum(fact_scores) / len(fact_scores)

rag_results = []
for item in evaluation_data:
    result = generate_answer(item["query"])
    
    rag_results.append({
        "query": item["query"],
        "answer": result["answer"],
        "context": result["context"],
        "retrieved_results": result["retrieved_results"],
        "expected_facts": item["expected_facts"],
        "relevant_chunks": item["relevant_chunks"],

        "fact_scores": [],
        "grounded": None,
        "abstained_correctly": None,
    })

for result in rag_results:
    print("=" * 100)
    print("Query", result["query"])
    print("Answer", result["answer"])
    print("\nExpected facts:")
    for i, fact in enumerate(result["expected_facts"], start=1):
        print(i, fact)
    
    print("Context:")
    print(result["context"])
        
answerable = [
    r for r in rag_results
    if r["abstained_correctly"] is None
]

average_fact_coverage = np.mean([
    r["fact_coverage"]
    for r in answerable
])

groundedness_rate = np.mean([
    r["grounded"]
    for r in answerable
])

abstention_cases = [
    r for r in rag_results
    if r["abstained_correctly"] is not None
]

abstention_accuracy = np.mean([
    r["abstained_correctly"]
    for r in abstention_cases
])


print("Average Fact Coverage:", average_fact_coverage)
print("Groundedness Rate:", groundedness_rate)
print("Abstention Accuracy:", abstention_accuracy)
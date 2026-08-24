generation_training_data = [
    {
        "query": "How do you prevent SQL injection in Go?",
        "context": """
            [Source 1]
            Document: docs_1.txt
            Prepared Statements with Parameterized Queries are the best
            and most secure way to protect against SQL Injections...
        """,
        "answer": (
            "Use prepared statements with parameterized queries. "
            "Avoid constructing SQL queries by concatenating "
            "user-controlled values. Placeholder syntax depends "
            "on the database."
        ),
    },
]
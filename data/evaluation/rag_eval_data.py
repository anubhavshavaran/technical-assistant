evaluation_data = [
    {
        "query": "What is net/http?",
        "relevant_chunks": [
            "docs_0_chunk_166",
            "docs_0_chunk_148",
        ],
        "expected_facts": [
            "net/http is a Go standard library package",
            "it provides HTTP client functionality",
            "it provides HTTP server functionality",
            "an HTTP server uses handlers, middleware, and a multiplexer",
        ],
    },

    {
        "query": "How does middleware work in Go?",
        "relevant_chunks": [
            "docs_2_chunk_7",
            "docs_0_chunk_196",
        ],
        "expected_facts": [
            "middleware can run before and after a request is executed",
            "middleware has access to the http.Request",
            "middleware can be used for logging",
            "middleware can be used for authentication",
            "Go middleware can be represented as a function that accepts an http.Handler and returns an http.Handler",
        ],
    },

    {
        "query": "How is Firestore used in the Go application?",
        "relevant_chunks": [
            "docs_2_chunk_10",
        ],
        "expected_facts": [
            "the Firestore client is created with firestore.NewClient",
            "the project ID is obtained from the GCP_PROJECT environment variable",
            "the Firestore client can be used to access a collection",
            "the example queries the trainer-hours collection",
        ],
    },

    {
        "query": "How does Firebase Authentication work in the application?",
        "relevant_chunks": [
            "docs_2_chunk_18",
            "docs_2_chunk_19",
        ],
        "expected_facts": [
            "the frontend initializes the Firebase SDK",
            "login uses Firebase Authentication",
            "signInWithEmailAndPassword is used for login",
            "getIdToken is used to obtain a JWT",
            "the backend verifies the Firebase ID token",
        ],
    },

    {
        "query": "How do you prevent SQL injection in Go?",
        "relevant_chunks": [
            "docs_1_chunk_6",
            "docs_1_chunk_34",
        ],
        "expected_facts": [
            "string concatenation with user-controlled values can cause SQL injection",
            "prepared statements with parameterized queries are recommended",
            "query placeholders separate SQL structure from user-provided values",
            "placeholder syntax depends on the database",
        ],
    },

    {
        "query": "What cryptographic algorithms and primitives are discussed in the Go security guide?",
        "relevant_chunks": [
            "docs_1_chunk_18",
            "docs_1_chunk_19",
        ],
        "expected_facts": [
            "AES is discussed as a symmetric encryption algorithm",
            "ChaCha20Poly1305 is discussed as a modern symmetric encryption algorithm with authentication",
            "x/crypto/nacl provides NaCl abstractions",
            "nacl/box is used for authenticated encrypted communication using public-key cryptography",
            "nacl/secretbox is used for authenticated encrypted communication using symmetric cryptography",
        ],
    },

    {
        "query": "How do you create socket-level programs in Go?",
        "relevant_chunks": [
            "docs_0_chunk_98",
            "docs_0_chunk_108",
            "docs_0_chunk_117",
        ],
        "expected_facts": [
            "the net package provides networking interfaces",
            "net.Dial can establish a client connection",
            "net.Listen can create a listening socket",
            "Go supports Unix domain socket types such as unix, unixgram, and unixpacket",
            "net.Conn abstracts communication across different network types",
        ],
    },

    {
        "query": "How is structured logging implemented in Go?",
        "relevant_chunks": [
            "docs_0_chunk_400",
            "docs_0_chunk_405",
        ],
        "expected_facts": [
            "structured logging adds metadata to log entries",
            "key-value pairs can be included in structured log entries",
            "structured loggers commonly encode entries as JSON",
            "log levels can be used to control which entries are emitted",
            "Zap provides multiple log levels",
        ],
    },

    {
        "query": "How does the Go example load data from a file?",
        "relevant_chunks": [
            "docs_0_chunk_228",
            "docs_0_chunk_229",
        ],
        "expected_facts": [
            "the example checks whether the data file exists",
            "os.Open is used to open the file",
            "the file is closed after use",
            "the application loads serialized data from the file",
        ],
    },

    {
        "query": "How is Domain-Driven Design applied in the Go application?",
        "relevant_chunks": [
            "docs_2_chunk_30",
            "docs_2_chunk_31",
            "docs_2_chunk_33",
            "docs_2_chunk_48",
        ],
        "expected_facts": [
            "domain logic is reflected directly in the code",
            "domain types should contain behavior rather than being only data structures",
            "business rules are implemented in the domain layer",
            "the domain should remain database agnostic",
            "the Repository pattern is used to abstract database implementations",
        ],
    },

    # These are deliberately outside the documented material or not sufficiently
    # supported by the retrieved corpus. They test whether the RAG system abstains.
    {
        "query": "How do I configure a Redis Cluster?",
        "relevant_chunks": [],
        "expected_facts": [
            "Redis Cluster configuration",
            "cluster node configuration",
            "Redis cluster topology",
        ],
    },

    {
        "query": "How do I fine-tune a Qwen language model with LoRA?",
        "relevant_chunks": [],
        "expected_facts": [
            "LoRA fine-tuning",
            "Qwen model fine-tuning",
            "training configuration for LoRA",
        ],
    },
]
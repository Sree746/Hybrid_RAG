from src.embeddings.embedder import EmbeddingModel


def main():

    model = EmbeddingModel()

    texts = [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning uses neural networks.",
        "The weather is sunny today."
    ]

    embeddings = model.embed_documents(texts)

    print(
        "Number of embeddings:",
        len(embeddings)
    )

    print(
        "Embedding dimension:",
        len(embeddings[0])
    )

    query = "What is artificial intelligence?"

    query_embedding = model.embed_query(
        query
    )

    print(
        "Query embedding dimension:",
        len(query_embedding)
    )


if __name__ == "__main__":
    main()
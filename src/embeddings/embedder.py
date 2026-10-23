from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Load the embedding model.

        Parameters
        ----------
        model_name : str
            Name of the Sentence Transformer model.
        """

        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> list[list[float]]:
        """
        Convert multiple documents/chunks into embeddings.

        Parameters
        ----------
        texts : list[str]
            List of document/chunk texts.

        batch_size : int
            Number of texts processed at once.

        Returns
        -------
        list[list[float]]
            Embedding vector for each text.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a user query into an embedding.

        Parameters
        ----------
        query : str
            User's question.

        Returns
        -------
        list[float]
            Query embedding.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Return dimensionality of the embedding vectors.
        """

        return self.model.get_sentence_embedding_dimension()
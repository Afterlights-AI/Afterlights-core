from sentence_transformers import SentenceTransformer

class EmbeddingModelController:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str|list[str]):
        """
        Embed a list of options using the model.
        
        Args:
            options (list): A list of strings to be embedded.
        
        Returns:
            list: A list of embeddings corresponding to the input options.
        """
        embeddings = self.model.encode(text)
        return embeddings


from qdrant_client import QdrantClient


class QdrantConnector:
    def connect(self, url: str = "http://localhost:6333",  # Default Qdrant URL
                prefer_grpc: bool = False,
                **client_kwargs):
        """
        Connect to Qdrant using the provided URL and optional parameters.

        Args:
            url (str): The URL of the Qdrant instance.
            prefer_grpc (bool): Whether to prefer gRPC over REST. Defaults to False.
            **client_kwargs: Additional keyword arguments for the Qdrant client.

        Returns:
            None
        """

        # Initialize the Qdrant client with the provided parameters
        return QdrantClient(url=url, prefer_grpc=prefer_grpc, **client_kwargs)

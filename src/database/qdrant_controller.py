from __future__ import annotations
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    PointIdsList,
    Filter,           # optional for advanced search / delete
)

class QdrantController:
    """
    Thin convenience layer around `qdrant_client.QdrantClient`
    ---------------------------------------------------------
    * Collection-level: create & drop
    * Point-level: upsert (create / update), read, search, delete
    """

    # ---------- connection --------------------------------------------------

    def __init__(
        self,
        client
    ):
        """
        Parameters
        ----------
        url : str
            REST endpoint of the running Qdrant service
        prefer_grpc : bool
            If True, use gRPC where possible (faster for large batches)
        **client_kwargs
            Any extra args accepted by `QdrantClient`
        """
        self.client = client

    # ---------- collections -------------------------------------------------

    def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str | Distance = "cosine",
        **kwargs,
    ) -> None:
        """Create a new collection (throws if it already exists)."""
        # if self.collection_exists(name):
        #     raise ValueError(f"Collection '{name}' already exists")
        # Accept both enum members and plain strings
        metric = (
            distance
            if isinstance(distance, Distance)
            else getattr(Distance, distance.upper())
        )
        self.client.create_collection(                    # :contentReference[oaicite:0]{index=0}
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=metric),
            **kwargs,
        )

    def delete_collection(self, name: str, **kwargs) -> None:
        """Drop the collection and all its points."""
        self.client.delete_collection(collection_name=name, **kwargs)  # :contentReference[oaicite:1]{index=1}

    # ---------- points ------------------------------------------------------

    @staticmethod
    def make_point(
        point_id: int | str,
        vector: list[float],
        payload: dict | None = None,
    ) -> PointStruct:
        """Helper to build a `PointStruct` in one line."""
        return PointStruct(id=point_id, vector=vector, payload=payload or {})

    def upsert_points(
        self,
        collection: str,
        points: list[PointStruct],
        wait: bool = True,
        **kwargs,
    ):
        """
        Create **or** update points (Qdrant's `upsert`).
        The same call handles *C* and *U* in CRUD.
        """
        return self.client.upsert(                         # :contentReference[oaicite:2]{index=2}
            collection_name=collection,
            points=points,
            wait=wait,
            **kwargs,
        )
    def batch_struct_points(
        self,
        points: list[list],
        wait: bool = False,
        **kwargs,
    ):
        """
        Batch upsert a list of embeddings as points.
        Each item in `points` should be [id, vector, payload (optional)].
        """
        point_structs = []
        for idx, p in enumerate(points):
            text = p['text']
            talker = p['talker']
            time = p['time']
            payload = {
                'text': text,
                'talker': talker,
                'time': time,
            }
            point_structs.append(
                self.make_point(
                    point_id=idx,
                    vector=p['embedding'], 
                    payload=payload
                ))
        return point_structs
    

    # ---------- read --------------------------------------------------------
    def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists."""
        return self.client.collection_exists(collection_name=collection)
    
    def get_points(
        self,
        collection: str,
        ids: list[int | str],
        with_vectors: bool = False,
        with_payload: bool = True,
    ):
        """Direct point lookup by primary key(s)."""
        return self.client.retrieve(
            collection_name=collection,
            ids=ids,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        query_filter: Filter | None = None,
        **kwargs,
    ):
        """K-NN search (vector similarity, optionally filtered)."""
        return self.client.search(                         # :contentReference[oaicite:3]{index=3}
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            **kwargs,
        )

    # ---------- delete points ----------------------------------------------

    def delete_points(
        self,
        collection: str,
        ids: list[int | str],
        wait: bool = True,
        **kwargs,
    ):
        """
        Physically remove points by id.
        (Pass a Filter instead of `PointIdsList` to delete by payload-based criteria.)
        """
        return self.client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=ids),
            wait=wait,
            **kwargs,
        )
    def drop_collection(self, collection: str, **kwargs) -> None:
        """Alias for delete_collection for convenience."""
        return self.client.delete_collection(collection, **kwargs)
    
    def make_filter(self, field: str, value) -> Filter:
        """
        Helper to create a Qdrant Filter for a field-value match.
        """
        return models.Filter(
            should=[
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value)
                )
            ]
        )
    
    def text_search(
        self,
        collection: str,
        scroll_filter: Filter,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ):
        """
        Search for points where a specific field matches a text value.
        """

        return self.client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            **kwargs,
        )
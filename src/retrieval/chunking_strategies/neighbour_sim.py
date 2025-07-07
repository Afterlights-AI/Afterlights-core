import re
from typing import Callable, List, Optional, Sequence
from model_management.embedding_model_controller import EmbeddingModelController
import numpy as np
from tqdm import tqdm

class NeibourSimilarityChunker:
    """
    
    text_chunker.py
    ---------------
    A hierarchical text chunker that keeps semantic integrity while respecting a
    maximum-token threshold.

    Preface rules implemented
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    * **Threshold**: 150 tokens (configurable via *max_tokens* argument).
    * **Delimiters (in order)**:
    1. Double newline ("\n\n")
    2. Single newline ("\n")
    3. Period (".")
    4. Comma (",")

    The algorithm recursively splits any segment that exceeds the threshold using
    the next delimiter in the sequence, guaranteeing that each returned chunk has
    no more than *max_tokens* tokens - unless it can no longer be split.

    A pluggable *tokenizer* callback lets you swap in a model-specific tokenizer
    (e.g., tiktoken). A very simple regex tokenizer is provided as default
    fallback.
    """

    def __init__(self, embedding_model_name: str):
        self.model_name = embedding_model_name

    def detect_language(self, text: str) -> str:
        """Detect if the text is English or Chinese (very basic heuristic)."""
        # Count Chinese characters (CJK Unified Ideographs)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # Count ASCII letters
        english_chars = re.findall(r'[A-Za-z]', text)
        if len(chinese_chars) > len(english_chars):
            return "chinese"
        return "english"

    def default_tokenizer(self, text: str) -> List[str]:
        """A naive tokenizer that roughly approximates OpenAI token counting.

        Splits on words and standalone punctuation characters. You can replace this
        with ``tiktoken.encoding_for_model(<model>).encode`` for exact counts.
        """
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    
    def cosine_similarity_matrix(self, emb: np.ndarray) -> np.ndarray:
        """Fully-vectorised cosine similarity matrix (diagonal == 1)."""
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9  # avoid divide-by-zero
        emb_norm = emb / norm
        
        return emb_norm @ emb_norm.T

    def chunk_by_similarity(
        self,
        chunks: Sequence[str],
        *,
        similarity_threshold: float = 0.85,
    ) -> List[List[str]]:
        

        """Order-preserving agglomerative clustering that always merges the most
        similar **adjacent** pair first.

        Parameters
        ----------
        chunks:
            List of text chunks (from :func:`chunk_text`).
        similarity_threshold:
            Minimum cosine similarity required to merge.
        embedder:
            Function mapping *chunks* → ``(n_chunks, dim)`` embedding matrix.
        """
        if not chunks:
            return []
        
        

        n = len(chunks)
        if n == 0:
            return []
        if n == 1:
            return [[chunks[0]]]

        # Compute embeddings and similarity matrix once at singleton level.
        embedder = EmbeddingModelController(model_name=self.model_name)
        embeddings = embedder.embed(chunks)
        sim_mat = self.cosine_similarity_matrix(embeddings)

        # Each cluster is a list of *original* indices it covers.
        clusters: List[List[int]] = [[i] for i in range(n)]

        def _mean_similarity(a: Sequence[int], b: Sequence[int]) -> float:
            sims = [sim_mat[i, j] for i in a for j in b]
            return float(sum(sims) / len(sims)) if sims else -1.0

        while True:
            best_sim = -1.0
            best_i = -1

            # Scan current neighbour list for the strongest similarity.
            for i in range(len(clusters) - 1):
                sim = _mean_similarity(clusters[i], clusters[i + 1])
                if sim > best_sim:
                    best_sim = sim
                    best_i = i

            # Stop if nothing meets the threshold.
            if best_sim < similarity_threshold or best_i == -1:
                break

            # Merge the best pair and continue.
            merged_cluster = clusters[best_i] + clusters[best_i + 1]
            clusters[best_i : best_i + 2] = [merged_cluster]

        # Resolve index clusters back to text clusters in original order.
        return [[chunks[idx] for idx in cluster] for cluster in clusters], clusters
        
    def chunk_text(
        self,
        text: str,
        max_tokens: int = 150,
        *,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> List[str]:
        """Chunk *text* into semantically coherent pieces ≤ *max_tokens*.

        The splitting strategy follows these steps in order, only advancing to the
        next delimiter when a chunk still exceeds *max_tokens*:

        1. Double new line ("\n\n")
        2. Single new line ("\n")
        3. Period (".")
        4. Comma (",")

        Parameters
        ----------
        text:
            The input text (structured or unstructured).
        max_tokens:
            Maximum allowed tokens in each chunk (default 150).
        tokenizer:
            Optional callable that returns a list of tokens. Defaults to
            :func:`default_tokenizer`.

        Returns
        -------
        List[str]
            A list of chunks, each no longer than *max_tokens* tokens.
        """
        if tokenizer is None:
            tokenizer = self.default_tokenizer
        language = self.detect_language(text)
        if language == "chinese":
            delimiters = ["\n\n", "\n", "。", "，"]
        elif language == "english":
            
            delimiters = ["\n\n", "\n", ".", ","]
        

        def _split(segment: str, level: int) -> List[str]:
            """Recursively split *segment* starting at *level* delimiter."""
            if level >= len(delimiters):
                # No more delimiters; return as is to preserve integrity.
                return [segment.strip()]

            # If the current segment already satisfies the limit, keep it whole.
            if len(tokenizer(segment)) <= max_tokens:
                return [segment.strip()]

            delim = delimiters[level]
            parts = segment.split(delim)

            chunks: List[str] = []
            buffer = ""
            # Re-assemble greedily while respecting the limit.
            for i, part in enumerate(parts):
                # Restore the delimiter we split on *except* for double‐newline to
                # keep paragraphs intact.
                append_delim = delim if level >= 2 and i < len(parts) - 1 else ""
                candidate = (buffer + part + append_delim).strip()

                if not candidate:
                    continue

                if len(tokenizer(candidate)) > max_tokens:
                    # The candidate is too large. Finalize current buffer and recurse.
                    if buffer:
                        chunks.extend(_split(buffer.strip(), level + 1))
                        buffer = part + append_delim
                    else:
                        # Single unit bigger than limit; try deeper splitting.
                        chunks.extend(_split(part + append_delim, level + 1))
                        buffer = ""
                else:
                    buffer = candidate + (" " if level < 2 else "")

            if buffer:
                chunks.append(buffer.strip())

            # Post-process in case nested chunks are still oversized.
            refined: List[str] = []
            for ch in chunks:
                refined.extend(_split(ch, level + 1))
            return refined

        # Normalize Windows line endings for consistency.
        normalized = text.replace("\r\n", "\n")
        return [c for c in _split(normalized, 0) if c]


    # -----------------------------------------------------------------------------
    # Example usage
    # -----------------------------------------------------------------------------
if __name__ == "__main__":
    with open("examples/cn_example_nazha_dataset.csv", "r", encoding="utf-8") as f:
        next(f)
        SAMPLE_TEXT = f.read()[:10000]

    chunker = NeibourSimilarityChunker(embedding_model_name="trained_model/nazha_model")
    chunks = chunker.chunk_text(SAMPLE_TEXT, max_tokens=150)
    for i, ch in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunker.default_tokenizer(ch))} tokens) ---")
        print(ch)
    clusters = chunker.chunk_by_similarity(chunks,  similarity_threshold=0.5)
    print("\nClusters after neighbour-mering (threshold=0.7):", len(clusters))
    for i, cluster in enumerate(clusters, 1):
        print(f"--- Cluster {i} ---")
        for ch in cluster:
            print(ch)
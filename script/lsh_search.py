from loguru import logger
import numpy as np
import json
from typing import List, Dict, Any

def load_embeddings(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading embeddings from {file_path}")
    embeddings = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            embeddings.append(entry)
    logger.success(f"Loaded {len(embeddings)} embeddings from {file_path}")
    return embeddings

class LSHIndex:
    def __init__(self, num_hash_tables: int = 10, hash_size: int = 16, dim: int = None):
        logger.info(f"Initializing LSH index with {num_hash_tables} hash tables and hash size {hash_size}")
        self.num_hash_tables = num_hash_tables
        self.hash_size = hash_size
        self.dim = dim
        self.hash_tables = [{} for _ in range(num_hash_tables)]
        self.hyperplanes = np.random.randn(num_hash_tables, hash_size, dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=2)[:, :, np.newaxis]
        logger.success("Random hyperplanes generated")

    def _hash(self, vector: np.ndarray) -> np.ndarray:
        projections = np.dot(self.hyperplanes, vector)
        return (projections >= 0).astype(int)

    def add(self, data: List[Dict[str, Any]]):
        logger.info("Adding data to LSH index")
        for idx, item in enumerate(data):
            vector = np.array(item['vector'])
            hash_codes = self._hash(vector)
            for i, hash_code in enumerate(hash_codes):
                bucket = self.hash_tables[i].setdefault(hash_code.tobytes(), [])
                bucket.append(item)
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} items")
        logger.success("All data added to LSH index")

    def query(self, vector: np.ndarray, num_results: int = 10) -> List[tuple]:
        candidates = []
        hash_codes = self._hash(vector)
        for i, hash_code in enumerate(hash_codes):
            bucket = self.hash_tables[i].get(hash_code.tobytes(), [])
            candidates.extend(bucket)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            # Use a hashable identifier for each candidate
            identifier = (candidate['label'], tuple(candidate['vector']))
            if identifier not in seen:
                seen.add(identifier)
                unique_candidates.append(candidate)
        
        if not unique_candidates:
            return []
        
        candidate_vectors = np.array([item['vector'] for item in unique_candidates])
        similarities = np.dot(candidate_vectors, vector) / (np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(vector))
        top_indices = np.argsort(similarities)[-num_results:][::-1]
        return [(similarities[i], unique_candidates[i]) for i in top_indices]

if __name__ == "__main__":
    logger.add("lsh_search.log", rotation="10 MB")
    embeddings = load_embeddings("embeddings_small.jsonl")
    dim = len(embeddings[0]['vector'])
    lsh_index = LSHIndex(num_hash_tables=10, hash_size=16, dim=dim)
    lsh_index.add(embeddings)
    query_embeddings = load_embeddings("query_embeddings_small.jsonl")
    for query in query_embeddings:
        query_vector = np.array(query['vector'])
        query_label = query['label']
        query_preview = query['metadata']['content_preview']
        logger.info(f"Searching for similar items to query {query_label}")
        logger.info(f"Query content preview: {query_preview}")
        results = lsh_index.query(query_vector, num_results=10)
        logger.info(f"Top results for query {query_label}:")
        for rank, (sim, item) in enumerate(results):
            logger.info(f"Rank {rank + 1}: Label={item['label']}, Similarity={sim:.4f}")
            logger.info(f"Content preview: {item['metadata']['content_preview']}")
        logger.info("---")
    logger.success("Search process completed")
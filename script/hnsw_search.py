from loguru import logger
import numpy as np
import json
import random
import heapq
from typing import List, Tuple, Dict, Any

def load_embeddings(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading embeddings from {file_path}")
    embeddings = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            embeddings.append(entry)
    logger.success(f"Loaded {len(embeddings)} embeddings from {file_path}")
    return embeddings

class HNSWNode:
    def __init__(self, data: Dict[str, Any], level: int):
        self.data = data
        self.level = level
        self.connections: List[List[int]] = [[] for _ in range(level + 1)]

class HNSWIndex:
    def __init__(self, M: int = 16, ef_construction: int = 200, ef: int = 50, max_elements: int = 1000000):
        logger.info(f"Initializing HNSW index with M={M}, ef_construction={ef_construction}, ef={ef}")
        self.M = M  # Max number of connections per node
        self.M0 = 2 * M  # Max number of connections for layer 0
        self.ef_construction = ef_construction  # Size of dynamic candidate list during construction
        self.ef = ef  # Size of dynamic candidate list during search
        self.enter_point = None
        self.nodes: List[HNSWNode] = []
        self.max_level = 0
        self.dim = None
        self.ml = 1 / np.log(M)
        self.max_elements = max_elements
        self.max_level_cap = int(np.log(max_elements) / np.log(M))

    def _distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        # Cosine distance
        return 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def _get_random_level(self) -> int:
        r = random.random()
        return min(int(-np.log(r) * self.ml), self.max_level_cap)

    def add(self, data: List[Dict[str, Any]]):
        logger.info("Adding data to HNSW index")
        for idx, item in enumerate(data):
            vector = np.array(item['vector'])
            if self.dim is None:
                self.dim = len(vector)
            level = self._get_random_level()
            node = HNSWNode(item, level)
            node_id = len(self.nodes)
            self.nodes.append(node)
            if level > self.max_level:
                self.max_level = level
                self.enter_point = node_id
            if self.enter_point is None:
                self.enter_point = node_id
                continue

            current_node_id = self.enter_point
            for l in range(self.max_level, -1, -1):
                current_distance = self._distance(vector, self.nodes[current_node_id].data['vector'])
                changed = True
                while changed:
                    changed = False
                    for neighbor_id in self.nodes[current_node_id].connections[l]:
                        distance = self._distance(vector, self.nodes[neighbor_id].data['vector'])
                        if distance < current_distance:
                            current_node_id = neighbor_id
                            current_distance = distance
                            changed = True
                if l <= level:
                    self._connect_node(node_id, current_node_id, l)
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} items")
        logger.success("All data added to HNSW index")

    def _select_neighbors(self, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        selected = []
        for distance, candidate_id in sorted(candidates):
            if len(selected) >= M:
                break
            keep = True
            for s in selected:
                if self._distance(self.nodes[candidate_id].data['vector'], self.nodes[s].data['vector']) < distance:
                    keep = False
                    break
            if keep:
                selected.append(candidate_id)
        return selected

    def _connect_node(self, node_id: int, entry_point_id: int, level: int):
        M = self.M0 if level == 0 else self.M
        candidates = self._search_layer(self.nodes[node_id].data['vector'], entry_point_id, self.ef_construction, level)
        neighbors = self._select_neighbors(candidates, M)
        self.nodes[node_id].connections[level].extend(neighbors)
        for neighbor_id in neighbors:
            self.nodes[neighbor_id].connections[level].append(node_id)
            if len(self.nodes[neighbor_id].connections[level]) > M:
                neighbor_connections = [(self._distance(self.nodes[neighbor_id].data['vector'], self.nodes[n].data['vector']), n) for n in self.nodes[neighbor_id].connections[level]]
                self.nodes[neighbor_id].connections[level] = self._select_neighbors(neighbor_connections, M)

    def _search_layer(self, query_vector: np.ndarray, entry_point_id: int, ef: int, level: int) -> List[Tuple[float, int]]:
        visited = set()
        candidates = []
        results = []

        distance = self._distance(query_vector, self.nodes[entry_point_id].data['vector'])
        heapq.heappush(candidates, (distance, entry_point_id))
        heapq.heappush(results, (-distance, entry_point_id))
        visited.add(entry_point_id)

        while candidates:
            current_distance, current_node_id = heapq.heappop(candidates)
            worst_distance = -results[0][0]
            if current_distance > worst_distance:
                break

            for neighbor_id in self.nodes[current_node_id].connections[level]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dist = self._distance(query_vector, self.nodes[neighbor_id].data['vector'])
                    if len(results) < ef or dist < -results[0][0]:
                        heapq.heappush(candidates, (dist, neighbor_id))
                        heapq.heappush(results, (-dist, neighbor_id))
                        if len(results) > ef:
                            heapq.heappop(results)
        final_results = [(-score, node_id) for score, node_id in results]
        return final_results

    def query(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, Dict[str, Any]]]:
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector dimension ({len(query_vector)}) does not match index dimension ({self.dim})")
        
        current_node_id = self.enter_point
        for level in range(self.max_level, -1, -1):
            current_distance = self._distance(query_vector, self.nodes[current_node_id].data['vector'])
            changed = True
            while changed:
                changed = False
                for neighbor_id in self.nodes[current_node_id].connections[level]:
                    distance = self._distance(query_vector, self.nodes[neighbor_id].data['vector'])
                    if distance < current_distance:
                        current_node_id = neighbor_id
                        current_distance = distance
                        changed = True
        result = self._search_layer(query_vector, current_node_id, max(self.ef, k), 0)
        result = [(1 - d, self.nodes[n].data) for d, n in result]
        result.sort(reverse=True)
        return result[:k]

if __name__ == "__main__":
    logger.add("hnsw_search.log", rotation="10 MB")
    # Load embeddings
    embeddings = load_embeddings("embeddings_small.jsonl")
    # Initialize HNSW index
    hnsw_index = HNSWIndex(M=16, ef_construction=200, ef=50, max_elements=len(embeddings))
    # Add embeddings to HNSW index
    hnsw_index.add(embeddings)
    # Load query embeddings
    query_embeddings = load_embeddings("query_embeddings_small.jsonl")
    # For each query, perform search
    for query in query_embeddings:
        query_vector = np.array(query['vector'])
        query_label = query['label']
        logger.info(f"Searching for similar items to query {query_label}")
        logger.info(f"{query['metadata']['content_preview']}")
        results = hnsw_index.query(query_vector, k=10)
        logger.info(f"Top results for query {query_label}:")
        for rank, (sim, item) in enumerate(results):
            logger.info(f"Rank {rank + 1}: Label={item['label']}, Similarity={sim:.4f}")
            logger.info(f"Content preview: {item['metadata']['content_preview']}")
        logger.info("---")
    logger.success("Search process completed")
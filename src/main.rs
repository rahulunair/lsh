//! This module implements a Locality-Sensitive Hashing (LSH) based similarity search system
//! optimized for transformer-based embeddings.

use bloom::{BloomFilter, ASMS};
use ndarray::Array1;
use ordered_float::OrderedFloat;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

/// Represents a similarity search result with a similarity score and an index.
#[derive(PartialEq, Eq)]
struct SimilarityResult(OrderedFloat<f64>, usize);

impl Ord for SimilarityResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0).reverse()
    }
}

impl PartialOrd for SimilarityResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Represents an item in the dataset with its label, vector representation, and content preview.
#[derive(Debug, Clone)]
struct Item {
    label: String,
    vector: Array1<f64>,
    content_preview: String,
}

/// The main LSH index structure for similarity search.
struct LSHIndex {
    items: Vec<f64>,
    item_metadata: Vec<(String, String)>,
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
    hash_functions: Vec<Array1<f64>>,
    hash_size: usize,
    num_hash_tables: usize,
    vector_dim: usize,
    bloom_filter: BloomFilter,
}

/// Represents a candidate for multi-probe LSH.
#[derive(Clone, Eq, PartialEq)]
struct ProbingCandidate {
    hash: u64,
    distance: u32,
    table_index: usize,
}

impl Ord for ProbingCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for ProbingCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// LSH implementation
impl LSHIndex {
    /// Creates a new LSH index from the given embeddings.
    fn new(embeddings: &[Item]) -> Self {
        let num_items = embeddings.len();
        let dim = embeddings[0].vector.len();
        let (num_hash_tables, hash_size) = calculate_lsh_params(num_items, dim);

        println!(
            "Initializing LSH index with {} hash tables and hash size {}",
            num_hash_tables, hash_size
        );

        let mut items = Vec::with_capacity(num_items * dim);
        let mut item_metadata = Vec::with_capacity(num_items);

        for item in embeddings {
            items.extend(item.vector.iter());
            item_metadata.push((item.label.clone(), item.content_preview.clone()));
        }

        let bloom_filter = BloomFilter::with_rate(0.01, num_items as u32);

        let mut lsh_index = LSHIndex {
            items,
            item_metadata,
            hash_tables: vec![HashMap::new(); num_hash_tables],
            hash_functions: Vec::new(),
            hash_size,
            num_hash_tables,
            vector_dim: dim,
            bloom_filter,
        };

        lsh_index.generate_angular_hash_functions();

        for i in 0..num_items {
            lsh_index.insert(i);
        }

        lsh_index
    }

    /// Generates angular hash functions for the LSH index.
    fn generate_angular_hash_functions(&mut self) {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        self.hash_functions = (0..self.num_hash_tables * self.hash_size)
            .map(|_| {
                Array1::from_vec(
                    (0..self.vector_dim)
                        .map(|_| normal.sample(&mut rng))
                        .collect(),
                )
            })
            .collect();
    }

    /// Inserts an item into the LSH index.
    fn insert(&mut self, idx: usize) {
        let vector = self.get_vector(idx);
        let hashes = self.angular_hash(vector);

        for (i, hash) in hashes.into_iter().enumerate() {
            self.hash_tables[i]
                .entry(hash)
                .or_insert_with(Vec::new)
                .push(idx);
        }

        self.bloom_filter.insert(&idx.to_be_bytes());
    }

    /// Retrieves the vector representation of an item by its index.
    fn get_vector(&self, idx: usize) -> &[f64] {
        let start = idx * self.vector_dim;
        let end = start + self.vector_dim;
        &self.items[start..end]
    }

    /// Computes the angular hash of a vector.
    fn angular_hash(&self, vector: &[f64]) -> Vec<u64> {
        (0..self.num_hash_tables)
            .map(|i| {
                let start = i * self.hash_size;
                let end = start + self.hash_size;
                let mut hash = 0u64;
                for (j, hash_func) in self.hash_functions[start..end].iter().enumerate() {
                    if vector
                        .iter()
                        .zip(hash_func.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                        > 0.0
                    {
                        hash |= 1 << j;
                    }
                }
                hash
            })
            .collect()
    }

    /// Performs a similarity query on the LSH index.
    fn query(&self, vector: &[f64], k: usize) -> Vec<(f64, usize)> {
        self.adaptive_query(vector, k, k * 10)
    }

    /// Performs an adaptive similarity query on the LSH index.
    fn adaptive_query(&self, vector: &[f64], k: usize, max_probes: usize) -> Vec<(f64, usize)> {
        let mut candidates = HashSet::new();
        let mut results = BinaryHeap::new();
        let hashes = self.angular_hash(vector);

        // Initialize the priority queue with the original hashes
        let mut probing_queue = BinaryHeap::new();
        for (i, &hash) in hashes.iter().enumerate() {
            probing_queue.push(ProbingCandidate {
                hash,
                distance: 0,
                table_index: i,
            });
        }

        let mut probes = 0;
        while let Some(candidate) = probing_queue.pop() {
            if probes >= max_probes {
                break;
            }
            probes += 1;

            if let Some(indices) = self.hash_tables[candidate.table_index].get(&candidate.hash) {
                for &idx in indices {
                    if candidates.insert(idx) && self.bloom_filter.contains(&idx.to_be_bytes()) {
                        let similarity = dot_product(vector, self.get_vector(idx));
                        results.push(SimilarityResult(OrderedFloat(similarity), idx));
                    }
                }
            }

            // Generate new candidates by flipping each bit in the hash code
            if candidate.distance < self.hash_size as u32 {
                for i in 0..self.hash_size {
                    let new_hash = candidate.hash ^ (1 << i);
                    let new_distance = candidate.distance + 1;
                    probing_queue.push(ProbingCandidate {
                        hash: new_hash,
                        distance: new_distance,
                        table_index: candidate.table_index,
                    });
                }
            }

            // Early termination based on results
            if results.len() >= k {
                let min_similarity = results.peek().unwrap().0;
                if min_similarity > OrderedFloat(0.9) {
                    // Adjust this threshold based on your data
                    break;
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .take(k)
            .map(|r| (r.0.into_inner(), r.1))
            .collect()
    }
}

/// Utility functions and main block
/// Computes the dot product between two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Loads vector embeddings from a JSON file.
fn load_vectors(file_path: &str) -> io::Result<Vec<Item>> {
    println!("Loading embeddings from {}", file_path);
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let json: Value = serde_json::from_str(&line)?;

        let label = json["label"].as_str().unwrap().to_string();
        let vector: Vec<f64> = json["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let content_preview = json["metadata"]["content_preview"]
            .as_str()
            .unwrap()
            .to_string();

        items.push(Item {
            label,
            vector: Array1::from(vector),
            content_preview,
        });
    }

    println!("Loaded {} embeddings from {}", items.len(), file_path);
    Ok(items)
}

/// Calculates optimal LSH parameters based on the dataset size and dimensionality.
fn calculate_lsh_params(num_items: usize, dim: usize) -> (usize, usize) {
    let num_hash_tables = (num_items as f64).powf(0.4).ceil() as usize;
    let hash_size = (dim as f64).log2().ceil() as usize;

    let num_hash_tables = num_hash_tables.clamp(8, 30);
    let hash_size = hash_size.clamp(12, 20);

    (num_hash_tables, hash_size)
}

fn main() -> io::Result<()> {
    let embeddings = load_vectors("embeddings_small.jsonl")?;

    println!("Building LSH index...");
    let lsh_index = LSHIndex::new(&embeddings);
    println!("LSH index built.");

    let query_embeddings = load_vectors("query_embeddings_small.jsonl")?;
    println!("Loaded {} query embeddings", query_embeddings.len());

    println!("Performing queries...");
    for query in &query_embeddings {
        println!("Searching for similar items to query {}", query.label);
        println!("Query content preview: {}", query.content_preview);

        let results = lsh_index.query(query.vector.as_slice().unwrap(), 10);
        println!("Top results for query {}:", query.label);

        if results.is_empty() {
            println!("No similar items found.");
        } else {
            for (rank, (similarity, idx)) in results.iter().enumerate() {
                let (label, content_preview) = &lsh_index.item_metadata[*idx];
                println!(
                    "Rank {}: Label={}, Similarity={:.4}",
                    rank + 1,
                    label,
                    similarity
                );
                println!("Content preview: {}", content_preview);
            }
        }
        println!("---");
    }

    println!("All queries completed.");
    Ok(())
}

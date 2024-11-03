# ⚡ seekvec

A lightning-fast similarity search engine for embedding vectors. Build efficient search capabilities without heavyweight vector databases. Perfect for when you want something light and fast! 🏃‍♂️

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/rahulunair/seekvec?style=social)](https://github.com/rahulunair/seekvec/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

</div>

## 🎯 Purpose

Ever felt vector databases were overkill for your embedding search needs? You're not alone! This project shows how to implement efficient similarity search when:
- You have up to millions of embeddings 📚
- Need blazing-fast approximate nearest neighbor search ⚡
- Want a lightweight solution (no heavy dependencies!) 🪶
- Require high performance with minimal resource usage 💪

## ✨ Features

### 🛠️ Multiple Implementations
- Rust LSH implementation (optimized for speed) 🦀
- Python LSH implementation (easy to use) 🐍
- Python HNSW implementation (for comparison) 🔄
- Embedding generation utilities 🎮

### 🎨 Smart Optimizations
- Bloom filters for super-efficient candidate filtering 🌸
- Multi-probe LSH for better recall without the overhead 🎯
- Adaptive early termination (because why wait?) ⏱️
- Angular LSH optimized for cosine similarity 📐
- Smart data structures for zippy retrieval 🏃‍♀️

## 📦 Installation

### Rust Implementation
```bash
# Clone the repository
git clone https://github.com/rahulunair/seekvec
cd seekvec

# Build the Rust implementation
cargo build --release
```

### Python Implementation
```bash
# Install required packages
pip install -r requirements.txt
```

## 🎮 Usage

### 1. Generate Your Embeddings
```bash
# Generate sample embeddings
python script/gen_embeddings.py
```

This will:
- 📚 Load sample data from AG News dataset
- 🤖 Generate embeddings using SentenceTransformer
- 💾 Create both main and query embedding files

### 2. Run the Search

#### 🦀 Using Rust LSH:
```bash
cargo run --release
```

#### 🐍 Using Python LSH:
```bash
python script/lsh_search.py
```

#### 🔄 Using Python HNSW:
```bash
python script/hnsw_search.py
```

## 🎯 Implementation Details

### 🎨 LSH Implementation
Our LSH implementation uses clever angular hashing optimized for cosine similarity:
- 🎲 Smart random projection generation
- 🎯 Multi-probe LSH with early stopping
- 🌸 Bloom filters for speed
- ⚡ SIMD-optimized vector operations (Rust)

### 🌳 HNSW Implementation
The HNSW implementation provides:
- 📊 Hierarchical graph structure
- 🏗️ Efficient graph construction
- 📈 Logarithmic search complexity

## 📊 Performance

Tested on 1M 768-dimensional embeddings:
- ⚡ Query time: ~10-50ms
- 💾 Memory usage: ~2-4GB
- 🏗️ Build time: ~5-10 minutes
- 🎯 Recall@10: ~0.8-0.9

## 🔧 Configuration

Key parameters (all auto-tuned but configurable):
```python
# LSH Parameters
num_hash_tables = 10  # More tables = better recall, more memory
hash_size = 16        # Larger size = better precision, slower search

# HNSW Parameters
M = 16               # More connections = better recall, more memory
ef_construction = 200 # Higher ef = better index quality, slower build
```

## 🤝 Contributing

Want to make this even better? Contributions are welcome! Areas for improvement:
- 📊 Additional similarity metrics
- ⚡ More optimization techniques
- 📈 Benchmarking tools
- 📚 Documentation improvements

## 📄 License

MIT License - Go wild! 🎉

## 👏 Acknowledgments

- 📚 Inspired by various LSH and HNSW papers
- 🦀 Built with love using Rust and Python
- 🤗 Tested with Sentence Transformers
- 🌟 Special thanks to the open-source community

## 📬 Contact

- GitHub: [@rahulunair](https://github.com/rahulunair)
- Repository: [seekvec](https://github.com/rahulunair/seekvec)

---
<div align="center">
⚡ Made with love for the ML community ⚡
</div>
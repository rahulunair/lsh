# 🐍 LSH Search Scripts

Collection of Python scripts for fast embedding similarity search using LSH and HNSW algorithms (used to build the rust version and to do quick iterations and testing).

## 📁 Files

- `gen_embeddings.py` - Generate embeddings from text data 🎯
- `lsh_search.py` - LSH-based similarity search implementation 🔍
- `hnsw_search.py` - HNSW-based similarity search implementation 🌳

## 🛠️ Setup

### Using uv (Recommended)
```bash
uv pip install -r requirements.txt
```

### Using rye
```bash
rye sync
```

### Using pip
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### 1. Generate Embeddings
```bash
python gen_embeddings.py \
    --dataset "ag_news" \
    --split "train" \
    --text-column "text" \
    --num-samples 1000 \
    --output "embeddings_small.jsonl"
```

This will:
- Load the AG News dataset 📚
- Generate embeddings using SentenceTransformer 🤖
- Save embeddings in JSONL format 💾

### 2. Run LSH Search
```bash
python lsh_search.py \
    --embeddings "embeddings_small.jsonl" \
    --num-tables 10 \
    --hash-size 16
```

### 3. Run HNSW Search
```bash
python hnsw_search.py \
    --embeddings "embeddings_small.jsonl" \
    --M 16 \
    --ef-construction 200
```

## 📊 Data Format

### Input Embeddings Format (JSONL)
```json
{
    "label": "doc_00001",
    "vector": [0.1, 0.2, ...],
    "metadata": {
        "file_path": "/data/documents/00/document_00001.txt",
        "content_preview": "Sample text content...",
        "chunk_index": 0
    }
}
```

## 🔧 Configuration

### LSH Parameters
```python
num_hash_tables = 10  # Number of hash tables
hash_size = 16        # Size of each hash
```

### HNSW Parameters
```python
M = 16                # Max number of connections
ef_construction = 200 # Size of dynamic candidate list
ef = 50              # Size of dynamic candidate list during search
```

## 📝 Logging

All scripts use `loguru` for logging:
- Logs are rotated at 10MB
- Default log files:
  - `lsh_search.log`
  - `hnsw_search.log`

## 🤔 Common Issues

1. **Memory Usage**
   - For large datasets, adjust batch size in `gen_embeddings.py`
   - Use memory-efficient numpy dtypes

2. **Performance**
   - Increase `num_hash_tables` for better recall
   - Adjust `ef` in HNSW for speed/accuracy trade-off

## 🔗 Dependencies

See `requirements.txt` for full list:
- numpy
- loguru
- sentence-transformers
- datasets
- torch (optional)

## 📚 References

- [LSH Paper](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/Gionis.pdf)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
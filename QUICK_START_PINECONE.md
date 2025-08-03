# 🚀 Quick Start: Nelson Pediatrics + Pinecone

Get your medical knowledge base running in 5 minutes!

## 🎯 Option 1: Automated Setup (Recommended)

```bash
# 1. Set up credentials interactively
./setup_credentials.sh

# 2. Run complete setup (creates index + uploads embeddings)
source .env
python setup_pinecone_complete.py

# 3. Test interactive search
python demo_pinecone_search.py interactive
```

## 🎯 Option 2: Manual Setup

```bash
# 1. Set credentials manually
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_ENVIRONMENT="us-east-1-aws"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Upload to Pinecone
python upload_to_pinecone.py

# 4. Test search
python demo_pinecone_search.py
```

## 📋 What You Need

1. **Pinecone Account**: Sign up at [pinecone.io](https://app.pinecone.io/)
2. **API Key**: Copy from your Pinecone dashboard
3. **Processed Data**: Already generated (`nelson_pediatrics_processed.jsonl`)

## 🎉 Expected Results

- **📊 Index Created**: `nelson-pediatrics` with 384 dimensions
- **📤 Vectors Uploaded**: ~15,339 medical text chunks
- **🔍 Search Ready**: Instant similarity search across all content
- **💰 Cost**: Free tier (100K vectors, you only need ~15K)

## 🔍 Example Searches

```python
# Basic search
results = search_system.search("asthma treatment in children", top_k=5)

# Topic-specific search  
results = search_system.search_by_topic("heart murmur", "The Cardiovascular System")

# Available topics: Respiratory, Cardiovascular, Nervous System, etc.
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| `PINECONE_API_KEY not set` | Run `./setup_credentials.sh` |
| `Index already exists` | Normal - script will use existing index |
| `Rate limit exceeded` | Script includes automatic batching |
| `SSL/Connection error` | Check your internet connection |

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `setup_pinecone_complete.py` | **🎯 Main setup script** - Does everything |
| `setup_credentials.sh` | **🔑 Credential setup** - Interactive setup |
| `upload_to_pinecone.py` | **📤 Upload script** - Manual upload |
| `demo_pinecone_search.py` | **🔍 Search demo** - Test functionality |

---

**Ready to search 23 medical textbooks instantly! 🏥⚡**


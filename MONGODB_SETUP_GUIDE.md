# Nelson Pediatrics MongoDB Setup Guide

This guide will help you process your text files, generate embeddings, and upload them to MongoDB.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Processing Script

```bash
python process_and_upload_to_mongodb.py
```

### 3. Test the Upload

```bash
python test_mongodb_connection.py
```

## 📋 What the Script Does

### Text Processing
- ✅ Reads all `.txt` files from the `txt_files/` directory
- ✅ Cleans and normalizes the text content
- ✅ Splits text into overlapping chunks (1000 chars with 200 char overlap)
- ✅ Preserves sentence boundaries to maintain context

### Embedding Generation
- 🤖 Uses Hugging Face `sentence-transformers` model: `all-MiniLM-L6-v2`
- 🧠 Generates 384-dimensional embeddings for each text chunk
- ⚡ Fast and efficient processing with progress bars

### MongoDB Upload
- 📊 Creates indexes for efficient querying
- 🔄 Uploads in batches for better performance
- 📈 Provides detailed statistics after upload

## 📊 Data Structure

Each document in MongoDB will have this structure:

```json
{
  "_id": "ObjectId(...)",
  "text": "The actual text content of the chunk...",
  "source_file": "The Respiratory System.txt",
  "topic": "The Respiratory System",
  "chunk_number": 1,
  "character_count": 987,
  "embedding": [0.123, -0.456, 0.789, ...],  // 384-dimensional vector
  "created_at": "2024-08-01T17:25:00Z",
  "metadata": {
    "file_size": 1412925,
    "total_chunks": 45,
    "embedding_model": 384
  }
}
```

## 🔍 Querying Examples

### Text Search
```python
# Search for specific terms
results = collection.find({"$text": {"$search": "asthma treatment"}})

# Search by topic
results = collection.find({"topic": "The Respiratory System"})

# Search by source file
results = collection.find({"source_file": "Digestive system.txt"})
```

### Vector Similarity Search
For vector similarity search, you'll need to:
1. Generate embedding for your query text
2. Use MongoDB's vector search capabilities (Atlas Search)

## 🛠️ Configuration Options

### Embedding Models
You can change the embedding model in the script:

```python
# Fast and efficient (default)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions

# Higher quality but slower
EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768 dimensions

# Balanced option
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # 384 dimensions
```

### Chunk Settings
Adjust chunking parameters:

```python
self.chunk_size = 1000  # Characters per chunk
self.overlap = 200      # Character overlap between chunks
```

## 📈 Expected Results

Based on your text files, you should expect:
- **~23 source files** processed
- **Several thousand text chunks** generated
- **Full-text search** capabilities
- **Vector similarity search** ready embeddings

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**: If you run out of memory, try:
   - Using a smaller embedding model
   - Processing files one at a time
   - Reducing batch size

2. **Connection Issues**: 
   - Verify your MongoDB connection string
   - Check network connectivity
   - Ensure MongoDB Atlas allows your IP

3. **Encoding Issues**:
   - The script handles both UTF-8 and Latin-1 encodings
   - Check your text files for special characters

## 🔒 Security Note

**Important**: After uploading your data, remember to:
1. Change your MongoDB password
2. Use environment variables for connection strings
3. Enable IP whitelisting in MongoDB Atlas

## 📞 Next Steps

After successful upload, you can:
1. Build a search API using the uploaded data
2. Create a web interface for querying
3. Implement vector similarity search
4. Migrate to PostgreSQL with pgvector if needed

Happy coding! 🎉


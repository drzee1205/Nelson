#!/usr/bin/env python3
"""
Nelson Pediatrics Text Processing Script (Local Processing)

This script:
1. Reads text files from txt_files directory
2. Chunks the text into manageable pieces
3. Generates embeddings using Hugging Face sentence-transformers
4. Saves processed data to JSON files for later MongoDB import
"""

import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

# Third-party imports
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NelsonTextProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the text processor with embedding model
        
        Args:
            model_name: Hugging Face model name for embeddings
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = 1000  # Characters per chunk
        self.overlap = 200      # Character overlap between chunks
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\%\/]', '', text)
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for better context preservation
        
        Args:
            text: Input text to chunk
            source_file: Source filename
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        text = self.clean_text(text)
        
        # Split by sentences first to avoid breaking mid-sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_num = 1
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source_file': source_file,
                    'chunk_number': chunk_num,
                    'character_count': len(current_chunk.strip())
                })
                
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_num += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source_file': source_file,
                'chunk_number': chunk_num,
                'character_count': len(current_chunk.strip())
            })
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single text file"""
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Extract topic from filename
        topic = file_path.stem.replace('_', ' ').title()
        
        # Chunk the text
        chunks = self.chunk_text(content, file_path.name)
        
        # Generate embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings and metadata to chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk = {
                'text': chunk['text'],
                'source_file': chunk['source_file'],
                'topic': topic,
                'chunk_number': chunk['chunk_number'],
                'character_count': chunk['character_count'],
                'embedding': embeddings[i].tolist(),  # Convert numpy array to list
                'created_at': datetime.utcnow().isoformat(),
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'total_chunks': len(chunks),
                    'embedding_dimension': len(embeddings[i]),
                    'embedding_model': self.model.get_sentence_embedding_dimension()
                }
            }
            processed_chunks.append(processed_chunk)
        
        return processed_chunks

def save_to_json(data: List[Dict[str, Any]], output_file: str):
    """Save processed data to JSON file"""
    logger.info(f"Saving {len(data)} documents to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Data saved successfully to {output_file}")

def save_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    """Save processed data to JSONL file (one JSON object per line)"""
    logger.info(f"Saving {len(data)} documents to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    logger.info(f"Data saved successfully to {output_file}")

def create_mongodb_import_script(jsonl_file: str, connection_string: str):
    """Create a MongoDB import script"""
    script_content = f'''#!/bin/bash
# MongoDB Import Script for Nelson Pediatrics Data

echo "🚀 Starting MongoDB import..."

# Import using mongoimport (requires MongoDB tools)
mongoimport --uri "{connection_string}" \\
    --collection nelson_book_content \\
    --file {jsonl_file} \\
    --jsonArray

echo "✅ Import completed!"

# Alternative: Using Python script
echo "📝 Alternative: Run the following Python script:"
echo "python import_to_mongodb.py"
'''
    
    with open('import_to_mongodb.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('import_to_mongodb.sh', 0o755)
    logger.info("Created import_to_mongodb.sh script")

def create_python_import_script(jsonl_file: str, connection_string: str):
    """Create a Python script for MongoDB import"""
    script_content = f'''#!/usr/bin/env python3
"""
MongoDB Import Script for Nelson Pediatrics Data
"""

import json
import pymongo
from tqdm import tqdm

def import_to_mongodb():
    """Import JSONL data to MongoDB"""
    
    # Connection settings
    connection_string = "{connection_string}"
    
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(
            connection_string,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000
        )
        
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
        
        db = client.nelson_pediatrics
        collection = db.nelson_book_content
        
        # Clear existing data (optional)
        # collection.delete_many({{}})
        
        # Read and import data
        documents = []
        with open('{jsonl_file}', 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading data"):
                if line.strip():
                    documents.append(json.loads(line))
        
        print(f"📊 Importing {{len(documents)}} documents...")
        
        # Insert in batches
        batch_size = 100
        for i in tqdm(range(0, len(documents), batch_size), desc="Importing"):
            batch = documents[i:i + batch_size]
            collection.insert_many(batch, ordered=False)
        
        # Create indexes
        print("🔍 Creating indexes...")
        collection.create_index([("text", "text")])
        collection.create_index("source_file")
        collection.create_index("topic")
        collection.create_index([("topic", 1), ("source_file", 1)])
        
        # Print statistics
        total_docs = collection.count_documents({{}})
        unique_sources = len(collection.distinct('source_file'))
        unique_topics = len(collection.distinct('topic'))
        
        print("\\n" + "="*50)
        print("🎉 IMPORT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Total documents: {{total_docs}}")
        print(f"📚 Source files: {{unique_sources}}")
        print(f"🏷️  Topics: {{unique_topics}}")
        
    except Exception as e:
        print(f"❌ Error: {{e}}")
        return False
    
    return True

if __name__ == "__main__":
    import_to_mongodb()
'''
    
    with open('import_to_mongodb.py', 'w') as f:
        f.write(script_content)
    
    logger.info("Created import_to_mongodb.py script")

def main():
    """Main execution function"""
    # Configuration
    TXT_FILES_DIR = Path("txt_files")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient model
    OUTPUT_JSON = "nelson_pediatrics_processed.json"
    OUTPUT_JSONL = "nelson_pediatrics_processed.jsonl"
    MONGODB_CONNECTION = "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson"
    
    logger.info("Starting Nelson Pediatrics text processing...")
    
    # Initialize processor
    processor = NelsonTextProcessor(model_name=EMBEDDING_MODEL)
    
    # Process all text files
    all_chunks = []
    txt_files = list(TXT_FILES_DIR.glob("*.txt"))
    
    logger.info(f"Found {len(txt_files)} text files to process")
    
    for file_path in txt_files:
        try:
            chunks = processor.process_file(file_path)
            all_chunks.extend(chunks)
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    
    if all_chunks:
        # Save to JSON files
        save_to_json(all_chunks, OUTPUT_JSON)
        save_to_jsonl(all_chunks, OUTPUT_JSONL)
        
        # Create import scripts
        create_mongodb_import_script(OUTPUT_JSONL, MONGODB_CONNECTION)
        create_python_import_script(OUTPUT_JSONL, MONGODB_CONNECTION)
        
        # Print statistics
        total_chunks = len(all_chunks)
        unique_sources = len(set(chunk['source_file'] for chunk in all_chunks))
        unique_topics = len(set(chunk['topic'] for chunk in all_chunks))
        total_chars = sum(chunk['character_count'] for chunk in all_chunks)
        
        print("\\n" + "="*60)
        print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Total text chunks: {total_chunks}")
        print(f"📚 Source files processed: {unique_sources}")
        print(f"🏷️  Topics covered: {unique_topics}")
        print(f"📝 Total characters: {total_chars:,}")
        print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
        print(f"🧠 Embedding dimension: {processor.model.get_sentence_embedding_dimension()}")
        
        print("\\n📁 Output files created:")
        print(f"   • {OUTPUT_JSON} - Full JSON format")
        print(f"   • {OUTPUT_JSONL} - JSONL format (recommended for MongoDB)")
        print(f"   • import_to_mongodb.sh - Bash import script")
        print(f"   • import_to_mongodb.py - Python import script")
        
        print("\\n🚀 Next steps:")
        print("1. Fix your MongoDB connection (check IP whitelist, credentials)")
        print("2. Run: python import_to_mongodb.py")
        print("3. Or use: ./import_to_mongodb.sh")
        
        # Show sample data
        print("\\n📄 Sample processed document:")
        sample = all_chunks[0].copy()
        sample['embedding'] = f"[{len(sample['embedding'])} dimensions]"  # Don't print full embedding
        print(json.dumps(sample, indent=2, default=str))
        
    else:
        logger.error("No chunks were generated. Please check your text files.")

if __name__ == "__main__":
    main()


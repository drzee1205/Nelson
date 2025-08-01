#!/usr/bin/env python3
"""
Nelson Pediatrics Text Processing and MongoDB Upload Script

This script:
1. Reads text files from txt_files directory
2. Chunks the text into manageable pieces
3. Generates embeddings using Hugging Face sentence-transformers
4. Uploads to MongoDB with embeddings
"""

import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

# Third-party imports
import pymongo
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
                'created_at': datetime.utcnow(),
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'total_chunks': len(chunks),
                    'embedding_model': self.model.get_sentence_embedding_dimension()
                }
            }
            processed_chunks.append(processed_chunk)
        
        return processed_chunks

class MongoDBUploader:
    def __init__(self, connection_string: str, database_name: str = "nelson_pediatrics"):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection string
            database_name: Database name to use
        """
        # Configure SSL settings for MongoDB Atlas
        self.client = pymongo.MongoClient(
            connection_string,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Test the connection
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful!")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
            
        self.db = self.client[database_name]
        self.collection = self.db.nelson_book_content
        
        logger.info(f"Connected to MongoDB database: {database_name}")
        
    def create_indexes(self):
        """Create necessary indexes for efficient querying"""
        logger.info("Creating database indexes...")
        
        # Text search index
        self.collection.create_index([("text", "text")])
        
        # Source file index
        self.collection.create_index("source_file")
        
        # Topic index
        self.collection.create_index("topic")
        
        # Compound index for efficient filtering
        self.collection.create_index([("topic", 1), ("source_file", 1)])
        
        logger.info("Indexes created successfully")
    
    def upload_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upload chunks to MongoDB in batches
        
        Args:
            chunks: List of processed text chunks
            batch_size: Number of documents to insert per batch
        """
        logger.info(f"Uploading {len(chunks)} chunks to MongoDB...")
        
        # Insert in batches for better performance
        for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading batches"):
            batch = chunks[i:i + batch_size]
            try:
                self.collection.insert_many(batch, ordered=False)
            except pymongo.errors.BulkWriteError as e:
                logger.warning(f"Some documents in batch {i//batch_size + 1} failed to insert: {e}")
        
        logger.info("Upload completed successfully")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the uploaded data"""
        stats = {
            'total_documents': self.collection.count_documents({}),
            'unique_sources': len(self.collection.distinct('source_file')),
            'unique_topics': len(self.collection.distinct('topic')),
            'sample_document': self.collection.find_one({}, {'embedding': 0})  # Exclude embedding for readability
        }
        return stats

def main():
    """Main execution function"""
    # Configuration
    MONGODB_CONNECTION = "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson"
    TXT_FILES_DIR = Path("txt_files")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient model
    
    # Alternative models you can try:
    # "all-mpnet-base-v2"  # Higher quality but slower
    # "sentence-transformers/all-MiniLM-L12-v2"  # Balanced option
    
    logger.info("Starting Nelson Pediatrics text processing and upload...")
    
    # Initialize processor and uploader
    processor = NelsonTextProcessor(model_name=EMBEDDING_MODEL)
    uploader = MongoDBUploader(MONGODB_CONNECTION)
    
    # Create indexes
    uploader.create_indexes()
    
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
    
    # Upload to MongoDB
    if all_chunks:
        uploader.upload_chunks(all_chunks)
        
        # Print statistics
        stats = uploader.get_collection_stats()
        logger.info("Upload Statistics:")
        logger.info(f"  Total documents: {stats['total_documents']}")
        logger.info(f"  Unique sources: {stats['unique_sources']}")
        logger.info(f"  Unique topics: {stats['unique_topics']}")
        
        print("\n" + "="*50)
        print("UPLOAD COMPLETED SUCCESSFULLY! 🎉")
        print("="*50)
        print(f"📊 Total documents uploaded: {stats['total_documents']}")
        print(f"📚 Source files processed: {stats['unique_sources']}")
        print(f"🏷️  Topics covered: {stats['unique_topics']}")
        print(f"🤖 Embedding model used: {EMBEDDING_MODEL}")
        print("\nSample document structure:")
        print(json.dumps(stats['sample_document'], indent=2, default=str))
    else:
        logger.error("No chunks were generated. Please check your text files.")

if __name__ == "__main__":
    main()

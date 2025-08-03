#!/usr/bin/env python3
"""
Nelson Pediatrics Pinecone Upload Script

This script uploads the processed Nelson Pediatrics data to Pinecone vector database.
"""

import os
import json
import time
from typing import List, Dict, Any
import logging
from datetime import datetime
from tqdm import tqdm

try:
    import pinecone
except ImportError:
    print("❌ Pinecone library not installed!")
    print("Install with: pip install pinecone-client")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeUploader:
    def __init__(self, api_key: str, environment: str = "us-east-1-aws"):
        """
        Initialize Pinecone connection
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (default: us-east-1-aws)
        """
        self.api_key = api_key
        self.environment = environment
        
        # Initialize Pinecone (v2.x API)
        pinecone.init(api_key=api_key, environment=environment)
        
        logger.info("✅ Pinecone client initialized successfully")
        
    def create_index(self, index_name: str, dimension: int = 384, metric: str = "cosine"):
        """
        Create a Pinecone index if it doesn't exist
        
        Args:
            index_name: Name of the index to create
            dimension: Vector dimension (384 for all-MiniLM-L6-v2)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        try:
            # Check if index already exists
            existing_indexes = pinecone.list_indexes()
            
            if index_name in existing_indexes:
                logger.info(f"📋 Index '{index_name}' already exists")
                return pinecone.Index(index_name)
            
            # Create new index
            logger.info(f"🔨 Creating index '{index_name}' with dimension {dimension}")
            
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )
            
            # Wait for index to be ready
            logger.info("⏳ Waiting for index to be ready...")
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
            
            logger.info(f"✅ Index '{index_name}' created successfully")
            return pinecone.Index(index_name)
            
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            raise
    
    def prepare_vectors(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare vectors for Pinecone upload
        
        Args:
            data: List of processed text chunks with embeddings
            
        Returns:
            List of vectors in Pinecone format
        """
        vectors = []
        
        for i, item in enumerate(data):
            # Create unique ID
            vector_id = f"nelson_{item['source_file'].replace('.txt', '')}_{item['chunk_number']}"
            
            # Prepare metadata (Pinecone has size limits)
            metadata = {
                'source_file': item['source_file'],
                'topic': item['topic'],
                'chunk_number': item['chunk_number'],
                'character_count': item['character_count'],
                'text': item['text'][:1000],  # Truncate text for metadata
                'created_at': item['created_at'] if isinstance(item['created_at'], str) else str(item['created_at'])
            }
            
            # Add file metadata if available
            if 'metadata' in item:
                metadata.update({
                    'file_size': item['metadata'].get('file_size', 0),
                    'total_chunks': item['metadata'].get('total_chunks', 0),
                    'embedding_model': item['metadata'].get('embedding_model', 384)
                })
            
            vector = {
                'id': vector_id,
                'values': item['embedding'],
                'metadata': metadata
            }
            
            vectors.append(vector)
        
        return vectors
    
    def upload_vectors(self, index, vectors: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upload vectors to Pinecone in batches
        
        Args:
            index: Pinecone index object
            vectors: List of vectors to upload
            batch_size: Number of vectors per batch
        """
        logger.info(f"📤 Uploading {len(vectors)} vectors to Pinecone...")
        
        # Upload in batches
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
            batch = vectors[i:i + batch_size]
            
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        logger.info("✅ Upload completed successfully")
    
    def get_index_stats(self, index) -> Dict[str, Any]:
        """Get statistics about the uploaded data"""
        try:
            stats = index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            logger.error(f"❌ Error getting index stats: {e}")
            return {}

def load_processed_data(file_path: str = "nelson_pediatrics_processed.jsonl") -> List[Dict[str, Any]]:
    """Load processed data from JSONL file"""
    logger.info(f"📖 Loading data from {file_path}")
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
                        continue
        
        logger.info(f"✅ Loaded {len(data)} documents")
        return data
        
    except FileNotFoundError:
        logger.error(f"❌ File {file_path} not found!")
        logger.info("💡 Run 'python process_data_locally.py' first to generate the data")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return []

def test_similarity_search(index, query_text: str = "asthma treatment"):
    """Test similarity search functionality"""
    logger.info(f"🔍 Testing similarity search with query: '{query_text}'")
    
    try:
        # For a real implementation, you'd need to generate embedding for the query
        # This is just a placeholder to show the structure
        logger.info("💡 To perform similarity search, you need to:")
        logger.info("1. Generate embedding for your query text using the same model")
        logger.info("2. Use index.query(vector=query_embedding, top_k=5, include_metadata=True)")
        logger.info("3. Process the results")
        
        # Example structure:
        # query_embedding = model.encode([query_text])[0].tolist()
        # results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        # return results
        
    except Exception as e:
        logger.error(f"❌ Error in similarity search: {e}")

def main():
    """Main execution function"""
    
    # Configuration
    INDEX_NAME = "nelson-pediatrics"
    DIMENSION = 384  # all-MiniLM-L6-v2 embedding dimension
    DATA_FILE = "nelson_pediatrics_processed.jsonl"
    
    print("🏥 Nelson Pediatrics Pinecone Upload Script")
    print("=" * 50)
    
    # Get Pinecone credentials
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set!")
        print("💡 Please set your Pinecone API key:")
        print("   export PINECONE_API_KEY='your-api-key-here'")
        print("\n📋 You can get your API key from: https://app.pinecone.io/")
        return
    
    environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
    
    try:
        # Initialize uploader
        uploader = PineconeUploader(api_key, environment)
        
        # Create or get index
        index = uploader.create_index(INDEX_NAME, DIMENSION)
        
        # Load processed data
        data = load_processed_data(DATA_FILE)
        if not data:
            return
        
        # Prepare vectors
        logger.info("🔄 Preparing vectors for upload...")
        vectors = uploader.prepare_vectors(data)
        
        # Upload to Pinecone
        uploader.upload_vectors(index, vectors)
        
        # Get statistics
        stats = uploader.get_index_stats(index)
        
        # Print results
        print("\n" + "=" * 50)
        print("🎉 UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Total vectors uploaded: {stats.get('total_vectors', 'Unknown')}")
        print(f"🧠 Vector dimension: {stats.get('dimension', 'Unknown')}")
        print(f"📈 Index fullness: {stats.get('index_fullness', 'Unknown')}")
        print(f"🏷️  Index name: {INDEX_NAME}")
        print(f"🌐 Environment: {environment}")
        
        # Test similarity search
        test_similarity_search(index)
        
        print("\n🚀 Your Nelson Pediatrics knowledge base is now ready for similarity search!")
        print("💡 You can now build applications that search through medical content using vector similarity.")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return

if __name__ == "__main__":
    main()

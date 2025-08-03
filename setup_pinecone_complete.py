#!/usr/bin/env python3
"""
Complete Pinecone Setup Script for Nelson Pediatrics

This script will:
1. Check for Pinecone credentials
2. Create the Pinecone index
3. Upload all embeddings
4. Test the search functionality
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
import logging
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import pinecone
        import sentence_transformers
        logger.info("✅ All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Install with: pip install -r requirements.txt")
        return False

def check_credentials():
    """Check for Pinecone credentials"""
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
    
    if not api_key:
        print("\n" + "="*60)
        print("🔑 PINECONE CREDENTIALS REQUIRED")
        print("="*60)
        print("❌ PINECONE_API_KEY environment variable not set!")
        print("\n📋 To get your Pinecone API key:")
        print("1. Go to https://app.pinecone.io/")
        print("2. Sign up or log in to your account")
        print("3. Create a new project (if needed)")
        print("4. Copy your API key from the dashboard")
        print("\n💡 Then set your credentials:")
        print("   export PINECONE_API_KEY='your-api-key-here'")
        print("   export PINECONE_ENVIRONMENT='us-east-1-aws'  # Optional")
        print("\n🔄 After setting credentials, run this script again:")
        print("   python setup_pinecone_complete.py")
        print("="*60)
        return None, None
    
    logger.info("✅ Pinecone credentials found")
    logger.info(f"🌐 Environment: {environment}")
    return api_key, environment

def check_processed_data():
    """Check if processed data exists"""
    data_file = "nelson_pediatrics_processed.jsonl"
    
    if not os.path.exists(data_file):
        logger.error(f"❌ {data_file} not found!")
        logger.info("💡 Run 'python process_data_locally.py' first to generate the data")
        return False
    
    # Check file size
    file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
    logger.info(f"✅ Found {data_file} ({file_size:.1f} MB)")
    
    # Count lines
    with open(data_file, 'r') as f:
        line_count = sum(1 for line in f if line.strip())
    
    logger.info(f"📊 Contains {line_count:,} processed documents")
    return True

def setup_pinecone_index(api_key: str, environment: str):
    """Set up Pinecone index"""
    try:
        import pinecone
        
        # Initialize Pinecone
        logger.info("🔌 Connecting to Pinecone...")
        pinecone.init(api_key=api_key, environment=environment)
        
        # Index configuration
        index_name = "nelson-pediatrics"
        dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        metric = "cosine"
        
        # Check if index exists
        existing_indexes = pinecone.list_indexes()
        
        if index_name in existing_indexes:
            logger.info(f"📋 Index '{index_name}' already exists")
            index = pinecone.Index(index_name)
        else:
            # Create new index
            logger.info(f"🔨 Creating index '{index_name}' with dimension {dimension}")
            
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )
            
            # Wait for index to be ready
            logger.info("⏳ Waiting for index to be ready...")
            while True:
                try:
                    desc = pinecone.describe_index(index_name)
                    if desc.status['ready']:
                        break
                    time.sleep(1)
                except:
                    time.sleep(1)
            
            logger.info(f"✅ Index '{index_name}' created successfully")
            index = pinecone.Index(index_name)
        
        return index
        
    except Exception as e:
        logger.error(f"❌ Error setting up Pinecone index: {e}")
        return None

def load_and_prepare_data():
    """Load and prepare data for upload"""
    logger.info("📖 Loading processed data...")
    
    data = []
    data_file = "nelson_pediatrics_processed.jsonl"
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
                        continue
        
        logger.info(f"✅ Loaded {len(data)} documents")
        
        # Prepare vectors for Pinecone
        logger.info("🔄 Preparing vectors for upload...")
        vectors = []
        
        for i, item in enumerate(data):
            # Create unique ID
            source_clean = item['source_file'].replace('.txt', '').replace(' ', '_').lower()
            vector_id = f"nelson_{source_clean}_{item['chunk_number']}"
            
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
        
        logger.info(f"✅ Prepared {len(vectors)} vectors for upload")
        return vectors
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return []

def upload_vectors(index, vectors: List[Dict[str, Any]]):
    """Upload vectors to Pinecone"""
    logger.info(f"📤 Uploading {len(vectors)} vectors to Pinecone...")
    
    batch_size = 100
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    # Upload in batches with progress bar
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches", total=total_batches):
        batch = vectors[i:i + batch_size]
        
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            logger.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
            continue
    
    logger.info("✅ Upload completed successfully")

def get_index_stats(index):
    """Get index statistics"""
    try:
        stats = index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness
        }
    except Exception as e:
        logger.error(f"❌ Error getting index stats: {e}")
        return {}

def test_search(index):
    """Test search functionality"""
    logger.info("🔍 Testing search functionality...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the same model used for embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test queries
        test_queries = [
            "asthma treatment in children",
            "heart murmur diagnosis",
            "fever management pediatric"
        ]
        
        for query in test_queries:
            logger.info(f"🔍 Testing query: '{query}'")
            
            # Generate embedding
            query_embedding = model.encode([query])[0].tolist()
            
            # Search
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            if results['matches']:
                logger.info(f"✅ Found {len(results['matches'])} results")
                for i, match in enumerate(results['matches'][:2], 1):
                    score = match['score']
                    topic = match['metadata'].get('topic', 'Unknown')
                    logger.info(f"   {i}. Score: {score:.3f} | Topic: {topic}")
            else:
                logger.warning(f"⚠️ No results found for '{query}'")
        
        logger.info("✅ Search test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🏥 Nelson Pediatrics - Complete Pinecone Setup")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Check credentials
    api_key, environment = check_credentials()
    if not api_key:
        return
    
    # Step 3: Check processed data
    if not check_processed_data():
        return
    
    # Step 4: Setup Pinecone index
    index = setup_pinecone_index(api_key, environment)
    if not index:
        return
    
    # Step 5: Load and prepare data
    vectors = load_and_prepare_data()
    if not vectors:
        return
    
    # Step 6: Upload vectors
    upload_vectors(index, vectors)
    
    # Step 7: Get statistics
    stats = get_index_stats(index)
    
    # Step 8: Test search
    search_success = test_search(index)
    
    # Final results
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Total vectors uploaded: {stats.get('total_vectors', 'Unknown')}")
    print(f"🧠 Vector dimension: {stats.get('dimension', 'Unknown')}")
    print(f"📈 Index fullness: {stats.get('index_fullness', 'Unknown')}")
    print(f"🏷️  Index name: nelson-pediatrics")
    print(f"🌐 Environment: {environment}")
    print(f"🔍 Search test: {'✅ Passed' if search_success else '❌ Failed'}")
    
    print("\n🚀 Next Steps:")
    print("1. Try interactive search: python demo_pinecone_search.py interactive")
    print("2. Build your medical search application!")
    print("3. Use the search API in your projects")
    
    print("\n💡 Example search code:")
    print("""
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize
pinecone.init(api_key="your-api-key", environment="us-east-1-aws")
index = pinecone.Index("nelson-pediatrics")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Search
query = "asthma treatment"
query_embedding = model.encode([query])[0].tolist()
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
""")
    
    print("\n🏥 Your Nelson Pediatrics knowledge base is ready! ⚡")

if __name__ == "__main__":
    main()


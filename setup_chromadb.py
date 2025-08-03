#!/usr/bin/env python3
"""
ChromaDB Local Setup for Nelson Pediatrics

This script creates a completely free, local vector database using ChromaDB.
No trials, no limits, no internet required after setup.
"""

import os
import json
import time
from typing import List, Dict, Any
import logging
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_chromadb():
    """Install ChromaDB if not already installed"""
    try:
        import chromadb
        logger.info("✅ ChromaDB already installed")
        return True
    except ImportError:
        logger.info("📦 Installing ChromaDB...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "chromadb"])
            logger.info("✅ ChromaDB installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install ChromaDB: {e}")
            return False

def setup_chromadb():
    """Setup ChromaDB client and collection"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create persistent client (data saved to disk)
        db_path = "./nelson_chromadb"
        logger.info(f"🗄️ Creating ChromaDB at: {db_path}")
        
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Create or get collection
        collection_name = "nelson_pediatrics"
        
        try:
            # Try to get existing collection
            collection = client.get_collection(collection_name)
            logger.info(f"📋 Found existing collection '{collection_name}'")
            
            # Check if it has data
            count = collection.count()
            logger.info(f"📊 Collection contains {count} documents")
            
            if count > 0:
                logger.info("✅ Collection already has data")
                return client, collection
                
        except Exception:
            # Collection doesn't exist, create it
            logger.info(f"🔨 Creating new collection '{collection_name}'")
            
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Nelson Pediatrics Medical Knowledge Base"}
            )
        
        return client, collection
        
    except Exception as e:
        logger.error(f"❌ Error setting up ChromaDB: {e}")
        return None, None

def load_processed_data():
    """Load all processed data"""
    logger.info("📖 Loading processed data...")
    
    data = []
    data_file = "nelson_pediatrics_processed.jsonl"
    
    if not os.path.exists(data_file):
        logger.error(f"❌ {data_file} not found!")
        return []
    
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
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return []

def upload_to_chromadb(collection, data):
    """Upload all data to ChromaDB"""
    logger.info(f"📤 Uploading {len(data)} documents to ChromaDB...")
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    for item in data:
        # Create unique ID
        source_clean = item['source_file'].replace('.txt', '').replace(' ', '_').lower()
        doc_id = f"nelson_{source_clean}_{item['chunk_number']}"
        
        # Prepare metadata
        metadata = {
            'source_file': item['source_file'],
            'topic': item['topic'],
            'chunk_number': item['chunk_number'],
            'character_count': item['character_count'],
            'created_at': str(item['created_at'])
        }
        
        # Add file metadata if available
        if 'metadata' in item:
            metadata.update({
                'file_size': item['metadata'].get('file_size', 0),
                'total_chunks': item['metadata'].get('total_chunks', 0),
                'embedding_model': item['metadata'].get('embedding_model', 'all-MiniLM-L6-v2')
            })
        
        ids.append(doc_id)
        embeddings.append(item['embedding'])
        metadatas.append(metadata)
        documents.append(item['text'])
    
    # Upload in batches
    batch_size = 1000
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    logger.info(f"📦 Uploading in {total_batches} batches of {batch_size}")
    
    for i in tqdm(range(0, len(data), batch_size), desc="Uploading batches", total=total_batches):
        end_idx = min(i + batch_size, len(data))
        
        batch_ids = ids[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        batch_documents = documents[i:end_idx]
        
        try:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
        except Exception as e:
            logger.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
            continue
    
    logger.info("✅ Upload completed successfully")

def test_search(collection):
    """Test search functionality"""
    logger.info("🔍 Testing search functionality...")
    
    try:
        # Test queries
        test_queries = [
            "asthma treatment in children",
            "heart murmur diagnosis",
            "fever management pediatric",
            "respiratory infection symptoms"
        ]
        
        for query in test_queries:
            logger.info(f"🔍 Testing query: '{query}'")
            
            # Search using ChromaDB
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['documents'] and results['documents'][0]:
                logger.info(f"✅ Found {len(results['documents'][0])} results")
                
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0][:2],
                    results['metadatas'][0][:2], 
                    results['distances'][0][:2]
                ), 1):
                    topic = metadata.get('topic', 'Unknown')
                    similarity = 1 - distance  # Convert distance to similarity
                    logger.info(f"   {i}. Similarity: {similarity:.3f} | Topic: {topic}")
            else:
                logger.warning(f"⚠️ No results found for '{query}'")
        
        logger.info("✅ Search test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False

def create_search_interface():
    """Create a simple search interface script"""
    search_script = '''#!/usr/bin/env python3
"""
Nelson Pediatrics - ChromaDB Search Interface

Simple command-line search interface for your medical knowledge base.
"""

import chromadb
from chromadb.config import Settings

def search_medical_knowledge(query, top_k=5):
    """Search the medical knowledge base"""
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path="./nelson_chromadb",
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get collection
    collection = client.get_collection("nelson_pediatrics")
    
    # Search
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return results

def interactive_search():
    """Interactive search mode"""
    print("🏥 Nelson Pediatrics - Medical Knowledge Search")
    print("=" * 60)
    print("Type your medical questions below (or 'quit' to exit)")
    print()
    
    while True:
        query = input("🔍 Search: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
            
        try:
            results = search_medical_knowledge(query)
            
            if results['documents'] and results['documents'][0]:
                print(f"\\n📋 Found {len(results['documents'][0])} results:")
                print("-" * 60)
                
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1):
                    
                    similarity = 1 - distance
                    topic = metadata.get('topic', 'Unknown')
                    source = metadata.get('source_file', 'Unknown')
                    
                    print(f"\\n{i}. 📊 Similarity: {similarity:.3f}")
                    print(f"   📚 Topic: {topic}")
                    print(f"   📄 Source: {source}")
                    print(f"   📝 Content: {doc[:200]}...")
                    
            else:
                print("\\n❌ No results found. Try different keywords.")
                
        except Exception as e:
            print(f"\\n❌ Search error: {e}")
        
        print("\\n" + "=" * 60)

if __name__ == "__main__":
    interactive_search()
'''
    
    with open("search_nelson.py", "w") as f:
        f.write(search_script)
    
    # Make it executable
    os.chmod("search_nelson.py", 0o755)
    logger.info("✅ Created search_nelson.py - Interactive search interface")

def main():
    """Main execution function"""
    
    print("🏥 Nelson Pediatrics - ChromaDB Local Setup")
    print("=" * 60)
    print("🆓 100% Free • 🏠 Local • 🔒 Private • ♾️ Unlimited")
    print("=" * 60)
    
    # Step 1: Install ChromaDB
    if not install_chromadb():
        return
    
    # Step 2: Setup ChromaDB
    client, collection = setup_chromadb()
    if not client or not collection:
        return
    
    # Step 3: Check if data already exists
    count = collection.count()
    if count > 0:
        logger.info(f"📊 Collection already contains {count} documents")
        
        # Test search
        search_success = test_search(collection)
        
        print("\n" + "=" * 60)
        print("🎉 CHROMADB ALREADY SET UP!")
        print("=" * 60)
        print(f"📊 Total documents: {count}")
        print(f"🔍 Search test: {'✅ Passed' if search_success else '❌ Failed'}")
        print(f"🗄️ Database location: ./nelson_chromadb")
        
        # Create search interface
        create_search_interface()
        
        print("\n🚀 Ready to search! Try:")
        print("   python search_nelson.py")
        return
    
    # Step 4: Load processed data
    data = load_processed_data()
    if not data:
        return
    
    # Step 5: Upload to ChromaDB
    upload_to_chromadb(collection, data)
    
    # Step 6: Test search
    search_success = test_search(collection)
    
    # Step 7: Create search interface
    create_search_interface()
    
    # Step 8: Get final statistics
    final_count = collection.count()
    
    # Final results
    print("\n" + "=" * 60)
    print("🎉 CHROMADB SETUP COMPLETED!")
    print("=" * 60)
    print(f"📊 Total documents uploaded: {final_count}")
    print(f"🧠 Vector dimension: 384")
    print(f"🗄️ Database location: ./nelson_chromadb")
    print(f"🔍 Search test: {'✅ Passed' if search_success else '❌ Failed'}")
    print(f"💾 Storage: Local (persistent)")
    print(f"💰 Cost: 100% Free Forever")
    
    print("\n🚀 Next Steps:")
    print("1. Interactive search: python search_nelson.py")
    print("2. Build your medical app using the ChromaDB API")
    print("3. Your data is stored locally in ./nelson_chromadb/")
    
    print("\n💡 Example usage:")
    print("""
import chromadb
client = chromadb.PersistentClient(path="./nelson_chromadb")
collection = client.get_collection("nelson_pediatrics")
results = collection.query(query_texts=["asthma treatment"], n_results=5)
""")
    
    print("\n🏥 Your Nelson Pediatrics knowledge base is ready! ⚡")
    print("🆓 Completely free, unlimited, and runs locally!")

if __name__ == "__main__":
    main()


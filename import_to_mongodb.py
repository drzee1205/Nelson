#!/usr/bin/env python3
"""
MongoDB Import Script for Nelson Pediatrics Data
"""

import json
import pymongo
from tqdm import tqdm

def import_to_mongodb():
    """Import JSONL data to MongoDB"""
    
    # Connection settings
    connection_string = "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson"
    
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
        # collection.delete_many({})
        
        # Read and import data
        documents = []
        with open('nelson_pediatrics_processed.jsonl', 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading data"):
                if line.strip():
                    documents.append(json.loads(line))
        
        print(f"📊 Importing {len(documents)} documents...")
        
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
        total_docs = collection.count_documents({})
        unique_sources = len(collection.distinct('source_file'))
        unique_topics = len(collection.distinct('topic'))
        
        print("\n" + "="*50)
        print("🎉 IMPORT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Total documents: {total_docs}")
        print(f"📚 Source files: {unique_sources}")
        print(f"🏷️  Topics: {unique_topics}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import_to_mongodb()

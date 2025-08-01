#!/usr/bin/env python3
"""
Test script to verify MongoDB connection and data upload
"""

import pymongo
import json
from datetime import datetime

def test_connection():
    """Test MongoDB connection and query uploaded data"""
    
    # Connection string
    MONGODB_CONNECTION = "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson"
    
    try:
        # Connect to MongoDB with SSL settings
        client = pymongo.MongoClient(
            MONGODB_CONNECTION,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        db = client.nelson_pediatrics
        collection = db.nelson_book_content
        
        print("🔗 Connected to MongoDB successfully!")
        
        # Get basic statistics
        total_docs = collection.count_documents({})
        print(f"📊 Total documents: {total_docs}")
        
        if total_docs > 0:
            # Get unique sources
            sources = collection.distinct('source_file')
            print(f"📚 Source files: {len(sources)}")
            for source in sources[:5]:  # Show first 5
                print(f"   - {source}")
            if len(sources) > 5:
                print(f"   ... and {len(sources) - 5} more")
            
            # Get unique topics
            topics = collection.distinct('topic')
            print(f"🏷️  Topics: {len(topics)}")
            for topic in topics[:5]:  # Show first 5
                print(f"   - {topic}")
            if len(topics) > 5:
                print(f"   ... and {len(topics) - 5} more")
            
            # Sample document
            sample = collection.find_one({}, {'embedding': 0})  # Exclude embedding for readability
            print(f"\n📄 Sample document:")
            print(json.dumps(sample, indent=2, default=str))
            
            # Test embedding search (if embeddings exist)
            sample_with_embedding = collection.find_one({'embedding': {'$exists': True}})
            if sample_with_embedding:
                embedding_dim = len(sample_with_embedding['embedding'])
                print(f"\n🧠 Embedding dimension: {embedding_dim}")
                print("✅ Embeddings are present and ready for vector search!")
            else:
                print("\n⚠️  No embeddings found in documents")
                
        else:
            print("📭 No documents found in the collection")
            
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return False
    
    return True

def search_example():
    """Example of how to search the uploaded content"""
    
    MONGODB_CONNECTION = "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson"
    
    try:
        client = pymongo.MongoClient(
            MONGODB_CONNECTION,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        db = client.nelson_pediatrics
        collection = db.nelson_book_content
        
        print("\n" + "="*50)
        print("🔍 SEARCH EXAMPLES")
        print("="*50)
        
        # Text search example
        search_term = "asthma"
        results = collection.find(
            {"$text": {"$search": search_term}},
            {"text": 1, "source_file": 1, "topic": 1, "score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(3)
        
        print(f"\n📝 Text search results for '{search_term}':")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Source: {doc['source_file']}")
            print(f"   Topic: {doc['topic']}")
            print(f"   Text: {doc['text'][:200]}...")
            print(f"   Score: {doc.get('score', 'N/A')}")
        
        # Topic-based search
        topic_results = collection.find({"topic": "The Respiratory System"}).limit(2)
        print(f"\n🫁 Sample results from 'The Respiratory System' topic:")
        for i, doc in enumerate(topic_results, 1):
            print(f"\n{i}. Chunk {doc.get('chunk_number', 'N/A')}")
            print(f"   Text: {doc['text'][:150]}...")
            
    except Exception as e:
        print(f"❌ Error during search: {e}")

if __name__ == "__main__":
    print("🧪 Testing MongoDB Connection and Data...")
    print("="*50)
    
    if test_connection():
        search_example()
        
    print("\n✅ Test completed!")

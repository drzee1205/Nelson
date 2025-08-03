#!/usr/bin/env python3
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
                print(f"\n📋 Found {len(results['documents'][0])} results:")
                print("-" * 60)
                
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1):
                    
                    similarity = 1 - distance
                    topic = metadata.get('topic', 'Unknown')
                    source = metadata.get('source_file', 'Unknown')
                    
                    print(f"\n{i}. 📊 Similarity: {similarity:.3f}")
                    print(f"   📚 Topic: {topic}")
                    print(f"   📄 Source: {source}")
                    print(f"   📝 Content: {doc[:200]}...")
                    
            else:
                print("\n❌ No results found. Try different keywords.")
                
        except Exception as e:
            print(f"\n❌ Search error: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    interactive_search()

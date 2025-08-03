#!/usr/bin/env python3
"""
Demo Pinecone Search Script for Nelson Pediatrics

This script demonstrates how to perform similarity search on your uploaded data.
"""

import os
import json
from typing import List, Dict, Any
import logging

try:
    import pinecone
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Required libraries not installed!")
    print("Install with: pip install pinecone-client sentence-transformers")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NelsonPediatricsSearch:
    def __init__(self, api_key: str, environment: str = "us-east-1-aws", index_name: str = "nelson-pediatrics"):
        """
        Initialize the search system
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
        """
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Connect to index
        self.index = pinecone.Index(index_name)
        
        # Initialize embedding model (same as used for upload)
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("✅ Nelson Pediatrics search system ready!")
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        logger.info(f"🔍 Searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Perform search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            result = {
                'score': match['score'],
                'id': match['id'],
                'metadata': match['metadata']
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def search_by_topic(self, query: str, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific medical topic
        
        Args:
            query: Search query text
            topic: Medical topic to filter by
            top_k: Number of results to return
            
        Returns:
            List of search results filtered by topic
        """
        filter_dict = {"topic": {"$eq": topic}}
        return self.search(query, top_k, filter_dict)
    
    def get_available_topics(self) -> List[str]:
        """Get list of available medical topics"""
        # This would require querying the index metadata
        # For now, return the known topics from our data
        return [
            "Allergic Disorder",
            "Behavioural & Pyschatrical Disorder", 
            "Bone And Joint Disorders",
            "Digestive System",
            "Diseases Of The Blood",
            "Ear",
            "Fluid &Electrolyte Disorder",
            "Growth Development & Behaviour",
            "Gynecologic History And  Physical Examination",
            "Humangenetics",
            "Immunology",
            "Learning & Developmental Disorder",
            "Metabolic Disorder",
            "Rehabilitation Medicine",
            "Rheumatic Disease",
            "Skin",
            "The Cardiovascular System",
            "The Endocrine System",
            "The Nervous System",
            "The Respiratory System",
            "Urology",
            "Aldocent Medicine",
            "Cancer & Benign Tumor"
        ]
    
    def print_results(self, results: List[Dict[str, Any]], max_text_length: int = 200):
        """Pretty print search results"""
        if not results:
            print("❌ No results found")
            return
        
        print(f"\n🎯 Found {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            score = result['score']
            
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   📚 Topic: {metadata.get('topic', 'Unknown')}")
            print(f"   📄 Source: {metadata.get('source_file', 'Unknown')}")
            print(f"   📝 Chunk: {metadata.get('chunk_number', 'Unknown')}")
            
            text = metadata.get('text', '')
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            print(f"   💬 Text: {text}")
            print("-" * 80)

def demo_searches():
    """Run demo searches"""
    
    # Get credentials
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set!")
        print("💡 Set your API key: export PINECONE_API_KEY='your-key'")
        return
    
    environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
    
    try:
        # Initialize search system
        search_system = NelsonPediatricsSearch(api_key, environment)
        
        # Demo searches
        demo_queries = [
            "asthma treatment in children",
            "heart murmur diagnosis",
            "fever management pediatric",
            "developmental milestones",
            "vaccination schedule"
        ]
        
        print("🏥 Nelson Pediatrics Search Demo")
        print("=" * 50)
        
        for query in demo_queries:
            print(f"\n🔍 Searching: '{query}'")
            results = search_system.search(query, top_k=3)
            search_system.print_results(results, max_text_length=150)
            
            # Wait for user input to continue
            input("\nPress Enter to continue to next search...")
        
        # Demo topic-specific search
        print("\n🎯 Topic-Specific Search Demo")
        print("=" * 50)
        
        respiratory_results = search_system.search_by_topic(
            "breathing problems", 
            "The Respiratory System", 
            top_k=3
        )
        
        print("\n🫁 Respiratory System - 'breathing problems':")
        search_system.print_results(respiratory_results)
        
        # Show available topics
        print("\n📋 Available Medical Topics:")
        topics = search_system.get_available_topics()
        for i, topic in enumerate(topics, 1):
            print(f"{i:2d}. {topic}")
        
        print("\n✅ Demo completed! Your Nelson Pediatrics search system is working!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

def interactive_search():
    """Interactive search mode"""
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set!")
        return
    
    environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
    
    try:
        search_system = NelsonPediatricsSearch(api_key, environment)
        
        print("🏥 Nelson Pediatrics Interactive Search")
        print("=" * 50)
        print("Type your medical questions. Type 'quit' to exit.")
        
        while True:
            query = input("\n🔍 Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            results = search_system.search(query, top_k=5)
            search_system.print_results(results)
            
    except Exception as e:
        logger.error(f"❌ Interactive search failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_search()
    else:
        demo_searches()


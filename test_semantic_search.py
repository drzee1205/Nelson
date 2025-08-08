#!/usr/bin/env python3
"""
Test Semantic Search - Nelson Pediatrics

This script tests the semantic search functionality after embeddings are generated.
It performs sample medical queries to verify the embedding quality.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import json

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing supabase dependency...")
    import os
    os.system("pip install supabase")
    from supabase import create_client, Client

# Hugging Face dependencies
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("❌ Installing Hugging Face dependencies...")
    import os
    os.system("pip install sentence-transformers torch")
    from sentence_transformers import SentenceTransformer
    import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

class SemanticSearchTester:
    """Test semantic search functionality for Nelson Pediatrics"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.supabase = None
        
        # Sample medical queries for testing
        self.test_queries = [
            {
                "query": "fever in children treatment",
                "expected_chapters": ["Infectious Diseases", "General Pediatrics"],
                "description": "Basic pediatric fever management"
            },
            {
                "query": "asthma symptoms and diagnosis",
                "expected_chapters": ["Respiratory System", "Allergic Disorders"],
                "description": "Respiratory condition diagnosis"
            },
            {
                "query": "growth and development milestones",
                "expected_chapters": ["Growth Development & Behaviour"],
                "description": "Developmental pediatrics"
            },
            {
                "query": "heart murmur in newborn",
                "expected_chapters": ["Cardiovascular System", "Neonatal"],
                "description": "Cardiac assessment in neonates"
            },
            {
                "query": "seizures in pediatric patients",
                "expected_chapters": ["Nervous System", "Neurology"],
                "description": "Neurological emergency"
            }
        ]
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase and verify embeddings exist"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Check if embeddings exist
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('count', count='exact')\
                .not_.is_('embedding', 'null')\
                .execute()
            
            embedding_count = result.count if result.count else 0
            
            # Get total records
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('count', count='exact')\
                .execute()
            
            total_count = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Records with embeddings: {embedding_count:,} / {total_count:,}")
            
            if embedding_count == 0:
                logger.error("❌ No embeddings found! Run embedding generation first.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the same model used for embedding generation"""
        try:
            logger.info(f"🤖 Loading model: {self.model_name}")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            
            logger.info(f"✅ Model loaded on {device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a search query"""
        try:
            # Preprocess query similar to how we processed the content
            processed_query = f"Medical Query: {query}"
            
            # Generate embedding
            embedding = self.model.encode(
                [processed_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error generating query embedding: {e}")
            return None
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            if not query_embedding:
                return []
            
            # Perform vector similarity search
            # Note: This uses a simple approach. In production, you'd use pgvector functions
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('id, content, chapter_title, section_title, page_number, embedding')\
                .not_.is_('embedding', 'null')\
                .limit(100)\
                .execute()  # Get more records to compute similarity locally
            
            if not result.data:
                return []
            
            # Compute cosine similarity locally
            similarities = []
            for record in result.data:
                if record['embedding']:
                    # Compute cosine similarity
                    doc_embedding = np.array(record['embedding'])
                    query_vec = np.array(query_embedding)
                    
                    similarity = np.dot(query_vec, doc_embedding) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(doc_embedding)
                    )
                    
                    similarities.append({
                        'record': record,
                        'similarity': float(similarity)
                    })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            results = []
            for item in similarities[:limit]:
                record = item['record']
                results.append({
                    'id': record['id'],
                    'content': record['content'],
                    'chapter_title': record['chapter_title'],
                    'section_title': record['section_title'],
                    'page_number': record['page_number'],
                    'similarity_score': item['similarity']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in semantic search: {e}")
            return []
    
    def test_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single query and evaluate results"""
        query = test_case['query']
        expected_chapters = test_case['expected_chapters']
        description = test_case['description']
        
        logger.info(f"🔍 Testing: {description}")
        logger.info(f"   Query: '{query}'")
        
        # Perform search
        results = self.semantic_search(query, limit=5)
        
        if not results:
            return {
                'query': query,
                'description': description,
                'success': False,
                'error': 'No results returned',
                'results': []
            }
        
        # Evaluate results
        found_chapters = set()
        relevant_results = 0
        
        for result in results:
            chapter = result['chapter_title']
            if chapter:
                found_chapters.add(chapter)
            
            # Check if result seems relevant (high similarity score)
            if result['similarity_score'] > 0.3:  # Threshold for relevance
                relevant_results += 1
        
        # Check if we found expected chapters
        expected_set = set(expected_chapters)
        chapter_match = bool(found_chapters.intersection(expected_set))
        
        success = chapter_match and relevant_results > 0
        
        return {
            'query': query,
            'description': description,
            'success': success,
            'relevant_results': relevant_results,
            'found_chapters': list(found_chapters),
            'expected_chapters': expected_chapters,
            'chapter_match': chapter_match,
            'top_similarity': results[0]['similarity_score'] if results else 0,
            'results': results[:3]  # Top 3 results for review
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test queries and compile results"""
        logger.info("🚀 Starting semantic search tests...")
        
        test_results = []
        successful_tests = 0
        
        for i, test_case in enumerate(self.test_queries, 1):
            logger.info(f"\n📋 Test {i}/{len(self.test_queries)}")
            
            result = self.test_query(test_case)
            test_results.append(result)
            
            if result['success']:
                successful_tests += 1
                logger.info(f"   ✅ SUCCESS - Found relevant results")
            else:
                logger.info(f"   ❌ FAILED - {result.get('error', 'Low relevance')}")
            
            # Show top result
            if result['results']:
                top_result = result['results'][0]
                logger.info(f"   🔝 Top result: {top_result['chapter_title']}")
                logger.info(f"   📊 Similarity: {top_result['similarity_score']:.3f}")
                logger.info(f"   📝 Content: {top_result['content'][:100]}...")
        
        # Compile summary
        success_rate = (successful_tests / len(self.test_queries)) * 100
        
        summary = {
            'total_tests': len(self.test_queries),
            'successful_tests': successful_tests,
            'failed_tests': len(self.test_queries) - successful_tests,
            'success_rate': success_rate,
            'test_results': test_results
        }
        
        return summary
    
    def print_test_summary(self, summary: Dict[str, Any]):
        """Print comprehensive test summary"""
        print("\n📊 SEMANTIC SEARCH TEST SUMMARY")
        print("=" * 60)
        print(f"🎯 Total tests: {summary['total_tests']}")
        print(f"✅ Successful: {summary['successful_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success rate: {summary['success_rate']:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(summary['test_results'], 1):
            status = "✅" if result['success'] else "❌"
            print(f"\n{i}. {status} {result['description']}")
            print(f"   Query: '{result['query']}'")
            print(f"   Relevant results: {result['relevant_results']}")
            print(f"   Top similarity: {result['top_similarity']:.3f}")
            print(f"   Found chapters: {', '.join(result['found_chapters'][:3])}")
            
            if result['results']:
                print(f"   Best match: {result['results'][0]['chapter_title']}")
        
        print(f"\n🎯 EVALUATION:")
        if summary['success_rate'] >= 80:
            print(f"🎉 EXCELLENT - Semantic search is working very well!")
            print(f"🔍 Your Nelson Pediatrics embeddings are high quality")
        elif summary['success_rate'] >= 60:
            print(f"✅ GOOD - Semantic search is working reasonably well")
            print(f"🔧 Consider fine-tuning for better results")
        else:
            print(f"⚠️ NEEDS IMPROVEMENT - Semantic search needs optimization")
            print(f"🔄 Consider regenerating embeddings with different model")

def main():
    """Main execution function"""
    
    print("🔍 NELSON PEDIATRICS SEMANTIC SEARCH TEST")
    print("=" * 60)
    print("🎯 Test embedding quality with medical queries")
    print("🏥 Verify semantic search functionality")
    print("=" * 60)
    
    # Initialize tester
    tester = SemanticSearchTester()
    
    # Connect to Supabase
    if not tester.connect_to_supabase():
        print("❌ Cannot connect to Supabase or no embeddings found.")
        print("💡 Run embedding generation first!")
        return
    
    # Load model
    if not tester.load_model():
        print("❌ Cannot load embedding model.")
        return
    
    # Run tests
    summary = tester.run_all_tests()
    
    # Print summary
    tester.print_test_summary(summary)
    
    if summary['success_rate'] >= 60:
        print("\n" + "=" * 60)
        print("🎉 SEMANTIC SEARCH TEST COMPLETE!")
        print("=" * 60)
        print("🔍 Your Nelson Pediatrics database supports semantic search")
        print("🏥 Ready for AI-powered medical queries")
    else:
        print("\n" + "=" * 60)
        print("⚠️ SEMANTIC SEARCH NEEDS IMPROVEMENT!")
        print("=" * 60)
        print("🔄 Consider regenerating embeddings or adjusting parameters")

if __name__ == "__main__":
    main()


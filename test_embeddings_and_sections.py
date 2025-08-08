#!/usr/bin/env python3
"""
Test Embeddings and Section Titles Implementation

This script tests the embedding generation, section title extraction,
and semantic search functionality.
"""

import os
import json
import time
from datetime import datetime

# Dependencies
try:
    from supabase import create_client, Client
    import requests
except ImportError:
    print("❌ Installing dependencies...")
    os.system("pip install supabase requests")
    from supabase import create_client, Client
    import requests

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def create_supabase_client():
    """Create Supabase client"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        return supabase
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return None

def test_database_schema(supabase: Client):
    """Test if database schema supports embeddings and sections"""
    print("🔍 Testing database schema...")
    
    try:
        # Check if embedding and section_title columns exist
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, embedding, section_title')\
            .limit(1)\
            .execute()
        
        if result.data:
            record = result.data[0]
            has_embedding_col = 'embedding' in record
            has_section_col = 'section_title' in record
            
            print(f"✅ Embedding column exists: {has_embedding_col}")
            print(f"✅ Section title column exists: {has_section_col}")
            
            return has_embedding_col and has_section_col
        else:
            print("❌ No data found in table")
            return False
            
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def test_embedding_coverage(supabase: Client):
    """Test embedding coverage in database"""
    print("\n📊 Testing embedding coverage...")
    
    try:
        # Total records
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_records = result.count if result.count else 0
        
        # Records with embeddings
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('embedding', 'null').execute()
        embedding_records = result.count if result.count else 0
        
        # Records with section titles
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('section_title', 'null').execute()
        section_records = result.count if result.count else 0
        
        embedding_coverage = (embedding_records / total_records * 100) if total_records > 0 else 0
        section_coverage = (section_records / total_records * 100) if total_records > 0 else 0
        
        print(f"📄 Total records: {total_records:,}")
        print(f"🤖 Records with embeddings: {embedding_records:,} ({embedding_coverage:.1f}%)")
        print(f"📝 Records with sections: {section_records:,} ({section_coverage:.1f}%)")
        
        return {
            'total': total_records,
            'embeddings': embedding_records,
            'sections': section_records,
            'embedding_coverage': embedding_coverage,
            'section_coverage': section_coverage
        }
        
    except Exception as e:
        print(f"❌ Coverage test failed: {e}")
        return None

def test_section_titles(supabase: Client):
    """Test section title extraction quality"""
    print("\n📝 Testing section title quality...")
    
    try:
        # Get sample records with section titles
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, chapter_title, section_title, content')\
            .not_.is_('section_title', 'null')\
            .limit(10)\
            .execute()
        
        if result.data:
            print("✅ Sample section titles:")
            for i, record in enumerate(result.data[:5], 1):
                section = record['section_title']
                chapter = record['chapter_title'][:30] + "..." if len(record['chapter_title']) > 30 else record['chapter_title']
                print(f"   {i}. Chapter: {chapter}")
                print(f"      Section: {section}")
            
            # Get section statistics
            result = supabase.table('nelson_textbook_chunks')\
                .select('section_title')\
                .not_.is_('section_title', 'null')\
                .execute()
            
            if result.data:
                section_counts = {}
                for record in result.data:
                    section = record['section_title']
                    section_counts[section] = section_counts.get(section, 0) + 1
                
                unique_sections = len(section_counts)
                most_common = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                print(f"\n📊 Section statistics:")
                print(f"   • Unique sections: {unique_sections}")
                print(f"   • Most common sections:")
                for section, count in most_common:
                    print(f"     - {section}: {count} documents")
                
                return True
        else:
            print("❌ No section titles found")
            return False
            
    except Exception as e:
        print(f"❌ Section title test failed: {e}")
        return False

def test_vector_search_function(supabase: Client):
    """Test vector search database function"""
    print("\n🔍 Testing vector search function...")
    
    try:
        # Test if search_embeddings function exists and works
        # We'll use a dummy embedding for testing
        dummy_embedding = [0.1] * 384  # 384-dimensional dummy vector
        
        result = supabase.rpc('search_embeddings', {
            'query_embedding': dummy_embedding,
            'match_threshold': 0.0,  # Very low threshold for testing
            'match_count': 3
        }).execute()
        
        if result.data:
            print(f"✅ Vector search function works: {len(result.data)} results")
            for i, record in enumerate(result.data, 1):
                similarity = record.get('similarity', 0)
                content_preview = record.get('content', '')[:50] + "..."
                print(f"   {i}. Similarity: {similarity:.3f} - {content_preview}")
            return True
        else:
            print("⚠️ Vector search function exists but returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Vector search function test failed: {e}")
        print("   This is expected if the function hasn't been created yet")
        return False

def test_api_endpoints():
    """Test the semantic search API endpoints"""
    print("\n🌐 Testing API endpoints...")
    
    api_base = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{api_base}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   • Status: {health_data.get('status')}")
            print(f"   • Semantic search available: {health_data.get('semantic_search_available')}")
            print(f"   • Embedding coverage: {health_data.get('embedding_coverage')}")
        else:
            print(f"⚠️ Health endpoint returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint test failed: {e}")
        print("   Make sure the API server is running: python semantic_search_api.py")
        return False
    
    # Test semantic search endpoint
    try:
        search_data = {
            "query": "asthma treatment",
            "top_k": 3,
            "min_similarity": 0.1
        }
        
        response = requests.post(f"{api_base}/search/semantic", 
                               json=search_data, 
                               timeout=10)
        
        if response.status_code == 200:
            search_results = response.json()
            print("✅ Semantic search endpoint working")
            print(f"   • Search type: {search_results.get('search_type')}")
            print(f"   • Results count: {search_results.get('results_count')}")
            
            if search_results.get('results'):
                for i, result in enumerate(search_results['results'][:2], 1):
                    similarity = result.get('similarity', 0)
                    content_preview = result.get('content', '')[:50] + "..."
                    print(f"   {i}. Similarity: {similarity:.3f} - {content_preview}")
        else:
            print(f"⚠️ Semantic search endpoint returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Semantic search endpoint test failed: {e}")
    
    # Test sections endpoint
    try:
        response = requests.get(f"{api_base}/search/sections", timeout=5)
        if response.status_code == 200:
            sections_data = response.json()
            print("✅ Sections endpoint working")
            print(f"   • Total sections: {sections_data.get('total_sections')}")
            
            if sections_data.get('sections'):
                print("   • Top sections:")
                for section in sections_data['sections'][:3]:
                    print(f"     - {section['title']}: {section['count']} documents")
        else:
            print(f"⚠️ Sections endpoint returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Sections endpoint test failed: {e}")
    
    return True

def run_embedding_generation_test():
    """Test the embedding generation script"""
    print("\n🤖 Testing embedding generation...")
    
    try:
        # Check if the script exists
        if os.path.exists('add_embeddings_and_sections.py'):
            print("✅ Embedding generation script found")
            print("   Run: python add_embeddings_and_sections.py")
            print("   This will generate embeddings for documents without them")
            return True
        else:
            print("❌ Embedding generation script not found")
            return False
    except Exception as e:
        print(f"❌ Embedding generation test failed: {e}")
        return False

def generate_test_report(supabase: Client):
    """Generate comprehensive test report"""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 70)
    
    # Database schema test
    schema_ok = test_database_schema(supabase)
    
    # Coverage test
    coverage_stats = test_embedding_coverage(supabase)
    
    # Section titles test
    sections_ok = test_section_titles(supabase)
    
    # Vector search function test
    vector_search_ok = test_vector_search_function(supabase)
    
    # API endpoints test
    api_ok = test_api_endpoints()
    
    # Embedding generation test
    embedding_script_ok = run_embedding_generation_test()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    tests = [
        ("Database Schema", schema_ok),
        ("Section Titles", sections_ok),
        ("Vector Search Function", vector_search_ok),
        ("API Endpoints", api_ok),
        ("Embedding Script", embedding_script_ok)
    ]
    
    passed_tests = sum(1 for _, result in tests if result)
    total_tests = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if coverage_stats:
        print(f"🤖 Embedding Coverage: {coverage_stats['embedding_coverage']:.1f}%")
        print(f"📝 Section Coverage: {coverage_stats['section_coverage']:.1f}%")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not schema_ok:
        print("   • Run update_database_schema.sql in Supabase SQL Editor")
    
    if coverage_stats and coverage_stats['embedding_coverage'] < 50:
        print("   • Run: python add_embeddings_and_sections.py")
    
    if not vector_search_ok:
        print("   • Deploy database functions from supabase_functions.sql")
    
    if not api_ok:
        print("   • Start API server: python semantic_search_api.py")
    
    print(f"\n🚀 Your Nelson Pediatrics database is {'ready' if passed_tests >= 3 else 'needs setup'} for AI-powered search!")

def main():
    """Main execution function"""
    
    print("🧪 NELSON PEDIATRICS - EMBEDDINGS & SECTIONS TEST SUITE")
    print("=" * 70)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Testing AI embeddings and section title functionality")
    print("=" * 70)
    
    # Connect to Supabase
    supabase = create_supabase_client()
    if not supabase:
        print("❌ Cannot connect to Supabase. Exiting.")
        return
    
    print("✅ Connected to Supabase successfully")
    
    # Run comprehensive tests
    generate_test_report(supabase)

if __name__ == "__main__":
    main()


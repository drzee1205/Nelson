#!/usr/bin/env python3
"""
Nelson Medical Knowledge API - Supabase Backend

This creates a REST API that connects to your Supabase database
for medical knowledge search. Perfect for NelsonGPT integration.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from typing import List, Dict, Any
import json

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing Supabase client...")
    os.system("pip install supabase")
    from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Supabase Configuration
SUPABASE_URL = os.getenv('VITE_SUPABASE_URL', 'https://nrtaztkewvbtzhbtkffc.supabase.co')
SUPABASE_SERVICE_KEY = os.getenv('VITE_SUPABASE_SERVICE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU')

# Global Supabase client
supabase_client = None

def get_supabase_client():
    """Get or create Supabase client"""
    global supabase_client
    
    if supabase_client is None:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            logger.info("✅ Supabase client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            return None
    
    return supabase_client

def calculate_text_similarity(query: str, text: str) -> float:
    """Simple text similarity calculation"""
    if not query or not text:
        return 0.0
    
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(query_words.intersection(text_words))
    union = len(query_words.union(text_words))
    
    if union == 0:
        return 0.0
    
    return intersection / union

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "status": "unhealthy",
                "error": "Supabase connection failed"
            }), 500
        
        # Test connection
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        count = result.count if result.count else 0
        
        return jsonify({
            "status": "healthy",
            "service": "Nelson Medical Knowledge API",
            "database": "Supabase PostgreSQL",
            "documents": count,
            "table": "nelson_textbook_chunks"
        })
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/search', methods=['POST'])
def search_medical_knowledge():
    """
    Search the medical knowledge base
    
    Request body:
    {
        "query": "asthma treatment in children",
        "top_k": 5,
        "include_metadata": true
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "example": {"query": "asthma treatment", "top_k": 5}
            }), 400
        
        query = data['query'].strip()
        top_k = data.get('top_k', 5)
        include_metadata = data.get('include_metadata', True)
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if top_k > 50:
            top_k = 50  # Limit results for performance
        
        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Search using Supabase text search
        try:
            # Try full-text search first
            result = supabase.table('nelson_textbook_chunks')\
                .select('id, chapter_title, section_title, content, chunk_index, metadata, created_at')\
                .text_search('content', query)\
                .limit(top_k)\
                .execute()
            
            search_results = result.data if result.data else []
            
            # If no results from full-text search, try ILIKE search
            if not search_results:
                result = supabase.table('nelson_textbook_chunks')\
                    .select('id, chapter_title, section_title, content, chunk_index, metadata, created_at')\
                    .ilike('content', f'%{query}%')\
                    .limit(top_k)\
                    .execute()
                
                search_results = result.data if result.data else []
            
        except Exception as search_error:
            logger.warning(f"⚠️ Advanced search failed, using basic search: {search_error}")
            
            # Fallback to basic ILIKE search
            result = supabase.table('nelson_textbook_chunks')\
                .select('id, chapter_title, section_title, content, chunk_index, metadata, created_at')\
                .ilike('content', f'%{query}%')\
                .limit(top_k)\
                .execute()
            
            search_results = result.data if result.data else []
        
        # Calculate similarity scores and format results
        formatted_results = []
        
        for i, record in enumerate(search_results):
            # Calculate similarity score
            similarity = calculate_text_similarity(query, record['content'])
            
            formatted_result = {
                "rank": i + 1,
                "content": record['content'],
                "similarity": round(similarity, 4)
            }
            
            if include_metadata:
                formatted_result["metadata"] = {
                    "id": record['id'],
                    "chapter_title": record['chapter_title'],
                    "section_title": record['section_title'],
                    "chunk_index": record['chunk_index'],
                    "created_at": record['created_at']
                }
                
                # Add original metadata if available
                if record.get('metadata'):
                    formatted_result["metadata"]["original"] = record['metadata']
            
            formatted_results.append(formatted_result)
        
        # Sort by similarity score
        formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/search/chapters', methods=['GET'])
def get_available_chapters():
    """Get all available medical chapters"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get distinct chapters with counts
        result = supabase.table('nelson_textbook_chunks')\
            .select('chapter_title')\
            .execute()
        
        if not result.data:
            return jsonify({
                "chapters": [],
                "count": 0,
                "status": "success"
            })
        
        # Count chapters
        chapter_counts = {}
        for record in result.data:
            chapter = record['chapter_title']
            chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
        
        # Sort by count
        sorted_chapters = sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)
        chapters = [chapter for chapter, count in sorted_chapters]
        
        return jsonify({
            "chapters": chapters,
            "count": len(chapters),
            "chapter_counts": dict(sorted_chapters),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Chapters error: {e}")
        return jsonify({
            "error": "Failed to get chapters",
            "message": str(e)
        }), 500

@app.route('/search/chapter/<chapter_name>', methods=['POST'])
def search_by_chapter(chapter_name):
    """
    Search within a specific medical chapter
    
    Request body:
    {
        "query": "heart murmur",
        "top_k": 5
    }
    """
    try:
        data = request.get_json()
        
        query = data.get('query', '') if data else ''
        top_k = data.get('top_k', 10) if data else 10
        
        if top_k > 50:
            top_k = 50
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Search within specific chapter
        query_builder = supabase.table('nelson_textbook_chunks')\
            .select('id, chapter_title, section_title, content, chunk_index, metadata')\
            .ilike('chapter_title', f'%{chapter_name}%')
        
        if query:
            query_builder = query_builder.ilike('content', f'%{query}%')
        
        result = query_builder.limit(top_k).execute()
        search_results = result.data if result.data else []
        
        # Format results
        formatted_results = []
        
        for i, record in enumerate(search_results):
            similarity = calculate_text_similarity(query, record['content']) if query else 1.0
            
            formatted_results.append({
                "rank": i + 1,
                "content": record['content'],
                "similarity": round(similarity, 4),
                "metadata": {
                    "id": record['id'],
                    "chapter_title": record['chapter_title'],
                    "section_title": record['section_title'],
                    "chunk_index": record['chunk_index']
                }
            })
        
        # Sort by similarity if query provided
        if query:
            formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            "query": query,
            "chapter": chapter_name,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Chapter search error: {e}")
        return jsonify({
            "error": "Chapter search failed",
            "message": str(e)
        }), 500

@app.route('/search/vector', methods=['POST'])
def vector_search():
    """
    Vector similarity search using embeddings
    
    Request body:
    {
        "query": "asthma treatment",
        "embedding": [0.1, 0.2, ...], // 384-dimensional vector
        "top_k": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'embedding' not in data:
            return jsonify({
                "error": "Missing 'embedding' in request body",
                "note": "Generate embedding for your query first"
            }), 400
        
        query = data.get('query', '')
        embedding = data['embedding']
        top_k = data.get('top_k', 5)
        
        if len(embedding) != 384:
            return jsonify({
                "error": "Embedding must be 384-dimensional"
            }), 400
        
        if top_k > 50:
            top_k = 50
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Vector similarity search using Supabase
        try:
            # Use RPC function for vector search (needs to be created in Supabase)
            result = supabase.rpc('search_embeddings', {
                'query_embedding': embedding,
                'match_threshold': 0.1,
                'match_count': top_k
            }).execute()
            
            search_results = result.data if result.data else []
            
        except Exception as vector_error:
            logger.warning(f"⚠️ Vector search failed: {vector_error}")
            return jsonify({
                "error": "Vector search not available",
                "message": "Please create the search_embeddings RPC function in Supabase",
                "note": "Falling back to text search"
            }), 501
        
        # Format results
        formatted_results = []
        
        for i, record in enumerate(search_results):
            formatted_results.append({
                "rank": i + 1,
                "content": record.get('content', ''),
                "similarity": record.get('similarity', 0.0),
                "metadata": {
                    "id": record.get('id'),
                    "chapter_title": record.get('chapter_title'),
                    "section_title": record.get('section_title'),
                    "chunk_index": record.get('chunk_index')
                }
            })
        
        return jsonify({
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "search_type": "vector_similarity",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Vector search error: {e}")
        return jsonify({
            "error": "Vector search failed",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get total count
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_documents = result.count if result.count else 0
        
        # Get chapter counts
        result = supabase.table('nelson_textbook_chunks')\
            .select('chapter_title')\
            .execute()
        
        chapter_counts = {}
        if result.data:
            for record in result.data:
                chapter = record['chapter_title']
                chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
        
        # Get latest update
        result = supabase.table('nelson_textbook_chunks')\
            .select('created_at')\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        latest_update = None
        if result.data:
            latest_update = result.data[0]['created_at']
        
        # Top chapters
        top_chapters = dict(sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return jsonify({
            "total_documents": total_documents,
            "total_chapters": len(chapter_counts),
            "latest_update": latest_update,
            "database_type": "Supabase PostgreSQL",
            "table_name": "nelson_textbook_chunks",
            "top_chapters": top_chapters,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return jsonify({
            "error": "Failed to get stats",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /search", 
            "GET /search/chapters",
            "POST /search/chapter/<chapter_name>",
            "POST /search/vector",
            "GET /stats"
        ]
    }), 404

if __name__ == '__main__':
    print("🏥 Nelson Medical Knowledge API - Supabase Backend")
    print("=" * 70)
    
    # Test Supabase connection
    supabase = get_supabase_client()
    if not supabase:
        print("❌ Failed to connect to Supabase. Exiting.")
        exit(1)
    
    try:
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        count = result.count if result.count else 0
        print(f"✅ Connected to Supabase with {count} documents")
    except Exception as e:
        print(f"⚠️ Connection test warning: {e}")
    
    print("🌐 Starting API server...")
    print("\n📋 Available endpoints:")
    print("  • GET  /health - Health check")
    print("  • POST /search - Search medical knowledge")
    print("  • GET  /search/chapters - Get available chapters")
    print("  • POST /search/chapter/<chapter> - Search by chapter")
    print("  • POST /search/vector - Vector similarity search")
    print("  • GET  /stats - Database statistics")
    
    print("\n💡 Example usage:")
    print("curl -X POST http://localhost:5000/search \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"query\": \"asthma treatment\", \"top_k\": 3}'")
    
    print("\n🚀 Server starting on http://localhost:5000")
    print("🌐 Connected to Supabase PostgreSQL")
    print("=" * 70)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )


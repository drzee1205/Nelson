#!/usr/bin/env python3
"""
Working Semantic Search API

This API reads embeddings from the metadata field where they are actually stored
and provides working semantic search functionality.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Flask and API dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS

# ML and embedding dependencies
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from supabase import create_client, Client
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("❌ Installing dependencies...")
    os.system("pip install sentence-transformers torch supabase flask flask-cors scikit-learn")
    from sentence_transformers import SentenceTransformer
    import torch
    from supabase import create_client, Client
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

# Global variables for models
embedding_model = None
supabase_client = None

def get_supabase_client():
    """Get or create Supabase client"""
    global supabase_client
    
    if not supabase_client:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            logger.info("✅ Supabase client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            return None
    
    return supabase_client

def get_embedding_model():
    """Get or load embedding model"""
    global embedding_model
    
    if not embedding_model:
        try:
            model_name = "all-MiniLM-L6-v2"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info(f"📥 Loading embedding model: {model_name} on {device}")
            embedding_model = SentenceTransformer(model_name, device=device)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return None
    
    return embedding_model

def get_embedding_from_metadata(record):
    """Extract embedding from metadata field"""
    try:
        metadata = record.get('metadata', {})
        if isinstance(metadata, dict) and 'embedding' in metadata:
            return metadata['embedding']
        return None
    except Exception as e:
        logger.warning(f"Error extracting embedding from metadata: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check with actual embedding status"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "status": "unhealthy",
                "error": "Supabase connection failed"
            }), 500
        
        # Get actual embedding statistics
        total_result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = total_result.count if total_result.count else 0
        
        # Count records with embeddings in metadata
        metadata_result = supabase.table('nelson_textbook_chunks')\
            .select('metadata')\
            .not_.is_('metadata', 'null')\
            .execute()
        
        embeddings_count = 0
        for record in metadata_result.data:
            if get_embedding_from_metadata(record):
                embeddings_count += 1
        
        return jsonify({
            "status": "healthy",
            "service": "Nelson Medical Knowledge API - Working Embeddings",
            "total_documents": total_count,
            "documents_with_embeddings": embeddings_count,
            "embedding_coverage": f"{embeddings_count/total_count*100:.2f}%" if total_count > 0 else "0%",
            "embedding_storage": "metadata field",
            "embedding_dimension": "384D",
            "embedding_model": "all-MiniLM-L6-v2",
            "semantic_search_available": embedding_model is not None and embeddings_count > 0,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/search/semantic', methods=['POST'])
def semantic_search():
    """
    Working semantic search using embeddings from metadata field
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "example": {
                    "query": "asthma treatment in children",
                    "top_k": 5,
                    "min_similarity": 0.1
                }
            }), 400
        
        query = data['query'].strip()
        top_k = min(data.get('top_k', 5), 20)
        min_similarity = data.get('min_similarity', 0.1)
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        supabase = get_supabase_client()
        model = get_embedding_model()
        
        if not supabase or not model:
            return jsonify({"error": "Service not available"}), 500
        
        # Generate query embedding
        query_embedding = model.encode(query, convert_to_tensor=False)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Get records with embeddings from metadata
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, content, chapter_title, section_title, page_number, chunk_index, metadata, created_at')\
            .not_.is_('metadata', 'null')\
            .limit(200)\
            .execute()  # Get more records to search through
        
        if not result.data:
            return jsonify({
                "query": query,
                "results": [],
                "results_count": 0,
                "message": "No records with embeddings found"
            })
        
        # Calculate similarities
        similarities = []
        
        for record in result.data:
            stored_embedding = get_embedding_from_metadata(record)
            
            if stored_embedding:
                try:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [stored_embedding]
                    )[0][0]
                    
                    if similarity >= min_similarity:
                        similarities.append({
                            'record': record,
                            'similarity': float(similarity)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error calculating similarity: {e}")
                    continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Format top results
        top_results = similarities[:top_k]
        formatted_results = []
        
        for i, item in enumerate(top_results):
            record = item['record']
            similarity = item['similarity']
            
            formatted_result = {
                "rank": i + 1,
                "content": record.get('content', ''),
                "similarity": round(similarity, 4),
                "page_number": record.get('page_number'),
                "section_title": record.get('section_title'),
                "chapter_title": record.get('chapter_title'),
                "chunk_index": record.get('chunk_index'),
                "id": record.get('id')
            }
            
            formatted_results.append(formatted_result)
        
        return jsonify({
            "query": query,
            "search_type": "semantic_metadata",
            "results_count": len(formatted_results),
            "total_searched": len([r for r in result.data if get_embedding_from_metadata(r)]),
            "results": formatted_results,
            "filters_applied": {
                "min_similarity": min_similarity,
                "top_k": top_k
            },
            "embedding_source": "metadata field",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Semantic search error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/embeddings/status', methods=['GET'])
def embedding_status():
    """Get detailed embedding status"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get comprehensive stats
        total_result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = total_result.count if total_result.count else 0
        
        # Get sample of metadata records
        metadata_result = supabase.table('nelson_textbook_chunks')\
            .select('metadata, chapter_title')\
            .not_.is_('metadata', 'null')\
            .limit(100)\
            .execute()
        
        embeddings_count = 0
        chapters_with_embeddings = set()
        
        for record in metadata_result.data:
            if get_embedding_from_metadata(record):
                embeddings_count += 1
                chapter = record.get('chapter_title', 'Unknown')
                chapters_with_embeddings.add(chapter)
        
        return jsonify({
            "total_records": total_count,
            "records_with_embeddings": embeddings_count,
            "coverage_percent": round(embeddings_count/total_count*100, 2) if total_count > 0 else 0,
            "chapters_with_embeddings": len(chapters_with_embeddings),
            "sample_chapters": list(chapters_with_embeddings)[:10],
            "embedding_storage": "metadata field",
            "embedding_dimension": "384D",
            "model": "all-MiniLM-L6-v2",
            "status": "working",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Embedding status error: {e}")
        return jsonify({
            "error": "Failed to get embedding status",
            "message": str(e)
        }), 500

@app.route('/test/search', methods=['GET'])
def test_search():
    """Test endpoint for quick semantic search testing"""
    try:
        query = request.args.get('q', 'asthma treatment')
        
        # Use the semantic search endpoint
        test_data = {
            'query': query,
            'top_k': 3,
            'min_similarity': 0.1
        }
        
        # Call our own semantic search function
        with app.test_request_context('/search/semantic', method='POST', json=test_data):
            response = semantic_search()
            return response
        
    except Exception as e:
        return jsonify({
            "error": "Test search failed",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health - System health check with embedding status",
            "POST /search/semantic - Semantic search using metadata embeddings",
            "GET /embeddings/status - Detailed embedding statistics",
            "GET /test/search?q=query - Quick test search"
        ]
    }), 404

if __name__ == '__main__':
    print("🚀 NELSON PEDIATRICS - WORKING SEMANTIC SEARCH API")
    print("=" * 80)
    
    # Initialize components
    supabase = get_supabase_client()
    if not supabase:
        print("❌ Failed to connect to Supabase. Exiting.")
        exit(1)
    
    # Load embedding model
    model = get_embedding_model()
    if model:
        print("✅ AI embedding model loaded successfully")
    else:
        print("⚠️ Embedding model failed to load - semantic search will not work")
    
    print("🌐 Starting working semantic search API...")
    print("\n📋 Working endpoints:")
    print("  • GET  /health - System health with actual embedding status")
    print("  • POST /search/semantic - Working semantic search")
    print("  • GET  /embeddings/status - Detailed embedding statistics")
    print("  • GET  /test/search?q=query - Quick test search")
    
    print("\n💡 Example semantic search:")
    print("curl -X POST http://localhost:5000/search/semantic \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print("    \"query\": \"asthma treatment in pediatric patients\",")
    print("    \"top_k\": 5,")
    print("    \"min_similarity\": 0.1")
    print("  }'")
    
    print("\n🔍 Quick test:")
    print("curl 'http://localhost:5000/test/search?q=asthma'")
    
    print("\n🚀 API server starting on http://localhost:5000")
    print("🤖 Semantic search using embeddings from metadata field!")
    print("=" * 80)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )


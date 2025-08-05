#!/usr/bin/env python3
"""
Semantic Search API with Hugging Face Embeddings

Enhanced API that supports semantic search using AI embeddings and section-based filtering.
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
except ImportError:
    print("❌ Installing dependencies...")
    os.system("pip install sentence-transformers torch supabase flask flask-cors")
    from sentence_transformers import SentenceTransformer
    import torch
    from supabase import create_client, Client
    import numpy as np

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
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info(f"📥 Loading embedding model: {model_name} on {device}")
            embedding_model = SentenceTransformer(model_name, device=device)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return None
    
    return embedding_model

def generate_query_embedding(query: str) -> Optional[List[float]]:
    """Generate embedding for search query"""
    try:
        model = get_embedding_model()
        if not model:
            return None
        
        # Clean and encode query
        cleaned_query = query.strip()
        embedding = model.encode(cleaned_query, convert_to_tensor=False)
        
        # Convert to list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        return embedding
        
    except Exception as e:
        logger.error(f"❌ Error generating query embedding: {e}")
        return None

def calculate_text_similarity(query: str, text: str) -> float:
    """Calculate similarity between query and text"""
    try:
        # Simple word overlap similarity as fallback
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
        
    except Exception as e:
        logger.error(f"❌ Error calculating similarity: {e}")
        return 0.0

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with embedding model status"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "status": "unhealthy",
                "error": "Supabase connection failed"
            }), 500
        
        # Test database connection
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = result.count if result.count else 0
        
        # Check embedding model
        model_status = "loaded" if embedding_model else "not_loaded"
        
        # Check records with embeddings
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('embedding', 'null').execute()
        embedding_count = result.count if result.count else 0
        
        # Check records with section titles
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('section_title', 'null').execute()
        section_count = result.count if result.count else 0
        
        return jsonify({
            "status": "healthy",
            "service": "Nelson Medical Knowledge API with Semantic Search",
            "database": "Supabase PostgreSQL",
            "total_documents": total_count,
            "documents_with_embeddings": embedding_count,
            "documents_with_sections": section_count,
            "embedding_coverage": f"{embedding_count/total_count*100:.1f}%" if total_count > 0 else "0%",
            "section_coverage": f"{section_count/total_count*100:.1f}%" if total_count > 0 else "0%",
            "embedding_model_status": model_status,
            "semantic_search_available": model_status == "loaded" and embedding_count > 0
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
    Semantic search using AI embeddings
    
    Request body:
    {
        "query": "asthma treatment in children",
        "top_k": 5,
        "min_similarity": 0.1,
        "include_metadata": true,
        "min_page": 1100,
        "max_page": 1200,
        "section_filter": "Treatment"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "example": {"query": "asthma treatment", "top_k": 5}
            }), 400
        
        query = data['query'].strip()
        top_k = data.get('top_k', 5)
        min_similarity = data.get('min_similarity', 0.1)
        include_metadata = data.get('include_metadata', True)
        min_page = data.get('min_page')
        max_page = data.get('max_page')
        section_filter = data.get('section_filter')
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if top_k > 50:
            top_k = 50
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        if query_embedding:
            # Use vector similarity search
            try:
                # Build RPC call for vector search
                rpc_params = {
                    'query_embedding': query_embedding,
                    'match_threshold': min_similarity,
                    'match_count': top_k
                }
                
                result = supabase.rpc('search_embeddings', rpc_params).execute()
                search_results = result.data if result.data else []
                
                # Apply additional filters
                if min_page or max_page or section_filter:
                    filtered_results = []
                    for record in search_results:
                        # Page filter
                        if min_page and record.get('page_number', 0) < min_page:
                            continue
                        if max_page and record.get('page_number', 99999) > max_page:
                            continue
                        
                        # Section filter
                        if section_filter and record.get('section_title'):
                            if section_filter.lower() not in record['section_title'].lower():
                                continue
                        
                        filtered_results.append(record)
                    
                    search_results = filtered_results[:top_k]
                
                search_type = "vector_similarity"
                
            except Exception as vector_error:
                logger.warning(f"⚠️ Vector search failed: {vector_error}")
                search_results = []
                search_type = "vector_failed"
        else:
            search_results = []
            search_type = "no_embedding"
        
        # Fallback to text search if vector search failed or no results
        if not search_results:
            logger.info("🔄 Falling back to text search")
            
            # Build text search query
            query_builder = supabase.table('nelson_textbook_chunks')\
                .select('id, chapter_title, section_title, content, page_number, chunk_index, metadata, created_at')\
                .text_search('content', query)
            
            # Apply filters
            if min_page:
                query_builder = query_builder.gte('page_number', min_page)
            if max_page:
                query_builder = query_builder.lte('page_number', max_page)
            if section_filter:
                query_builder = query_builder.ilike('section_title', f'%{section_filter}%')
            
            result = query_builder.limit(top_k).execute()
            search_results = result.data if result.data else []
            search_type = "text_search"
        
        # Format results
        formatted_results = []
        
        for i, record in enumerate(search_results):
            # Calculate similarity score
            if 'similarity' in record:
                similarity = record['similarity']
            else:
                similarity = calculate_text_similarity(query, record.get('content', ''))
            
            formatted_result = {
                "rank": i + 1,
                "content": record.get('content', ''),
                "similarity": round(float(similarity), 4),
                "page_number": record.get('page_number'),
                "section_title": record.get('section_title')
            }
            
            if include_metadata:
                formatted_result["metadata"] = {
                    "id": record.get('id'),
                    "chapter_title": record.get('chapter_title'),
                    "chunk_index": record.get('chunk_index'),
                    "created_at": record.get('created_at')
                }
                
                # Add original metadata if available
                if record.get('metadata'):
                    formatted_result["metadata"]["original"] = record['metadata']
            
            formatted_results.append(formatted_result)
        
        # Sort by similarity score
        formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            "query": query,
            "search_type": search_type,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "filters": {
                "min_page": min_page,
                "max_page": max_page,
                "section_filter": section_filter,
                "min_similarity": min_similarity
            },
            "semantic_search_available": query_embedding is not None,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Semantic search error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/search/sections', methods=['GET'])
def get_available_sections():
    """Get all available section titles"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get distinct section titles
        result = supabase.table('nelson_textbook_chunks')\
            .select('section_title')\
            .not_.is_('section_title', 'null')\
            .execute()
        
        if not result.data:
            return jsonify({
                "sections": [],
                "count": 0,
                "status": "success"
            })
        
        # Count sections
        section_counts = {}
        for record in result.data:
            section = record['section_title']
            if section:
                section_counts[section] = section_counts.get(section, 0) + 1
        
        # Sort by count
        sorted_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            "sections": [{"title": section, "count": count} for section, count in sorted_sections],
            "total_sections": len(sorted_sections),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Sections error: {e}")
        return jsonify({
            "error": "Failed to get sections",
            "message": str(e)
        }), 500

@app.route('/search/by-section/<section_name>', methods=['POST'])
def search_by_section(section_name):
    """
    Search within a specific section
    
    Request body:
    {
        "query": "treatment options",
        "top_k": 10
    }
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if top_k > 50:
            top_k = 50
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Search within specific section
        query_builder = supabase.table('nelson_textbook_chunks')\
            .select('id, chapter_title, section_title, content, page_number, chunk_index, created_at')\
            .ilike('section_title', f'%{section_name}%')
        
        if query:
            query_builder = query_builder.text_search('content', query)
        
        result = query_builder.limit(top_k).execute()
        search_results = result.data if result.data else []
        
        # Calculate similarity scores if query provided
        if query:
            for record in search_results:
                record['similarity'] = calculate_text_similarity(query, record.get('content', ''))
            
            # Sort by similarity
            search_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return jsonify({
            "section": section_name,
            "query": query,
            "results_count": len(search_results),
            "results": search_results,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Section search error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/embeddings/generate', methods=['POST'])
def generate_embedding_endpoint():
    """
    Generate embedding for a given text
    
    Request body:
    {
        "text": "asthma treatment in pediatric patients"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' in request body"
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        # Generate embedding
        embedding = generate_query_embedding(text)
        
        if embedding:
            return jsonify({
                "text": text,
                "embedding": embedding,
                "dimension": len(embedding),
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Failed to generate embedding"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Embedding generation error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /search/semantic",
            "GET /search/sections",
            "POST /search/by-section/<section_name>",
            "POST /embeddings/generate"
        ]
    }), 404

if __name__ == '__main__':
    print("🤖 Nelson Medical Knowledge API - Semantic Search Edition")
    print("=" * 80)
    
    # Initialize components
    supabase = get_supabase_client()
    if not supabase:
        print("❌ Failed to connect to Supabase. Exiting.")
        exit(1)
    
    # Test database
    try:
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = result.count if result.count else 0
        
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('embedding', 'null').execute()
        embedding_count = result.count if result.count else 0
        
        print(f"✅ Connected to Supabase with {total_count:,} documents")
        print(f"🤖 Documents with embeddings: {embedding_count:,} ({embedding_count/total_count*100:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Connection test warning: {e}")
    
    # Load embedding model
    model = get_embedding_model()
    if model:
        print("✅ Embedding model loaded successfully")
    else:
        print("⚠️ Embedding model failed to load - semantic search will be limited")
    
    print("🌐 Starting API server...")
    print("\n📋 Available endpoints:")
    print("  • GET  /health - Health check with embedding status")
    print("  • POST /search/semantic - AI-powered semantic search")
    print("  • GET  /search/sections - Get available section titles")
    print("  • POST /search/by-section/<name> - Search within specific section")
    print("  • POST /embeddings/generate - Generate embedding for text")
    
    print("\n💡 Example semantic search:")
    print("curl -X POST http://localhost:5000/search/semantic \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"query\": \"asthma treatment\", \"top_k\": 5, \"min_similarity\": 0.1}'")
    
    print("\n🚀 Server starting on http://localhost:5000")
    print("🤖 AI-powered semantic search enabled!")
    print("=" * 80)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )


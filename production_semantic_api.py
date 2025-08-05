#!/usr/bin/env python3
"""
Production Semantic Search API for Nelson Pediatrics

Enhanced API with full AI capabilities, health monitoring, and production features.
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
            model_name = "sentence-transformers/all-mpnet-base-v2"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info(f"📥 Loading embedding model: {model_name} on {device}")
            embedding_model = SentenceTransformer(model_name, device=device)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return None
    
    return embedding_model

def generate_query_embedding(query: str) -> Optional[List[float]]:
    """Generate 1536D embedding for search query"""
    try:
        model = get_embedding_model()
        if not model:
            return None
        
        # Clean and encode query
        cleaned_query = query.strip()
        embedding = model.encode(cleaned_query, convert_to_tensor=False)
        
        # Convert to list and adjust to 1536D
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Pad to 1536 dimensions if needed
        current_dim = len(embedding)
        if current_dim < 1536:
            padding = [0.0] * (1536 - current_dim)
            embedding = embedding + padding
        elif current_dim > 1536:
            embedding = embedding[:1536]
        
        return embedding
        
    except Exception as e:
        logger.error(f"❌ Error generating query embedding: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with AI system status"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "status": "unhealthy",
                "error": "Supabase connection failed"
            }), 500
        
        # Use AI health check function if available
        try:
            result = supabase.rpc('ai_system_health_check').execute()
            if result.data:
                health_metrics = {}
                for metric in result.data:
                    health_metrics[metric['metric_name']] = {
                        'value': metric['metric_value'],
                        'status': metric['status'],
                        'details': metric.get('details', {})
                    }
                
                return jsonify({
                    "status": "healthy",
                    "service": "Nelson Medical Knowledge API - Production",
                    "ai_system_health": health_metrics,
                    "embedding_model_status": "loaded" if embedding_model else "not_loaded",
                    "timestamp": datetime.now().isoformat()
                })
        except:
            # Fallback to basic health check
            result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            total_count = result.count if result.count else 0
            
            result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('embedding', 'null').execute()
            embedding_count = result.count if result.count else 0
            
            return jsonify({
                "status": "healthy",
                "service": "Nelson Medical Knowledge API - Production",
                "total_documents": total_count,
                "documents_with_embeddings": embedding_count,
                "embedding_coverage": f"{embedding_count/total_count*100:.1f}%" if total_count > 0 else "0%",
                "embedding_model_status": "loaded" if embedding_model else "not_loaded",
                "ai_search_available": embedding_model is not None and embedding_count > 0
            })
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/search/ai', methods=['POST'])
def ai_semantic_search():
    """
    Advanced AI semantic search with full filtering capabilities
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "example": {
                    "query": "asthma treatment in children",
                    "top_k": 5,
                    "min_similarity": 0.1,
                    "min_page": 1100,
                    "max_page": 1200,
                    "section_filter": "Treatment",
                    "chapter_filter": "Allergic"
                }
            }), 400
        
        query = data['query'].strip()
        top_k = min(data.get('top_k', 5), 50)
        min_similarity = data.get('min_similarity', 0.1)
        min_page = data.get('min_page')
        max_page = data.get('max_page')
        section_filter = data.get('section_filter')
        chapter_filter = data.get('chapter_filter')
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        if query_embedding:
            try:
                # Use enhanced filtered search function
                rpc_params = {
                    'query_embedding': query_embedding,
                    'match_threshold': min_similarity,
                    'match_count': top_k,
                    'min_page_filter': min_page,
                    'max_page_filter': max_page,
                    'section_filter': section_filter,
                    'chapter_filter': chapter_filter
                }
                
                result = supabase.rpc('search_embeddings_filtered', rpc_params).execute()
                search_results = result.data if result.data else []
                search_type = "ai_vector_filtered"
                
            except Exception as vector_error:
                logger.warning(f"⚠️ Filtered vector search failed: {vector_error}")
                
                # Fallback to basic vector search
                try:
                    rpc_params = {
                        'query_embedding': query_embedding,
                        'match_threshold': min_similarity,
                        'match_count': top_k
                    }
                    
                    result = supabase.rpc('search_embeddings', rpc_params).execute()
                    search_results = result.data if result.data else []
                    search_type = "ai_vector_basic"
                    
                    # Apply client-side filters
                    if min_page or max_page or section_filter or chapter_filter:
                        filtered_results = []
                        for record in search_results:
                            if min_page and record.get('page_number', 0) < min_page:
                                continue
                            if max_page and record.get('page_number', 99999) > max_page:
                                continue
                            if section_filter and record.get('section_title'):
                                if section_filter.lower() not in record['section_title'].lower():
                                    continue
                            if chapter_filter and record.get('chapter_title'):
                                if chapter_filter.lower() not in record['chapter_title'].lower():
                                    continue
                            filtered_results.append(record)
                        
                        search_results = filtered_results[:top_k]
                        search_type = "ai_vector_client_filtered"
                
                except Exception as basic_error:
                    logger.warning(f"⚠️ Basic vector search failed: {basic_error}")
                    search_results = []
                    search_type = "vector_failed"
        else:
            search_results = []
            search_type = "no_embedding"
        
        # Fallback to text search if vector search failed
        if not search_results:
            logger.info("🔄 Falling back to text search")
            
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
            if chapter_filter:
                query_builder = query_builder.ilike('chapter_title', f'%{chapter_filter}%')
            
            result = query_builder.limit(top_k).execute()
            search_results = result.data if result.data else []
            search_type = "text_search_fallback"
        
        # Format results
        formatted_results = []
        
        for i, record in enumerate(search_results):
            similarity = record.get('similarity', 0.0)
            
            formatted_result = {
                "rank": i + 1,
                "content": record.get('content', ''),
                "similarity": round(float(similarity), 4),
                "page_number": record.get('page_number'),
                "section_title": record.get('section_title'),
                "chapter_title": record.get('chapter_title'),
                "chunk_index": record.get('chunk_index'),
                "id": record.get('id')
            }
            
            formatted_results.append(formatted_result)
        
        return jsonify({
            "query": query,
            "search_type": search_type,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "filters_applied": {
                "min_page": min_page,
                "max_page": max_page,
                "section_filter": section_filter,
                "chapter_filter": chapter_filter,
                "min_similarity": min_similarity
            },
            "ai_powered": query_embedding is not None,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ AI semantic search error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/search/medical-specialty', methods=['POST'])
def medical_specialty_search():
    """Search for medical specialties using database function"""
    try:
        data = request.get_json() or {}
        keywords = data.get('keywords', [])
        limit = min(data.get('limit', 20), 50)
        
        if not keywords:
            return jsonify({
                "error": "Missing 'keywords' array",
                "example": {"keywords": ["asthma", "allergy"], "limit": 10}
            }), 400
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            result = supabase.rpc('find_medical_specialties', {
                'specialty_keywords': keywords,
                'page_limit': limit,
                'include_embeddings': True
            }).execute()
            
            return jsonify({
                "keywords": keywords,
                "results": result.data if result.data else [],
                "results_count": len(result.data) if result.data else 0,
                "status": "success"
            })
            
        except Exception as e:
            return jsonify({
                "error": "Medical specialty search failed",
                "message": str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"❌ Medical specialty search error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/admin/stats', methods=['GET'])
def admin_statistics():
    """Administrative statistics and system overview"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get comprehensive stats
        stats = {}
        
        # Basic counts
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        stats['total_documents'] = result.count if result.count else 0
        
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('embedding', 'null').execute()
        stats['documents_with_embeddings'] = result.count if result.count else 0
        
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('section_title', 'null').execute()
        stats['documents_with_sections'] = result.count if result.count else 0
        
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').not_.is_('page_number', 'null').execute()
        stats['documents_with_pages'] = result.count if result.count else 0
        
        # Calculate percentages
        total = stats['total_documents']
        if total > 0:
            stats['embedding_coverage_percent'] = round(stats['documents_with_embeddings'] / total * 100, 2)
            stats['section_coverage_percent'] = round(stats['documents_with_sections'] / total * 100, 2)
            stats['page_coverage_percent'] = round(stats['documents_with_pages'] / total * 100, 2)
        else:
            stats['embedding_coverage_percent'] = 0
            stats['section_coverage_percent'] = 0
            stats['page_coverage_percent'] = 0
        
        # System status
        stats['ai_search_ready'] = stats['documents_with_embeddings'] > 100
        stats['embedding_model_loaded'] = embedding_model is not None
        stats['production_ready'] = stats['ai_search_ready'] and stats['embedding_model_loaded']
        
        return jsonify({
            "system_statistics": stats,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Admin statistics error: {e}")
        return jsonify({
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health - System health check",
            "POST /search/ai - Advanced AI semantic search",
            "POST /search/medical-specialty - Medical specialty finder",
            "GET /admin/stats - System statistics"
        ]
    }), 404

if __name__ == '__main__':
    print("🚀 NELSON PEDIATRICS - PRODUCTION AI API")
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
        print("⚠️ Embedding model failed to load - will use text search fallback")
    
    print("🌐 Starting production API server...")
    print("\n📋 Production endpoints:")
    print("  • GET  /health - AI system health monitoring")
    print("  • POST /search/ai - Advanced semantic search with filters")
    print("  • POST /search/medical-specialty - Medical specialty finder")
    print("  • GET  /admin/stats - Administrative statistics")
    
    print("\n💡 Example AI search:")
    print("curl -X POST http://localhost:5000/search/ai \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print("    \"query\": \"asthma treatment in pediatric patients\",")
    print("    \"top_k\": 5,")
    print("    \"min_similarity\": 0.1,")
    print("    \"min_page\": 1100,")
    print("    \"max_page\": 1200,")
    print("    \"section_filter\": \"Treatment\"")
    print("  }'")
    
    print("\n🚀 Production server starting on http://localhost:5000")
    print("🤖 AI-powered medical search ready for NelsonGPT integration!")
    print("=" * 80)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Production mode
    )

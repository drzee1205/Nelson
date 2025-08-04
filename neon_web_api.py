#!/usr/bin/env python3
"""
Nelson Medical Knowledge API - Neon PostgreSQL Backend

This creates a REST API that connects directly to your Neon PostgreSQL database
for medical knowledge search. Perfect for production web applications.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os
from typing import List, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Neon Database Configuration
NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_4TWsIBXtja9b@ep-delicate-credit-a1h2uxg9-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Global database connection pool
db_pool = None

def get_db_connection():
    """Get database connection from pool"""
    try:
        conn = psycopg2.connect(
            NEON_CONNECTION_STRING,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return None

def initialize_database():
    """Initialize database connection and verify setup"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Test connection and get count
        cursor.execute("SELECT COUNT(*) as count FROM nelson_book_of_pediatrics;")
        result = cursor.fetchone()
        count = result['count'] if result else 0
        
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Connected to Neon database with {count} documents")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                "status": "unhealthy",
                "error": "Database connection failed"
            }), 500
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM nelson_book_of_pediatrics;")
        result = cursor.fetchone()
        count = result['count'] if result else 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "service": "Nelson Medical Knowledge API",
            "database": "Neon PostgreSQL",
            "documents": count
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
    Search the medical knowledge base using text search
    
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
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Use the search function we created
        search_sql = """
        SELECT * FROM search_medical_text(%s, %s);
        """
        
        cursor.execute(search_sql, (query, top_k))
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        
        for i, result in enumerate(results):
            formatted_result = {
                "rank": i + 1,
                "content": result['text'],
                "similarity": float(result['similarity_score']) if result['similarity_score'] else 0.0
            }
            
            if include_metadata:
                formatted_result["metadata"] = {
                    "id": result['id'],
                    "topic": result['topic'],
                    "source_file": result['source_file'],
                    "chunk_number": result['chunk_number']
                }
            
            formatted_results.append(formatted_result)
        
        cursor.close()
        conn.close()
        
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

@app.route('/search/topics', methods=['GET'])
def get_available_topics():
    """Get all available medical topics"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Get distinct topics
        cursor.execute("""
            SELECT DISTINCT topic, COUNT(*) as document_count
            FROM nelson_book_of_pediatrics 
            WHERE topic IS NOT NULL AND topic != ''
            GROUP BY topic
            ORDER BY topic;
        """)
        
        results = cursor.fetchall()
        
        topics = []
        topic_counts = {}
        
        for result in results:
            topic = result['topic']
            count = result['document_count']
            topics.append(topic)
            topic_counts[topic] = count
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "topics": topics,
            "count": len(topics),
            "topic_counts": topic_counts,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Topics error: {e}")
        return jsonify({
            "error": "Failed to get topics",
            "message": str(e)
        }), 500

@app.route('/search/topic/<topic_name>', methods=['POST'])
def search_by_topic(topic_name):
    """
    Search within a specific medical topic
    
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
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Use the topic search function
        cursor.execute("SELECT * FROM search_by_topic(%s, %s, %s);", (topic_name, query, top_k))
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        
        for i, result in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "content": result['text'],
                "metadata": {
                    "id": result['id'],
                    "topic": result['topic'],
                    "source_file": result['source_file'],
                    "chunk_number": result['chunk_number']
                }
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "query": query,
            "topic": topic_name,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Topic search error: {e}")
        return jsonify({
            "error": "Topic search failed",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Use the stats function
        cursor.execute("SELECT * FROM get_database_stats();")
        stats = cursor.fetchone()
        
        # Get top topics
        cursor.execute("""
            SELECT topic, COUNT(*) as count
            FROM nelson_book_of_pediatrics 
            WHERE topic IS NOT NULL AND topic != ''
            GROUP BY topic
            ORDER BY count DESC
            LIMIT 10;
        """)
        
        top_topics = {}
        for result in cursor.fetchall():
            top_topics[result['topic']] = result['count']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "total_documents": int(stats['total_documents']),
            "total_topics": int(stats['total_topics']),
            "total_sources": int(stats['total_sources']),
            "avg_text_length": float(stats['avg_text_length']),
            "latest_update": str(stats['latest_update']),
            "database_type": "Neon PostgreSQL",
            "top_topics": top_topics,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return jsonify({
            "error": "Failed to get stats",
            "message": str(e)
        }), 500

@app.route('/search/vector', methods=['POST'])
def vector_search():
    """
    Vector similarity search (requires embedding generation)
    
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
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Vector similarity search
        vector_search_sql = """
        SELECT 
            id,
            text,
            topic,
            source_file,
            chunk_number,
            1 - (embedding <=> %s::vector) as similarity
        FROM nelson_book_of_pediatrics 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        
        cursor.execute(vector_search_sql, (embedding, embedding, top_k))
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        
        for i, result in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "content": result['text'],
                "similarity": float(result['similarity']),
                "metadata": {
                    "id": result['id'],
                    "topic": result['topic'],
                    "source_file": result['source_file'],
                    "chunk_number": result['chunk_number']
                }
            })
        
        cursor.close()
        conn.close()
        
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /search", 
            "GET /search/topics",
            "POST /search/topic/<topic_name>",
            "POST /search/vector",
            "GET /stats"
        ]
    }), 404

if __name__ == '__main__':
    print("🏥 Nelson Medical Knowledge API - Neon PostgreSQL")
    print("=" * 60)
    
    # Initialize database connection
    if not initialize_database():
        print("❌ Failed to initialize database. Exiting.")
        exit(1)
    
    print("✅ Database initialized successfully")
    print("🌐 Starting API server...")
    print("\n📋 Available endpoints:")
    print("  • GET  /health - Health check")
    print("  • POST /search - Search medical knowledge (text)")
    print("  • GET  /search/topics - Get available topics")
    print("  • POST /search/topic/<topic> - Search by topic")
    print("  • POST /search/vector - Vector similarity search")
    print("  • GET  /stats - Database statistics")
    
    print("\n💡 Example usage:")
    print("curl -X POST http://localhost:5000/search \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"query\": \"asthma treatment\", \"top_k\": 3}'")
    
    print("\n🚀 Server starting on http://localhost:5000")
    print("🌐 Connected to Neon PostgreSQL database")
    print("=" * 60)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )


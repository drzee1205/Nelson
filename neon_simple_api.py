#!/usr/bin/env python3
"""
Nelson Medical Knowledge API - Neon PostgreSQL (Simplified)

This creates a REST API that connects directly to your Neon PostgreSQL database
using simple SQL queries for reliable medical knowledge search.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Neon Database Configuration
NEON_CONNECTION_STRING = os.getenv(
    'NEON_DATABASE_URL',
    'postgresql://neondb_owner:npg_4TWsIBXtja9b@ep-delicate-credit-a1h2uxg9-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
)

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(
            NEON_CONNECTION_STRING,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return None

def calculate_text_similarity(query, text):
    """Simple text similarity calculation"""
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
    Search the medical knowledge base using simple text matching
    
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
        
        # Simple text search using ILIKE for pattern matching
        search_sql = """
        SELECT 
            id,
            text,
            topic,
            source_file,
            chunk_number,
            character_count,
            created_at
        FROM nelson_book_of_pediatrics 
        WHERE text ILIKE %s 
        ORDER BY 
            CASE 
                WHEN text ILIKE %s THEN 3
                WHEN text ILIKE %s THEN 2
                ELSE 1
            END DESC,
            character_count DESC
        LIMIT %s;
        """
        
        # Create search patterns
        exact_pattern = f'%{query}%'
        words_pattern = '%' + '%'.join(query.split()) + '%'
        any_word_pattern = '%' + query.split()[0] + '%' if query.split() else exact_pattern
        
        cursor.execute(search_sql, (exact_pattern, exact_pattern, words_pattern, top_k * 2))
        results = cursor.fetchall()
        
        # Calculate similarity scores and format results
        formatted_results = []
        
        for result in results:
            # Calculate similarity score
            similarity = calculate_text_similarity(query, result['text'])
            
            # Skip results with very low similarity
            if similarity < 0.01:
                continue
            
            formatted_result = {
                "rank": len(formatted_results) + 1,
                "content": result['text'],
                "similarity": round(similarity, 4)
            }
            
            if include_metadata:
                formatted_result["metadata"] = {
                    "id": result['id'],
                    "topic": result['topic'],
                    "source_file": result['source_file'],
                    "chunk_number": result['chunk_number'],
                    "character_count": result['character_count'],
                    "created_at": str(result['created_at'])
                }
            
            formatted_results.append(formatted_result)
            
            # Stop when we have enough results
            if len(formatted_results) >= top_k:
                break
        
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
        
        # Get distinct topics with counts
        cursor.execute("""
            SELECT topic, COUNT(*) as document_count
            FROM nelson_book_of_pediatrics 
            WHERE topic IS NOT NULL AND topic != ''
            GROUP BY topic
            ORDER BY document_count DESC, topic;
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
        
        # Search within specific topic
        if query:
            search_sql = """
            SELECT 
                id, text, topic, source_file, chunk_number, character_count
            FROM nelson_book_of_pediatrics 
            WHERE topic ILIKE %s AND text ILIKE %s
            ORDER BY character_count DESC
            LIMIT %s;
            """
            cursor.execute(search_sql, (f'%{topic_name}%', f'%{query}%', top_k))
        else:
            search_sql = """
            SELECT 
                id, text, topic, source_file, chunk_number, character_count
            FROM nelson_book_of_pediatrics 
            WHERE topic ILIKE %s
            ORDER BY chunk_number, character_count DESC
            LIMIT %s;
            """
            cursor.execute(search_sql, (f'%{topic_name}%', top_k))
        
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        
        for i, result in enumerate(results):
            similarity = calculate_text_similarity(query, result['text']) if query else 1.0
            
            formatted_results.append({
                "rank": i + 1,
                "content": result['text'],
                "similarity": round(similarity, 4),
                "metadata": {
                    "id": result['id'],
                    "topic": result['topic'],
                    "source_file": result['source_file'],
                    "chunk_number": result['chunk_number'],
                    "character_count": result['character_count']
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
        
        # Get basic statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT topic) as total_topics,
                COUNT(DISTINCT source_file) as total_sources,
                AVG(character_count) as avg_text_length,
                MAX(created_at) as latest_update
            FROM nelson_book_of_pediatrics;
        """)
        
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
            "avg_text_length": float(stats['avg_text_length']) if stats['avg_text_length'] else 0,
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "POST /search", 
            "GET /search/topics",
            "POST /search/topic/<topic_name>",
            "GET /stats"
        ]
    }), 404

if __name__ == '__main__':
    print("🏥 Nelson Medical Knowledge API - Neon PostgreSQL (Simple)")
    print("=" * 70)
    
    # Test database connection
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to database. Exiting.")
        exit(1)
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM nelson_book_of_pediatrics;")
    result = cursor.fetchone()
    count = result['count'] if result else 0
    cursor.close()
    conn.close()
    
    print(f"✅ Connected to Neon database with {count} documents")
    print("🌐 Starting API server...")
    print("\n📋 Available endpoints:")
    print("  • GET  /health - Health check")
    print("  • POST /search - Search medical knowledge")
    print("  • GET  /search/topics - Get available topics")
    print("  • POST /search/topic/<topic> - Search by topic")
    print("  • GET  /stats - Database statistics")
    
    print("\n💡 Example usage:")
    print("curl -X POST http://localhost:5000/search \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"query\": \"asthma treatment\", \"top_k\": 3}'")
    
    print("\n🚀 Server starting on http://localhost:5000")
    print("🌐 Connected to Neon PostgreSQL database")
    print("=" * 70)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

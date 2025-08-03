#!/usr/bin/env python3
"""
Nelson Medical Knowledge API for Web Applications

This creates a REST API that your NelsonGPT web app can call to search
the medical knowledge base. Perfect for integration with any web framework.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.config import Settings
import logging
import os
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Global ChromaDB client and collection
client = None
collection = None

def initialize_chromadb():
    """Initialize ChromaDB connection"""
    global client, collection
    
    try:
        db_path = "./nelson_chromadb"
        
        if not os.path.exists(db_path):
            logger.error(f"❌ ChromaDB not found at {db_path}")
            logger.error("💡 Run 'python setup_chromadb.py' first to create the database")
            return False
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        collection = client.get_collection("nelson_pediatrics")
        
        # Test connection
        count = collection.count()
        logger.info(f"✅ Connected to ChromaDB with {count} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Nelson Medical Knowledge API",
        "database": "ChromaDB",
        "documents": collection.count() if collection else 0
    })

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
        
        if top_k > 20:
            top_k = 20  # Limit results for performance
        
        # Search ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                
                similarity = 1 - distance  # Convert distance to similarity
                
                result = {
                    "rank": i + 1,
                    "content": doc,
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4)
                }
                
                if include_metadata:
                    result["metadata"] = {
                        "topic": metadata.get('topic', 'Unknown'),
                        "source_file": metadata.get('source_file', 'Unknown'),
                        "chunk_number": metadata.get('chunk_number', 0),
                        "character_count": metadata.get('character_count', 0),
                        "created_at": metadata.get('created_at', '')
                    }
                
                formatted_results.append(result)
        
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
        # Get a sample of documents to extract topics
        results = collection.query(
            query_texts=["medical"],
            n_results=100
        )
        
        topics = set()
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                topic = metadata.get('topic', '').strip()
                if topic and topic != 'Unknown':
                    topics.add(topic)
        
        return jsonify({
            "topics": sorted(list(topics)),
            "count": len(topics),
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
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query'].strip()
        top_k = data.get('top_k', 5)
        
        # Search with topic filter
        results = collection.query(
            query_texts=[query],
            n_results=top_k * 3,  # Get more results to filter
            where={"topic": topic_name}
        )
        
        # Format results (same as general search)
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0][:top_k],  # Limit to requested count
                results['metadatas'][0][:top_k],
                results['distances'][0][:top_k]
            )):
                
                similarity = 1 - distance
                
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc,
                    "similarity": round(similarity, 4),
                    "metadata": {
                        "topic": metadata.get('topic', 'Unknown'),
                        "source_file": metadata.get('source_file', 'Unknown'),
                        "chunk_number": metadata.get('chunk_number', 0)
                    }
                })
        
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
        count = collection.count()
        
        # Get sample to analyze topics
        sample_results = collection.query(
            query_texts=["medical"],
            n_results=min(1000, count)
        )
        
        topics = {}
        sources = {}
        
        if sample_results['metadatas'] and sample_results['metadatas'][0]:
            for metadata in sample_results['metadatas'][0]:
                topic = metadata.get('topic', 'Unknown')
                source = metadata.get('source_file', 'Unknown')
                
                topics[topic] = topics.get(topic, 0) + 1
                sources[source] = sources.get(source, 0) + 1
        
        return jsonify({
            "total_documents": count,
            "vector_dimension": 384,
            "database_type": "ChromaDB",
            "topics_count": len(topics),
            "sources_count": len(sources),
            "top_topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]),
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
    print("🏥 Nelson Medical Knowledge API")
    print("=" * 50)
    
    # Initialize ChromaDB
    if not initialize_chromadb():
        print("❌ Failed to initialize database. Exiting.")
        exit(1)
    
    print("✅ Database initialized successfully")
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
    print("=" * 50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )


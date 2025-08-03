#!/usr/bin/env python3
"""
NelsonGPT Web App Integration Examples

Examples showing how to integrate ChromaDB with different web frameworks
for your NelsonGPT application.
"""

# ============================================================================
# OPTION 1: Direct ChromaDB Integration (Recommended for most cases)
# ============================================================================

def direct_chromadb_integration():
    """
    Direct integration - embed ChromaDB directly in your web app
    Best for: Single-server deployments, simple architectures
    """
    
    # Flask Example
    flask_example = '''
from flask import Flask, request, jsonify
import chromadb
from chromadb.config import Settings

app = Flask(__name__)

# Initialize ChromaDB once at startup
client = chromadb.PersistentClient(
    path="./nelson_chromadb",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("nelson_pediatrics")

@app.route('/api/search', methods=['POST'])
def search_medical():
    data = request.get_json()
    query = data.get('query', '')
    
    # Search the medical knowledge base
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    # Format for your NelsonGPT frontend
    formatted_results = []
    if results['documents'] and results['documents'][0]:
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted_results.append({
                'content': doc,
                'topic': metadata.get('topic', 'Unknown'),
                'source': metadata.get('source_file', 'Unknown'),
                'similarity': 1 - distance,
                'relevance_score': round((1 - distance) * 100, 1)
            })
    
    return jsonify({
        'query': query,
        'results': formatted_results,
        'total_found': len(formatted_results)
    })

if __name__ == '__main__':
    app.run(debug=True)
'''

    # FastAPI Example
    fastapi_example = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings

app = FastAPI(title="Nelson Medical Knowledge API")

# Initialize ChromaDB
client = chromadb.PersistentClient(
    path="./nelson_chromadb",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("nelson_pediatrics")

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

class SearchResult(BaseModel):
    content: str
    topic: str
    source: str
    similarity: float
    relevance_score: float

@app.post("/search", response_model=list[SearchResult])
async def search_medical_knowledge(request: SearchRequest):
    try:
        results = collection.query(
            query_texts=[request.query],
            n_results=request.max_results
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                formatted_results.append(SearchResult(
                    content=doc,
                    topic=metadata.get('topic', 'Unknown'),
                    source=metadata.get('source_file', 'Unknown'),
                    similarity=1 - distance,
                    relevance_score=round((1 - distance) * 100, 1)
                ))
        
        return formatted_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

    return flask_example, fastapi_example

# ============================================================================
# OPTION 2: Microservice API (For scalable deployments)
# ============================================================================

def microservice_integration():
    """
    Microservice approach - separate API service for medical knowledge
    Best for: Multi-server deployments, microservice architectures
    """
    
    # Your NelsonGPT app calls the API
    frontend_integration = '''
// JavaScript/TypeScript integration for your NelsonGPT frontend

class NelsonMedicalAPI {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async searchMedicalKnowledge(query, maxResults = 5) {
        try {
            const response = await fetch(`${this.baseUrl}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: maxResults,
                    include_metadata: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data.results;
            
        } catch (error) {
            console.error('Medical search error:', error);
            throw error;
        }
    }
    
    async getAvailableTopics() {
        try {
            const response = await fetch(`${this.baseUrl}/search/topics`);
            const data = await response.json();
            return data.topics;
        } catch (error) {
            console.error('Topics error:', error);
            throw error;
        }
    }
    
    async searchByTopic(topic, query, maxResults = 5) {
        try {
            const response = await fetch(`${this.baseUrl}/search/topic/${topic}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: maxResults
                })
            });
            
            const data = await response.json();
            return data.results;
        } catch (error) {
            console.error('Topic search error:', error);
            throw error;
        }
    }
}

// Usage in your NelsonGPT app
const medicalAPI = new NelsonMedicalAPI();

// Example: Enhanced chat with medical context
async function enhanceResponseWithMedicalContext(userQuery) {
    try {
        // Search medical knowledge base
        const medicalResults = await medicalAPI.searchMedicalKnowledge(userQuery, 3);
        
        // Add medical context to your GPT prompt
        const medicalContext = medicalResults
            .map(result => `${result.metadata.topic}: ${result.content}`)
            .join('\\n\\n');
        
        const enhancedPrompt = `
Medical Context from Nelson Pediatrics:
${medicalContext}

User Question: ${userQuery}

Please provide a response based on the medical context above.
`;
        
        // Send to your GPT model with enhanced context
        return await sendToGPT(enhancedPrompt);
        
    } catch (error) {
        console.error('Error enhancing with medical context:', error);
        // Fallback to regular GPT response
        return await sendToGPT(userQuery);
    }
}
'''

    return frontend_integration

# ============================================================================
# OPTION 3: Hybrid Integration (Best of both worlds)
# ============================================================================

def hybrid_integration():
    """
    Hybrid approach - ChromaDB embedded with API endpoints
    Best for: Flexible deployments, gradual scaling
    """
    
    hybrid_example = '''
from flask import Flask, request, jsonify, render_template
import chromadb
from chromadb.config import Settings
import openai  # or your preferred LLM library

app = Flask(__name__)

# Initialize ChromaDB
client = chromadb.PersistentClient(
    path="./nelson_chromadb",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("nelson_pediatrics")

class NelsonGPTService:
    def __init__(self):
        self.medical_db = collection
        # Initialize your LLM here
        
    def get_medical_context(self, query, max_results=3):
        """Get relevant medical context for a query"""
        results = self.medical_db.query(
            query_texts=[query],
            n_results=max_results
        )
        
        context = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                context.append({
                    'content': doc,
                    'topic': metadata.get('topic', 'Unknown'),
                    'source': metadata.get('source_file', 'Unknown')
                })
        
        return context
    
    def generate_enhanced_response(self, user_query):
        """Generate GPT response enhanced with medical knowledge"""
        
        # Get medical context
        medical_context = self.get_medical_context(user_query)
        
        # Build enhanced prompt
        context_text = "\\n\\n".join([
            f"Topic: {ctx['topic']}\\nContent: {ctx['content']}"
            for ctx in medical_context
        ])
        
        enhanced_prompt = f"""
You are NelsonGPT, a pediatric medical assistant. Use the following medical knowledge from Nelson Textbook of Pediatrics to inform your response:

Medical Knowledge:
{context_text}

User Question: {user_query}

Provide a helpful, accurate response based on the medical knowledge above. Always recommend consulting with healthcare professionals for medical decisions.
"""
        
        # Generate response with your LLM
        # response = openai.ChatCompletion.create(...)
        
        return {
            'response': 'Your GPT response here',
            'medical_sources': medical_context,
            'query': user_query
        }

# Initialize service
nelson_service = NelsonGPTService()

@app.route('/')
def home():
    return render_template('chat.html')  # Your NelsonGPT chat interface

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    # Generate enhanced response
    result = nelson_service.generate_enhanced_response(user_message)
    
    return jsonify(result)

@app.route('/api/search', methods=['POST'])
def search():
    """Direct medical knowledge search"""
    data = request.get_json()
    query = data.get('query', '')
    
    context = nelson_service.get_medical_context(query, max_results=5)
    
    return jsonify({
        'query': query,
        'results': context
    })

if __name__ == '__main__':
    app.run(debug=True)
'''

    return hybrid_example

# ============================================================================
# DEPLOYMENT CONFIGURATIONS
# ============================================================================

def deployment_configs():
    """Different deployment configurations for production"""
    
    # Docker configuration
    dockerfile = '''
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Copy ChromaDB database
COPY nelson_chromadb ./nelson_chromadb

EXPOSE 5000

CMD ["python", "nelson_web_api.py"]
'''

    # Docker Compose for full stack
    docker_compose = '''
version: '3.8'

services:
  nelson-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./nelson_chromadb:/app/nelson_chromadb
    environment:
      - FLASK_ENV=production
    
  nelson-frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - nelson-api
    environment:
      - REACT_APP_API_URL=http://nelson-api:5000
'''

    # Requirements file
    requirements = '''
flask==2.3.3
flask-cors==4.0.0
chromadb==1.0.15
sentence-transformers==2.2.2
numpy==1.24.3
requests==2.31.0
gunicorn==21.2.0  # For production deployment
'''

    return dockerfile, docker_compose, requirements

if __name__ == "__main__":
    print("🌐 NelsonGPT Web Integration Guide")
    print("=" * 50)
    print("Choose your integration approach:")
    print("1. Direct ChromaDB Integration (Simplest)")
    print("2. Microservice API (Most Scalable)")
    print("3. Hybrid Approach (Most Flexible)")
    print("\nRun 'python nelson_web_api.py' to start the API server!")


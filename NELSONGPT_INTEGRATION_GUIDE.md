# 🌐 NelsonGPT Web App Integration Guide

## 🎯 **3 Integration Options for Your NelsonGPT Web App**

### Option 1: **REST API Integration** ⭐ **RECOMMENDED**
**Best for**: Most web apps, easy to implement, scalable

```bash
# 1. Start the medical knowledge API
python nelson_web_api.py

# 2. Your NelsonGPT app calls the API
# API runs on: http://localhost:5000
```

### Option 2: **Direct ChromaDB Integration**
**Best for**: Simple deployments, single-server apps

### Option 3: **Hybrid Integration**
**Best for**: Advanced use cases, custom implementations

---

## 🚀 **Quick Start: REST API Integration**

### Step 1: Start the Medical Knowledge API

```bash
# Install dependencies
pip install -r requirements_web.txt

# Start the API server
python nelson_web_api.py
```

**API will be available at**: `http://localhost:5000`

### Step 2: Integrate with Your NelsonGPT Frontend

#### JavaScript/TypeScript Integration:

```javascript
class NelsonMedicalAPI {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async searchMedicalKnowledge(query, maxResults = 5) {
        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: maxResults,
                include_metadata: true
            })
        });
        
        const data = await response.json();
        return data.results;
    }
    
    async getAvailableTopics() {
        const response = await fetch(`${this.baseUrl}/search/topics`);
        const data = await response.json();
        return data.topics;
    }
}

// Usage in your NelsonGPT app
const medicalAPI = new NelsonMedicalAPI();

// Enhanced chat with medical context
async function enhanceResponseWithMedicalContext(userQuery) {
    try {
        // Get medical context from Nelson Pediatrics
        const medicalResults = await medicalAPI.searchMedicalKnowledge(userQuery, 3);
        
        // Build enhanced prompt for your GPT model
        const medicalContext = medicalResults
            .map(result => `${result.metadata.topic}: ${result.content}`)
            .join('\\n\\n');
        
        const enhancedPrompt = `
Medical Context from Nelson Pediatrics:
${medicalContext}

User Question: ${userQuery}

Please provide a response based on the medical context above.
`;
        
        // Send to your GPT model with medical context
        return await sendToGPT(enhancedPrompt);
        
    } catch (error) {
        console.error('Error enhancing with medical context:', error);
        return await sendToGPT(userQuery); // Fallback
    }
}
```

#### Python Backend Integration:

```python
import requests

class NelsonMedicalAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def search_medical_knowledge(self, query, max_results=5):
        response = requests.post(f"{self.base_url}/search", json={
            "query": query,
            "top_k": max_results,
            "include_metadata": True
        })
        return response.json()["results"]
    
    def get_available_topics(self):
        response = requests.get(f"{self.base_url}/search/topics")
        return response.json()["topics"]

# Usage in your NelsonGPT backend
medical_api = NelsonMedicalAPI()

def enhance_gpt_with_medical_context(user_query):
    # Get medical context
    medical_results = medical_api.search_medical_knowledge(user_query, 3)
    
    # Build enhanced prompt
    medical_context = "\\n\\n".join([
        f"{result['metadata']['topic']}: {result['content']}"
        for result in medical_results
    ])
    
    enhanced_prompt = f"""
Medical Knowledge from Nelson Pediatrics:
{medical_context}

User Question: {user_query}

Provide a helpful medical response based on the context above.
"""
    
    # Send to your GPT model
    return send_to_gpt(enhanced_prompt)
```

---

## 📋 **Available API Endpoints**

### 🔍 **Search Medical Knowledge**
```bash
POST /search
Content-Type: application/json

{
    "query": "asthma treatment in children",
    "top_k": 5,
    "include_metadata": true
}
```

**Response:**
```json
{
    "query": "asthma treatment in children",
    "results_count": 5,
    "results": [
        {
            "rank": 1,
            "content": "Children with mild persistent asthma are at Treatment Step 2...",
            "similarity": 0.483,
            "metadata": {
                "topic": "Allergic Disorder",
                "source_file": "Allergic Disorder.txt",
                "chunk_number": 45
            }
        }
    ]
}
```

### 📚 **Get Available Topics**
```bash
GET /search/topics
```

**Response:**
```json
{
    "topics": [
        "Allergic Disorder",
        "The Cardiovascular System", 
        "Respiratory System",
        "Nervous System"
    ],
    "count": 23
}
```

### 🎯 **Search by Topic**
```bash
POST /search/topic/Allergic%20Disorder
Content-Type: application/json

{
    "query": "asthma treatment",
    "top_k": 3
}
```

### 📊 **Database Statistics**
```bash
GET /stats
```

### ❤️ **Health Check**
```bash
GET /health
```

---

## 🏗️ **Architecture Options**

### **Option A: Microservice Architecture** (Recommended)
```
┌─────────────────┐    HTTP API    ┌──────────────────┐
│   NelsonGPT     │ ──────────────► │ Medical Knowledge│
│   Web App       │                │ API (Port 5000)  │
│   (Frontend)    │                │                  │
└─────────────────┘                └──────────────────┘
                                           │
                                           ▼
                                   ┌──────────────────┐
                                   │   ChromaDB       │
                                   │   (Local)        │
                                   └──────────────────┘
```

**Benefits:**
- ✅ Scalable and maintainable
- ✅ Language-agnostic frontend
- ✅ Easy to deploy and update
- ✅ Can handle multiple clients

### **Option B: Direct Integration**
```
┌─────────────────────────────────┐
│        NelsonGPT Web App        │
│                                 │
│  ┌─────────────┐ ┌─────────────┐│
│  │  Frontend   │ │  Backend    ││
│  │             │ │             ││
│  └─────────────┘ └─────────────┘│
│                   │             │
│                   ▼             │
│            ┌─────────────┐      │
│            │  ChromaDB   │      │
│            │  (Embedded) │      │
│            └─────────────┘      │
└─────────────────────────────────┘
```

**Benefits:**
- ✅ Simpler deployment
- ✅ Lower latency
- ✅ No network dependencies

---

## 🚀 **Production Deployment**

### **Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt

COPY . .
COPY nelson_chromadb ./nelson_chromadb

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "nelson_web_api:app"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  nelson-medical-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./nelson_chromadb:/app/nelson_chromadb
    environment:
      - FLASK_ENV=production
  
  nelson-gpt-app:
    build: ./your-nelsongpt-app
    ports:
      - "3000:3000"
    depends_on:
      - nelson-medical-api
    environment:
      - MEDICAL_API_URL=http://nelson-medical-api:5000
```

### **Cloud Deployment Options**

1. **Heroku**: Simple deployment with git push
2. **AWS ECS**: Container-based deployment
3. **Google Cloud Run**: Serverless container deployment
4. **DigitalOcean App Platform**: Easy container deployment

---

## 💡 **Integration Examples**

### **React/Next.js Integration**

```jsx
import { useState, useEffect } from 'react';

function MedicalSearchComponent() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const searchMedical = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: 5 })
            });
            
            const data = await response.json();
            setResults(data.results);
        } catch (error) {
            console.error('Search error:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div>
            <input 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search medical knowledge..."
            />
            <button onClick={searchMedical} disabled={loading}>
                {loading ? 'Searching...' : 'Search'}
            </button>
            
            {results.map((result, index) => (
                <div key={index} className="result">
                    <h4>{result.metadata.topic}</h4>
                    <p>{result.content}</p>
                    <small>Similarity: {(result.similarity * 100).toFixed(1)}%</small>
                </div>
            ))}
        </div>
    );
}
```

### **Vue.js Integration**

```vue
<template>
  <div>
    <input v-model="query" placeholder="Search medical knowledge..." />
    <button @click="searchMedical" :disabled="loading">
      {{ loading ? 'Searching...' : 'Search' }}
    </button>
    
    <div v-for="result in results" :key="result.rank" class="result">
      <h4>{{ result.metadata.topic }}</h4>
      <p>{{ result.content }}</p>
      <small>Similarity: {{ (result.similarity * 100).toFixed(1) }}%</small>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      query: '',
      results: [],
      loading: false
    };
  },
  methods: {
    async searchMedical() {
      this.loading = true;
      try {
        const response = await fetch('http://localhost:5000/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: this.query, top_k: 5 })
        });
        
        const data = await response.json();
        this.results = data.results;
      } catch (error) {
        console.error('Search error:', error);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>
```

---

## 🔧 **Customization Options**

### **Custom Scoring**
```python
# Add custom relevance scoring
def custom_relevance_score(similarity, metadata):
    base_score = similarity * 100
    
    # Boost certain topics
    topic_boost = {
        'Allergic Disorder': 1.2,
        'The Cardiovascular System': 1.1,
        'Emergency Medicine': 1.3
    }
    
    topic = metadata.get('topic', '')
    boost = topic_boost.get(topic, 1.0)
    
    return min(base_score * boost, 100)
```

### **Response Filtering**
```python
# Filter results by minimum similarity
def filter_results(results, min_similarity=0.3):
    return [r for r in results if r['similarity'] >= min_similarity]
```

---

## 🎉 **You're Ready!**

Your medical knowledge base is now ready for web integration! Choose the approach that best fits your NelsonGPT architecture:

1. **🚀 Start with REST API** (recommended for most cases)
2. **🔧 Customize as needed** for your specific requirements  
3. **📈 Scale when ready** with production deployment

**Your NelsonGPT app now has access to 15,339 medical text chunks from Nelson Pediatrics!** 🏥⚡


# 🌐 NelsonGPT Platform Integration Guide

## 🎯 **Complete Integration Package for Next.js, React Native & Java**

This guide provides production-ready code examples for integrating your NelsonGPT application with the Nelson Pediatrics medical knowledge base across different platforms.

---

## 🚀 **Quick Start**

### 1. **Start the Medical Knowledge API**
```bash
# Install dependencies
pip install -r requirements_web.txt

# Start the Nelson Medical API
python nelson_web_api.py

# API will be available at: http://localhost:5000
```

### 2. **Choose Your Platform Integration**

| Platform | Use Case | Integration Method |
|----------|----------|-------------------|
| **Next.js** | Web applications, SSR | API Routes + React Hooks |
| **React Native** | Mobile applications | Service Layer + Hooks |
| **Java Spring Boot** | Enterprise backends | RestTemplate + DTOs |

---

## 📱 **Next.js Integration**

### **Features:**
- ✅ Server-side API routes for medical search
- ✅ TypeScript support with full type safety
- ✅ React hooks for easy component integration
- ✅ SEO-friendly server-side rendering
- ✅ Automatic medical context enhancement

### **Quick Setup:**
```bash
# 1. Copy the Next.js example files
cp -r nextjs_example/* your-nextjs-app/

# 2. Install dependencies
cd your-nextjs-app
npm install

# 3. Set environment variable
echo "MEDICAL_API_URL=http://localhost:5000" > .env.local

# 4. Start development server
npm run dev
```

### **Usage Example:**
```typescript
// In your Next.js component
import { useMedicalSearch } from '../hooks/useMedicalSearch';

function ChatComponent() {
  const { enhanceWithMedicalContext } = useMedicalSearch();
  
  const handleUserMessage = async (userQuery: string) => {
    // Enhance query with medical context
    const enhancedPrompt = await enhanceWithMedicalContext(userQuery);
    
    // Send to your GPT model
    const response = await sendToGPT(enhancedPrompt);
    return response;
  };
  
  // ... rest of component
}
```

### **API Endpoints:**
- `POST /api/medical-search` - Search medical knowledge
- `GET /api/medical-topics` - Get available topics
- `POST /api/medical-enhance` - Enhance query with context

---

## 📱 **React Native Integration**

### **Features:**
- ✅ Cross-platform iOS/Android support
- ✅ TypeScript service layer
- ✅ React hooks for state management
- ✅ Offline-capable with caching
- ✅ Native performance optimizations

### **Quick Setup:**
```bash
# 1. Copy the React Native service
cp react_native_example/MedicalSearchService.ts your-rn-app/src/services/

# 2. Install dependencies (if needed)
npm install

# 3. Import and use in your components
```

### **Usage Example:**
```typescript
// In your React Native component
import React, { useState } from 'react';
import { useMedicalSearch } from '../hooks/useMedicalSearch';

export const ChatScreen = () => {
  const { enhanceWithMedicalContext } = useMedicalSearch();
  
  const handleUserMessage = async (userQuery: string) => {
    try {
      // Get medical context
      const enhancedPrompt = await enhanceWithMedicalContext(userQuery);
      
      // Send to your GPT API
      const response = await fetch('your-gpt-api', {
        method: 'POST',
        body: JSON.stringify({ prompt: enhancedPrompt })
      });
      
      return await response.json();
    } catch (error) {
      console.error('Chat error:', error);
    }
  };
  
  // ... rest of component
};
```

### **Service Methods:**
- `searchMedicalKnowledge(query, maxResults)` - Search medical database
- `getAvailableTopics()` - Get all medical topics
- `searchByTopic(topic, query)` - Topic-specific search
- `enhanceWithMedicalContext(query)` - Enhance for GPT

---

## ☕ **Java Spring Boot Integration**

### **Features:**
- ✅ Enterprise-ready Spring Boot service
- ✅ Full type safety with DTOs
- ✅ Comprehensive error handling
- ✅ Logging and monitoring support
- ✅ Production-ready configuration

### **Quick Setup:**
```bash
# 1. Copy Java files to your Spring Boot project
cp java_example/* src/main/java/com/yourpackage/

# 2. Add to application.properties
echo "medical.api.base-url=http://localhost:5000" >> application.properties

# 3. Build and run
./mvnw spring-boot:run
```

### **Usage Example:**
```java
// In your Spring Boot controller
@RestController
public class ChatController {
    
    @Autowired
    private MedicalSearchService medicalSearchService;
    
    @PostMapping("/api/chat")
    public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest request) {
        try {
            // Enhance user query with medical context
            String enhancedPrompt = medicalSearchService.enhanceWithMedicalContext(
                request.getMessage()
            );
            
            // Send to your GPT service
            String gptResponse = gptService.generateResponse(enhancedPrompt);
            
            return ResponseEntity.ok(new ChatResponse(gptResponse));
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}
```

### **Service Methods:**
- `searchMedicalKnowledge(query, maxResults)` - Search medical database
- `getAvailableTopics()` - Get all medical topics  
- `searchByTopic(topic, query, maxResults)` - Topic-specific search
- `enhanceWithMedicalContext(query)` - Enhance for GPT
- `checkHealth()` - API health check

---

## 🔧 **Configuration Options**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDICAL_API_URL` | `http://localhost:5000` | Nelson Medical API base URL |
| `MEDICAL_API_TIMEOUT` | `30000` | Request timeout in milliseconds |
| `MEDICAL_MAX_RESULTS` | `20` | Maximum search results per query |

### **Next.js Configuration**
```javascript
// next.config.js
module.exports = {
  env: {
    MEDICAL_API_URL: process.env.MEDICAL_API_URL,
  },
  async rewrites() {
    return [
      {
        source: '/api/medical/:path*',
        destination: `${process.env.MEDICAL_API_URL}/:path*`,
      },
    ];
  },
};
```

### **React Native Configuration**
```typescript
// config.ts
export const config = {
  medicalApiUrl: __DEV__ 
    ? 'http://localhost:5000' 
    : 'https://your-production-api.com',
  timeout: 30000,
  maxResults: 10,
};
```

### **Java Configuration**
```properties
# application.properties
medical.api.base-url=http://localhost:5000
medical.api.timeout=30000
medical.api.max-results=20

# Logging
logging.level.com.nelsongpt.service.MedicalSearchService=INFO
```

---

## 🚀 **Production Deployment**

### **Docker Deployment**

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
  
  nelsongpt-nextjs:
    build: ./nextjs-app
    ports:
      - "3000:3000"
    depends_on:
      - nelson-medical-api
    environment:
      - MEDICAL_API_URL=http://nelson-medical-api:5000
  
  nelsongpt-java:
    build: ./java-app
    ports:
      - "8080:8080"
    depends_on:
      - nelson-medical-api
    environment:
      - MEDICAL_API_BASE_URL=http://nelson-medical-api:5000
```

### **Cloud Deployment Options**

1. **Heroku**: Simple git-based deployment
2. **AWS ECS**: Container orchestration
3. **Google Cloud Run**: Serverless containers
4. **Azure Container Instances**: Managed containers
5. **DigitalOcean App Platform**: Easy container deployment

---

## 📊 **Performance Optimization**

### **Caching Strategies**

```typescript
// Next.js - API route caching
export default async function handler(req, res) {
  // Cache medical search results for 1 hour
  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate');
  
  // ... search logic
}
```

```java
// Java - Service-level caching
@Service
@EnableCaching
public class MedicalSearchService {
    
    @Cacheable(value = "medicalSearch", key = "#query")
    public MedicalSearchResponse searchMedicalKnowledge(String query, int maxResults) {
        // ... search logic
    }
}
```

### **Request Optimization**

- **Debouncing**: Delay search requests to avoid excessive API calls
- **Pagination**: Limit results and implement pagination
- **Compression**: Enable gzip compression for API responses
- **Connection Pooling**: Reuse HTTP connections

---

## 🧪 **Testing Examples**

### **Next.js Testing**
```typescript
// __tests__/medical-search.test.ts
import { createMocks } from 'node-mocks-http';
import handler from '../pages/api/medical-search';

describe('/api/medical-search', () => {
  it('should return medical search results', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      body: { query: 'asthma treatment', maxResults: 5 },
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(200);
    const data = JSON.parse(res._getData());
    expect(data.success).toBe(true);
    expect(data.results).toBeDefined();
  });
});
```

### **Java Testing**
```java
// MedicalSearchServiceTest.java
@SpringBootTest
class MedicalSearchServiceTest {
    
    @Autowired
    private MedicalSearchService medicalSearchService;
    
    @Test
    void shouldSearchMedicalKnowledge() {
        // Given
        String query = "asthma treatment";
        int maxResults = 5;
        
        // When
        MedicalSearchResponse response = medicalSearchService
            .searchMedicalKnowledge(query, maxResults);
        
        // Then
        assertThat(response).isNotNull();
        assertThat(response.getResults()).isNotEmpty();
        assertThat(response.getQuery()).isEqualTo(query);
    }
}
```

---

## 🎯 **Integration Summary**

| Feature | Next.js | React Native | Java |
|---------|---------|--------------|------|
| **Medical Search** | ✅ | ✅ | ✅ |
| **Topic Filtering** | ✅ | ✅ | ✅ |
| **Context Enhancement** | ✅ | ✅ | ✅ |
| **Type Safety** | ✅ | ✅ | ✅ |
| **Error Handling** | ✅ | ✅ | ✅ |
| **Caching** | ✅ | ✅ | ✅ |
| **Production Ready** | ✅ | ✅ | ✅ |

---

## 🆘 **Troubleshooting**

### **Common Issues**

1. **API Connection Failed**
   ```bash
   # Check if Nelson Medical API is running
   curl http://localhost:5000/health
   ```

2. **CORS Issues (Browser)**
   ```javascript
   // Ensure CORS is enabled in nelson_web_api.py
   from flask_cors import CORS
   CORS(app)
   ```

3. **Timeout Errors**
   ```typescript
   // Increase timeout in your HTTP client
   const response = await fetch(url, { 
     signal: AbortSignal.timeout(30000) 
   });
   ```

### **Debug Mode**

```bash
# Enable debug logging for Nelson Medical API
DEBUG=1 python nelson_web_api.py
```

---

## 🎉 **You're Ready!**

Your NelsonGPT application can now leverage the complete Nelson Pediatrics medical knowledge base across all platforms:

- **📱 Mobile**: React Native integration ready
- **🌐 Web**: Next.js integration ready  
- **🏢 Enterprise**: Java Spring Boot integration ready

**All platforms now have access to 15,339 medical text chunks for enhanced, medically-informed responses! 🏥⚡**


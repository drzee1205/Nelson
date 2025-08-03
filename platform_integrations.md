# 🌐 NelsonGPT Platform-Specific Integrations

## 🚀 **Next.js Integration**

### **Option 1: API Routes (Server-Side)**

```typescript
// pages/api/medical-search.ts or app/api/medical-search/route.ts
import { NextApiRequest, NextApiResponse } from 'next';

interface MedicalSearchRequest {
  query: string;
  maxResults?: number;
}

interface MedicalResult {
  content: string;
  similarity: number;
  metadata: {
    topic: string;
    source_file: string;
    chunk_number: number;
  };
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query, maxResults = 5 }: MedicalSearchRequest = req.body;

  try {
    // Call your Nelson Medical API
    const response = await fetch('http://localhost:5000/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: maxResults,
        include_metadata: true
      })
    });

    if (!response.ok) {
      throw new Error(`Medical API error: ${response.status}`);
    }

    const data = await response.json();
    
    res.status(200).json({
      success: true,
      results: data.results,
      query: data.query,
      resultsCount: data.results_count
    });

  } catch (error) {
    console.error('Medical search error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to search medical knowledge'
    });
  }
}
```

### **Option 2: Client-Side Hook**

```typescript
// hooks/useMedicalSearch.ts
import { useState, useCallback } from 'react';

interface MedicalResult {
  content: string;
  similarity: number;
  metadata: {
    topic: string;
    source_file: string;
  };
}

interface UseMedicalSearchReturn {
  results: MedicalResult[];
  loading: boolean;
  error: string | null;
  searchMedical: (query: string, maxResults?: number) => Promise<void>;
  enhanceWithMedicalContext: (userQuery: string) => Promise<string>;
}

export const useMedicalSearch = (): UseMedicalSearchReturn => {
  const [results, setResults] = useState<MedicalResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const searchMedical = useCallback(async (query: string, maxResults = 5) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/medical-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, maxResults })
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setResults(data.results);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const enhanceWithMedicalContext = useCallback(async (userQuery: string): Promise<string> => {
    try {
      // Get medical context
      const response = await fetch('/api/medical-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, maxResults: 3 })
      });

      const data = await response.json();
      
      if (!data.success || !data.results.length) {
        return userQuery; // Return original query if no medical context
      }

      // Build enhanced prompt
      const medicalContext = data.results
        .map((result: MedicalResult) => 
          `${result.metadata.topic}: ${result.content}`
        )
        .join('\n\n');

      return `
Medical Context from Nelson Pediatrics:
${medicalContext}

User Question: ${userQuery}

Please provide a response based on the medical context above.
`;

    } catch (error) {
      console.error('Error enhancing with medical context:', error);
      return userQuery; // Fallback to original query
    }
  }, []);

  return {
    results,
    loading,
    error,
    searchMedical,
    enhanceWithMedicalContext
  };
};
```

### **Option 3: Next.js Component Example**

```tsx
// components/MedicalSearchComponent.tsx
'use client';

import React, { useState } from 'react';
import { useMedicalSearch } from '../hooks/useMedicalSearch';

interface Props {
  onResultSelect?: (result: any) => void;
}

export const MedicalSearchComponent: React.FC<Props> = ({ onResultSelect }) => {
  const [query, setQuery] = useState('');
  const { results, loading, error, searchMedical } = useMedicalSearch();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      await searchMedical(query);
    }
  };

  return (
    <div className="medical-search">
      <form onSubmit={handleSearch} className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search Nelson Pediatrics..."
            className="flex-1 px-3 py-2 border rounded-lg"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {error && (
        <div className="text-red-500 mb-4">
          Error: {error}
        </div>
      )}

      <div className="results">
        {results.map((result, index) => (
          <div
            key={index}
            className="border rounded-lg p-4 mb-3 cursor-pointer hover:bg-gray-50"
            onClick={() => onResultSelect?.(result)}
          >
            <div className="flex justify-between items-start mb-2">
              <h4 className="font-semibold text-blue-600">
                {result.metadata.topic}
              </h4>
              <span className="text-sm text-gray-500">
                {(result.similarity * 100).toFixed(1)}% match
              </span>
            </div>
            <p className="text-gray-700 text-sm">
              {result.content.substring(0, 200)}...
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
```

---

## 📱 **React Native Integration**

### **Medical Search Service**

```typescript
// services/MedicalSearchService.ts
interface MedicalResult {
  content: string;
  similarity: number;
  metadata: {
    topic: string;
    source_file: string;
    chunk_number: number;
  };
}

interface MedicalSearchResponse {
  query: string;
  results_count: number;
  results: MedicalResult[];
  status: string;
}

class MedicalSearchService {
  private baseUrl: string;

  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }

  async searchMedicalKnowledge(
    query: string, 
    maxResults = 5
  ): Promise<MedicalResult[]> {
    try {
      const response = await fetch(`${this.baseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          top_k: maxResults,
          include_metadata: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: MedicalSearchResponse = await response.json();
      return data.results;

    } catch (error) {
      console.error('Medical search error:', error);
      throw new Error('Failed to search medical knowledge');
    }
  }

  async getAvailableTopics(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/search/topics`);
      const data = await response.json();
      return data.topics;
    } catch (error) {
      console.error('Topics error:', error);
      throw new Error('Failed to get topics');
    }
  }

  async searchByTopic(
    topic: string, 
    query: string, 
    maxResults = 5
  ): Promise<MedicalResult[]> {
    try {
      const response = await fetch(`${this.baseUrl}/search/topic/${encodeURIComponent(topic)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          top_k: maxResults
        })
      });

      const data = await response.json();
      return data.results;
    } catch (error) {
      console.error('Topic search error:', error);
      throw new Error('Failed to search by topic');
    }
  }
}

export const medicalSearchService = new MedicalSearchService();
```

### **React Native Hook**

```typescript
// hooks/useMedicalSearch.ts
import { useState, useCallback } from 'react';
import { medicalSearchService } from '../services/MedicalSearchService';

export const useMedicalSearch = () => {
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const searchMedical = useCallback(async (query: string, maxResults = 5) => {
    setLoading(true);
    setError(null);

    try {
      const searchResults = await medicalSearchService.searchMedicalKnowledge(query, maxResults);
      setResults(searchResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const enhanceWithMedicalContext = useCallback(async (userQuery: string): Promise<string> => {
    try {
      const medicalResults = await medicalSearchService.searchMedicalKnowledge(userQuery, 3);
      
      if (!medicalResults.length) {
        return userQuery;
      }

      const medicalContext = medicalResults
        .map(result => `${result.metadata.topic}: ${result.content}`)
        .join('\n\n');

      return `
Medical Context from Nelson Pediatrics:
${medicalContext}

User Question: ${userQuery}

Please provide a response based on the medical context above.
`;

    } catch (error) {
      console.error('Error enhancing with medical context:', error);
      return userQuery;
    }
  }, []);

  return {
    results,
    loading,
    error,
    searchMedical,
    enhanceWithMedicalContext
  };
};
```

### **React Native Component**

```tsx
// components/MedicalSearchScreen.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Alert
} from 'react-native';
import { useMedicalSearch } from '../hooks/useMedicalSearch';

export const MedicalSearchScreen: React.FC = () => {
  const [query, setQuery] = useState('');
  const { results, loading, error, searchMedical } = useMedicalSearch();

  const handleSearch = async () => {
    if (query.trim()) {
      try {
        await searchMedical(query);
      } catch (err) {
        Alert.alert('Error', 'Failed to search medical knowledge');
      }
    }
  };

  const renderResult = ({ item, index }: { item: any; index: number }) => (
    <View style={styles.resultCard}>
      <View style={styles.resultHeader}>
        <Text style={styles.topicText}>{item.metadata.topic}</Text>
        <Text style={styles.similarityText}>
          {(item.similarity * 100).toFixed(1)}% match
        </Text>
      </View>
      <Text style={styles.contentText} numberOfLines={3}>
        {item.content}
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          value={query}
          onChangeText={setQuery}
          placeholder="Search Nelson Pediatrics..."
          editable={!loading}
        />
        <TouchableOpacity
          style={[styles.searchButton, loading && styles.disabledButton]}
          onPress={handleSearch}
          disabled={loading || !query.trim()}
        >
          {loading ? (
            <ActivityIndicator color="white" size="small" />
          ) : (
            <Text style={styles.searchButtonText}>Search</Text>
          )}
        </TouchableOpacity>
      </View>

      {error && (
        <Text style={styles.errorText}>Error: {error}</Text>
      )}

      <FlatList
        data={results}
        renderItem={renderResult}
        keyExtractor={(item, index) => index.toString()}
        style={styles.resultsList}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  searchContainer: {
    flexDirection: 'row',
    marginBottom: 16,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: 'white',
  },
  searchButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    justifyContent: 'center',
  },
  disabledButton: {
    opacity: 0.5,
  },
  searchButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  errorText: {
    color: 'red',
    marginBottom: 16,
    textAlign: 'center',
  },
  resultsList: {
    flex: 1,
  },
  resultCard: {
    backgroundColor: 'white',
    padding: 16,
    marginBottom: 8,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  topicText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
    flex: 1,
  },
  similarityText: {
    fontSize: 12,
    color: '#666',
  },
  contentText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
});
```

---

## ☕ **Java Backend Integration**

### **Medical Search Service (Spring Boot)**

```java
// MedicalSearchService.java
package com.nelsongpt.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MedicalSearchService {
    
    @Value("${medical.api.base-url:http://localhost:5000}")
    private String baseUrl;
    
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    public MedicalSearchService(RestTemplate restTemplate, ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }
    
    public MedicalSearchResponse searchMedicalKnowledge(String query, int maxResults) {
        try {
            String url = baseUrl + "/search";
            
            // Prepare request body
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("query", query);
            requestBody.put("top_k", maxResults);
            requestBody.put("include_metadata", true);
            
            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
            
            // Make request
            ResponseEntity<MedicalSearchResponse> response = restTemplate.postForEntity(
                url, request, MedicalSearchResponse.class
            );
            
            return response.getBody();
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to search medical knowledge: " + e.getMessage(), e);
        }
    }
    
    public List<String> getAvailableTopics() {
        try {
            String url = baseUrl + "/search/topics";
            
            ResponseEntity<TopicsResponse> response = restTemplate.getForEntity(
                url, TopicsResponse.class
            );
            
            return response.getBody().getTopics();
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to get topics: " + e.getMessage(), e);
        }
    }
    
    public MedicalSearchResponse searchByTopic(String topic, String query, int maxResults) {
        try {
            String url = baseUrl + "/search/topic/" + topic;
            
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("query", query);
            requestBody.put("top_k", maxResults);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
            
            ResponseEntity<MedicalSearchResponse> response = restTemplate.postForEntity(
                url, request, MedicalSearchResponse.class
            );
            
            return response.getBody();
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to search by topic: " + e.getMessage(), e);
        }
    }
    
    public String enhanceWithMedicalContext(String userQuery) {
        try {
            MedicalSearchResponse searchResults = searchMedicalKnowledge(userQuery, 3);
            
            if (searchResults.getResults().isEmpty()) {
                return userQuery;
            }
            
            StringBuilder medicalContext = new StringBuilder();
            for (MedicalResult result : searchResults.getResults()) {
                medicalContext.append(result.getMetadata().getTopic())
                           .append(": ")
                           .append(result.getContent())
                           .append("\n\n");
            }
            
            return String.format("""
                Medical Context from Nelson Pediatrics:
                %s
                
                User Question: %s
                
                Please provide a response based on the medical context above.
                """, medicalContext.toString(), userQuery);
                
        } catch (Exception e) {
            // Fallback to original query if medical enhancement fails
            return userQuery;
        }
    }
}
```

### **Data Transfer Objects**

```java
// MedicalSearchResponse.java
package com.nelsongpt.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class MedicalSearchResponse {
    private String query;
    
    @JsonProperty("results_count")
    private int resultsCount;
    
    private List<MedicalResult> results;
    private String status;
    
    // Constructors, getters, and setters
    public MedicalSearchResponse() {}
    
    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }
    
    public int getResultsCount() { return resultsCount; }
    public void setResultsCount(int resultsCount) { this.resultsCount = resultsCount; }
    
    public List<MedicalResult> getResults() { return results; }
    public void setResults(List<MedicalResult> results) { this.results = results; }
    
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}

// MedicalResult.java
package com.nelsongpt.dto;

public class MedicalResult {
    private int rank;
    private String content;
    private double similarity;
    private double distance;
    private MedicalMetadata metadata;
    
    // Constructors, getters, and setters
    public MedicalResult() {}
    
    public int getRank() { return rank; }
    public void setRank(int rank) { this.rank = rank; }
    
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    
    public double getSimilarity() { return similarity; }
    public void setSimilarity(double similarity) { this.similarity = similarity; }
    
    public double getDistance() { return distance; }
    public void setDistance(double distance) { this.distance = distance; }
    
    public MedicalMetadata getMetadata() { return metadata; }
    public void setMetadata(MedicalMetadata metadata) { this.metadata = metadata; }
}

// MedicalMetadata.java
package com.nelsongpt.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class MedicalMetadata {
    private String topic;
    
    @JsonProperty("source_file")
    private String sourceFile;
    
    @JsonProperty("chunk_number")
    private int chunkNumber;
    
    @JsonProperty("character_count")
    private int characterCount;
    
    @JsonProperty("created_at")
    private String createdAt;
    
    // Constructors, getters, and setters
    public MedicalMetadata() {}
    
    public String getTopic() { return topic; }
    public void setTopic(String topic) { this.topic = topic; }
    
    public String getSourceFile() { return sourceFile; }
    public void setSourceFile(String sourceFile) { this.sourceFile = sourceFile; }
    
    public int getChunkNumber() { return chunkNumber; }
    public void setChunkNumber(int chunkNumber) { this.chunkNumber = chunkNumber; }
    
    public int getCharacterCount() { return characterCount; }
    public void setCharacterCount(int characterCount) { this.characterCount = characterCount; }
    
    public String getCreatedAt() { return createdAt; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }
}

// TopicsResponse.java
package com.nelsongpt.dto;

import java.util.List;

public class TopicsResponse {
    private List<String> topics;
    private int count;
    private String status;
    
    // Constructors, getters, and setters
    public TopicsResponse() {}
    
    public List<String> getTopics() { return topics; }
    public void setTopics(List<String> topics) { this.topics = topics; }
    
    public int getCount() { return count; }
    public void setCount(int count) { this.count = count; }
    
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
```

### **REST Controller**

```java
// MedicalSearchController.java
package com.nelsongpt.controller;

import com.nelsongpt.dto.MedicalSearchResponse;
import com.nelsongpt.service.MedicalSearchService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/medical")
@CrossOrigin(origins = "*") // Configure appropriately for production
public class MedicalSearchController {
    
    private final MedicalSearchService medicalSearchService;
    
    public MedicalSearchController(MedicalSearchService medicalSearchService) {
        this.medicalSearchService = medicalSearchService;
    }
    
    @PostMapping("/search")
    public ResponseEntity<MedicalSearchResponse> searchMedical(
            @RequestBody Map<String, Object> request) {
        
        String query = (String) request.get("query");
        Integer maxResults = (Integer) request.getOrDefault("maxResults", 5);
        
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        
        try {
            MedicalSearchResponse response = medicalSearchService.searchMedicalKnowledge(
                query, maxResults
            );
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
    
    @GetMapping("/topics")
    public ResponseEntity<List<String>> getTopics() {
        try {
            List<String> topics = medicalSearchService.getAvailableTopics();
            return ResponseEntity.ok(topics);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
    
    @PostMapping("/search/topic/{topic}")
    public ResponseEntity<MedicalSearchResponse> searchByTopic(
            @PathVariable String topic,
            @RequestBody Map<String, Object> request) {
        
        String query = (String) request.get("query");
        Integer maxResults = (Integer) request.getOrDefault("maxResults", 5);
        
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        
        try {
            MedicalSearchResponse response = medicalSearchService.searchByTopic(
                topic, query, maxResults
            );
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
    
    @PostMapping("/enhance")
    public ResponseEntity<Map<String, String>> enhanceWithMedicalContext(
            @RequestBody Map<String, String> request) {
        
        String userQuery = request.get("query");
        
        if (userQuery == null || userQuery.trim().isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        
        try {
            String enhancedPrompt = medicalSearchService.enhanceWithMedicalContext(userQuery);
            return ResponseEntity.ok(Map.of(
                "originalQuery", userQuery,
                "enhancedPrompt", enhancedPrompt
            ));
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}
```

### **Configuration**

```java
// AppConfig.java
package com.nelsongpt.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AppConfig {
    
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

### **Application Properties**

```properties
# application.properties
medical.api.base-url=http://localhost:5000
server.port=8080

# CORS configuration
spring.web.cors.allowed-origins=*
spring.web.cors.allowed-methods=GET,POST,PUT,DELETE,OPTIONS
spring.web.cors.allowed-headers=*
```

---

## 🚀 **Quick Start Commands**

### **Next.js**
```bash
# Install dependencies
npm install

# Start your Nelson Medical API
python nelson_web_api.py

# Start Next.js app
npm run dev
```

### **React Native**
```bash
# Install dependencies
npm install

# Start your Nelson Medical API
python nelson_web_api.py

# Start React Native (iOS)
npx react-native run-ios

# Start React Native (Android)
npx react-native run-android
```

### **Java Spring Boot**
```bash
# Start your Nelson Medical API
python nelson_web_api.py

# Start Spring Boot app
./mvnw spring-boot:run
```

---

## 🎯 **Integration Summary**

| Platform | Integration Method | Benefits |
|----------|-------------------|----------|
| **Next.js** | API Routes + Hooks | Server-side rendering, SEO-friendly |
| **React Native** | Service + Hooks | Cross-platform mobile, native performance |
| **Java** | RestTemplate + DTOs | Enterprise-ready, type-safe, scalable |

**All platforms can now enhance their responses with Nelson Pediatrics medical knowledge! 🏥⚡**


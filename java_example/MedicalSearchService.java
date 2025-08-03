// Java Spring Boot Medical Search Service
package com.nelsongpt.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nelsongpt.dto.MedicalSearchResponse;
import com.nelsongpt.dto.TopicsResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MedicalSearchService {
    
    private static final Logger logger = LoggerFactory.getLogger(MedicalSearchService.class);
    
    @Value("${medical.api.base-url:http://localhost:5000}")
    private String baseUrl;
    
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    public MedicalSearchService(RestTemplate restTemplate, ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }
    
    /**
     * Search medical knowledge base
     */
    public MedicalSearchResponse searchMedicalKnowledge(String query, int maxResults) {
        logger.info("Searching medical knowledge for query: '{}' with maxResults: {}", query, maxResults);
        
        try {
            String url = baseUrl + "/search";
            
            // Prepare request body
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("query", query.trim());
            requestBody.put("top_k", Math.min(maxResults, 20)); // Limit to 20 results
            requestBody.put("include_metadata", true);
            
            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
            
            // Make request
            ResponseEntity<MedicalSearchResponse> response = restTemplate.postForEntity(
                url, request, MedicalSearchResponse.class
            );
            
            MedicalSearchResponse result = response.getBody();
            logger.info("Medical search completed. Found {} results", 
                result != null ? result.getResultsCount() : 0);
            
            return result;
            
        } catch (RestClientException e) {
            logger.error("Failed to search medical knowledge", e);
            throw new RuntimeException("Failed to search medical knowledge: " + e.getMessage(), e);
        }
    }
    
    /**
     * Get all available medical topics
     */
    public List<String> getAvailableTopics() {
        logger.info("Fetching available medical topics");
        
        try {
            String url = baseUrl + "/search/topics";
            
            ResponseEntity<TopicsResponse> response = restTemplate.getForEntity(
                url, TopicsResponse.class
            );
            
            TopicsResponse result = response.getBody();
            List<String> topics = result != null ? result.getTopics() : List.of();
            
            logger.info("Found {} medical topics", topics.size());
            return topics;
            
        } catch (RestClientException e) {
            logger.error("Failed to get topics", e);
            throw new RuntimeException("Failed to get topics: " + e.getMessage(), e);
        }
    }
    
    /**
     * Search within a specific medical topic
     */
    public MedicalSearchResponse searchByTopic(String topic, String query, int maxResults) {
        logger.info("Searching in topic '{}' for query: '{}' with maxResults: {}", 
            topic, query, maxResults);
        
        try {
            String url = baseUrl + "/search/topic/" + topic;
            
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("query", query.trim());
            requestBody.put("top_k", Math.min(maxResults, 20));
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
            
            ResponseEntity<MedicalSearchResponse> response = restTemplate.postForEntity(
                url, request, MedicalSearchResponse.class
            );
            
            MedicalSearchResponse result = response.getBody();
            logger.info("Topic search completed. Found {} results", 
                result != null ? result.getResultsCount() : 0);
            
            return result;
            
        } catch (RestClientException e) {
            logger.error("Failed to search by topic", e);
            throw new RuntimeException("Failed to search by topic: " + e.getMessage(), e);
        }
    }
    
    /**
     * Get database statistics
     */
    public Map<String, Object> getDatabaseStats() {
        logger.info("Fetching database statistics");
        
        try {
            String url = baseUrl + "/stats";
            
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            Map<String, Object> stats = response.getBody();
            
            logger.info("Database stats retrieved successfully");
            return stats != null ? stats : Map.of();
            
        } catch (RestClientException e) {
            logger.error("Failed to get database stats", e);
            throw new RuntimeException("Failed to get database stats: " + e.getMessage(), e);
        }
    }
    
    /**
     * Check API health
     */
    public boolean checkHealth() {
        try {
            String url = baseUrl + "/health";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            
            boolean isHealthy = response.getStatusCode().is2xxSuccessful();
            logger.info("Health check result: {}", isHealthy ? "HEALTHY" : "UNHEALTHY");
            
            return isHealthy;
            
        } catch (Exception e) {
            logger.warn("Health check failed", e);
            return false;
        }
    }
    
    /**
     * Enhance user query with medical context for GPT
     */
    public String enhanceWithMedicalContext(String userQuery) {
        logger.info("Enhancing query with medical context: '{}'", userQuery);
        
        try {
            MedicalSearchResponse searchResults = searchMedicalKnowledge(userQuery, 3);
            
            if (searchResults == null || searchResults.getResults().isEmpty()) {
                logger.info("No medical context found, returning original query");
                return userQuery;
            }
            
            // Build medical context
            StringBuilder medicalContext = new StringBuilder();
            searchResults.getResults().forEach(result -> {
                medicalContext.append(result.getMetadata().getTopic())
                           .append(": ")
                           .append(result.getContent())
                           .append("\n\n");
            });
            
            // Create enhanced prompt
            String enhancedPrompt = String.format("""
                Medical Context from Nelson Pediatrics:
                %s
                
                User Question: %s
                
                Please provide a response based on the medical context above. Always recommend consulting with healthcare professionals for medical decisions.
                """, medicalContext.toString().trim(), userQuery);
            
            logger.info("Query enhanced with {} medical results", searchResults.getResultsCount());
            return enhancedPrompt;
                
        } catch (Exception e) {
            logger.warn("Failed to enhance query with medical context, returning original", e);
            return userQuery; // Fallback to original query
        }
    }
    
    /**
     * Validate medical API connection
     */
    public void validateConnection() {
        if (!checkHealth()) {
            throw new RuntimeException(
                "Cannot connect to medical API at " + baseUrl + 
                ". Please ensure the Nelson Medical API is running."
            );
        }
        logger.info("Medical API connection validated successfully");
    }
}


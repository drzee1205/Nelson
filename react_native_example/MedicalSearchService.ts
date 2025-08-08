// React Native Medical Search Service
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

interface TopicsResponse {
  topics: string[];
  count: number;
  status: string;
}

class MedicalSearchService {
  private baseUrl: string;

  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }

  /**
   * Search medical knowledge base
   */
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
          query: query.trim(),
          top_k: maxResults,
          include_metadata: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: MedicalSearchResponse = await response.json();
      return data.results || [];

    } catch (error) {
      console.error('Medical search error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to search medical knowledge'
      );
    }
  }

  /**
   * Get all available medical topics
   */
  async getAvailableTopics(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/search/topics`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: TopicsResponse = await response.json();
      return data.topics || [];

    } catch (error) {
      console.error('Topics error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to get topics'
      );
    }
  }

  /**
   * Search within a specific medical topic
   */
  async searchByTopic(
    topic: string, 
    query: string, 
    maxResults = 5
  ): Promise<MedicalResult[]> {
    try {
      const encodedTopic = encodeURIComponent(topic);
      const response = await fetch(`${this.baseUrl}/search/topic/${encodedTopic}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: maxResults
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: MedicalSearchResponse = await response.json();
      return data.results || [];

    } catch (error) {
      console.error('Topic search error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to search by topic'
      );
    }
  }

  /**
   * Get database statistics
   */
  async getDatabaseStats(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/stats`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();

    } catch (error) {
      console.error('Stats error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to get database stats'
      );
    }
  }

  /**
   * Check API health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check error:', error);
      return false;
    }
  }

  /**
   * Enhance user query with medical context for GPT
   */
  async enhanceWithMedicalContext(userQuery: string): Promise<string> {
    try {
      const medicalResults = await this.searchMedicalKnowledge(userQuery, 3);
      
      if (!medicalResults.length) {
        return userQuery; // Return original if no medical context found
      }

      // Build medical context
      const medicalContext = medicalResults
        .map(result => `${result.metadata.topic}: ${result.content}`)
        .join('\n\n');

      // Create enhanced prompt
      return `
Medical Context from Nelson Pediatrics:
${medicalContext}

User Question: ${userQuery}

Please provide a response based on the medical context above. Always recommend consulting with healthcare professionals for medical decisions.
`;

    } catch (error) {
      console.error('Error enhancing with medical context:', error);
      return userQuery; // Fallback to original query
    }
  }
}

// Export singleton instance
export const medicalSearchService = new MedicalSearchService();

// Export types for use in components
export type { MedicalResult, MedicalSearchResponse, TopicsResponse };


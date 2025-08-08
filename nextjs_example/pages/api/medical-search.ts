// Next.js API route for medical search
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

interface MedicalSearchResponse {
  success: boolean;
  query?: string;
  results?: MedicalResult[];
  resultsCount?: number;
  error?: string;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<MedicalSearchResponse>
) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ 
      success: false, 
      error: 'Method not allowed' 
    });
  }

  const { query, maxResults = 5 }: MedicalSearchRequest = req.body;

  // Validate input
  if (!query || query.trim().length === 0) {
    return res.status(400).json({
      success: false,
      error: 'Query is required'
    });
  }

  try {
    // Call your Nelson Medical API
    const medicalApiUrl = process.env.MEDICAL_API_URL || 'http://localhost:5000';
    
    const response = await fetch(`${medicalApiUrl}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query.trim(),
        top_k: Math.min(maxResults, 20), // Limit to 20 results max
        include_metadata: true
      })
    });

    if (!response.ok) {
      throw new Error(`Medical API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    
    // Return formatted response
    res.status(200).json({
      success: true,
      query: data.query,
      results: data.results,
      resultsCount: data.results_count
    });

  } catch (error) {
    console.error('Medical search error:', error);
    
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to search medical knowledge'
    });
  }
}


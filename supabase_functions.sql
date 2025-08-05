-- =====================================================
-- SUPABASE FUNCTIONS FOR NELSON PEDIATRICS DATABASE
-- =====================================================
-- Run these functions in your Supabase SQL Editor
-- They will enhance search capabilities and page management

-- =====================================================
-- 1. VECTOR SIMILARITY SEARCH FUNCTION
-- =====================================================
-- This function enables semantic search using embeddings
CREATE OR REPLACE FUNCTION search_embeddings(
  query_embedding vector(384),
  match_threshold float DEFAULT 0.1,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id uuid,
  content text,
  chapter_title text,
  section_title text,
  page_number integer,
  chunk_index integer,
  similarity float,
  metadata jsonb,
  created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    nelson_textbook_chunks.id,
    nelson_textbook_chunks.content,
    nelson_textbook_chunks.chapter_title,
    nelson_textbook_chunks.section_title,
    nelson_textbook_chunks.page_number,
    nelson_textbook_chunks.chunk_index,
    1 - (nelson_textbook_chunks.embedding <=> query_embedding) AS similarity,
    nelson_textbook_chunks.metadata,
    nelson_textbook_chunks.created_at
  FROM nelson_textbook_chunks
  WHERE nelson_textbook_chunks.embedding IS NOT NULL
    AND 1 - (nelson_textbook_chunks.embedding <=> query_embedding) > match_threshold
  ORDER BY nelson_textbook_chunks.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- =====================================================
-- 2. PAGE RANGE SEARCH FUNCTION
-- =====================================================
-- Enhanced search within specific page ranges
CREATE OR REPLACE FUNCTION search_page_range(
  search_query text,
  min_page_num integer DEFAULT 1,
  max_page_num integer DEFAULT 99999,
  result_limit integer DEFAULT 10
)
RETURNS TABLE (
  id uuid,
  content text,
  chapter_title text,
  section_title text,
  page_number integer,
  chunk_index integer,
  relevance_score float,
  created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    n.id,
    n.content,
    n.chapter_title,
    n.section_title,
    n.page_number,
    n.chunk_index,
    ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', search_query)) AS relevance_score,
    n.created_at
  FROM nelson_textbook_chunks n
  WHERE n.page_number IS NOT NULL
    AND n.page_number >= min_page_num
    AND n.page_number <= max_page_num
    AND to_tsvector('english', n.content) @@ plainto_tsquery('english', search_query)
  ORDER BY relevance_score DESC, n.page_number ASC, n.chunk_index ASC
  LIMIT result_limit;
END;
$$;

-- =====================================================
-- 3. CHAPTER PAGE STATISTICS FUNCTION
-- =====================================================
-- Get comprehensive statistics for chapter page ranges
CREATE OR REPLACE FUNCTION get_chapter_page_stats(chapter_pattern text DEFAULT '%')
RETURNS TABLE (
  chapter_title text,
  min_page integer,
  max_page integer,
  total_pages integer,
  document_count bigint,
  avg_content_length float,
  last_updated timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    n.chapter_title,
    MIN(n.page_number) AS min_page,
    MAX(n.page_number) AS max_page,
    (MAX(n.page_number) - MIN(n.page_number) + 1) AS total_pages,
    COUNT(*) AS document_count,
    AVG(LENGTH(n.content)) AS avg_content_length,
    MAX(n.created_at) AS last_updated
  FROM nelson_textbook_chunks n
  WHERE n.page_number IS NOT NULL
    AND n.chapter_title ILIKE chapter_pattern
  GROUP BY n.chapter_title
  ORDER BY MIN(n.page_number);
END;
$$;

-- =====================================================
-- 4. MEDICAL SPECIALTY PAGE FINDER
-- =====================================================
-- Find pages related to specific medical specialties
CREATE OR REPLACE FUNCTION find_specialty_pages(
  specialty_keywords text[],
  page_limit integer DEFAULT 20
)
RETURNS TABLE (
  specialty_match text,
  page_number integer,
  chapter_title text,
  content_preview text,
  match_count integer
)
LANGUAGE plpgsql
AS $$
DECLARE
  keyword text;
  query_text text := '';
BEGIN
  -- Build search query from keywords
  FOREACH keyword IN ARRAY specialty_keywords
  LOOP
    IF query_text != '' THEN
      query_text := query_text || ' | ';
    END IF;
    query_text := query_text || keyword;
  END LOOP;

  RETURN QUERY
  SELECT
    array_to_string(specialty_keywords, ', ') AS specialty_match,
    n.page_number,
    n.chapter_title,
    LEFT(n.content, 200) || '...' AS content_preview,
    (
      SELECT COUNT(*)::integer
      FROM unnest(specialty_keywords) AS kw
      WHERE n.content ILIKE '%' || kw || '%'
    ) AS match_count
  FROM nelson_textbook_chunks n
  WHERE n.page_number IS NOT NULL
    AND to_tsvector('english', n.content) @@ to_tsquery('english', query_text)
  ORDER BY match_count DESC, n.page_number ASC
  LIMIT page_limit;
END;
$$;

-- =====================================================
-- 5. PAGE CONTENT AGGREGATOR
-- =====================================================
-- Get all content from a specific page, properly ordered
CREATE OR REPLACE FUNCTION get_page_content(target_page integer)
RETURNS TABLE (
  page_number integer,
  total_chunks integer,
  chapter_title text,
  full_content text,
  chunk_details jsonb
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    target_page AS page_number,
    COUNT(*)::integer AS total_chunks,
    MAX(n.chapter_title) AS chapter_title,
    STRING_AGG(n.content, ' ' ORDER BY n.chunk_index) AS full_content,
    jsonb_agg(
      jsonb_build_object(
        'chunk_index', n.chunk_index,
        'content_length', LENGTH(n.content),
        'section_title', n.section_title,
        'id', n.id
      ) ORDER BY n.chunk_index
    ) AS chunk_details
  FROM nelson_textbook_chunks n
  WHERE n.page_number = target_page
  GROUP BY target_page;
END;
$$;

-- =====================================================
-- 6. BULK PAGE NUMBER UPDATE FUNCTION
-- =====================================================
-- Efficiently update page numbers based on content analysis
CREATE OR REPLACE FUNCTION update_page_numbers_bulk()
RETURNS TABLE (
  updated_count integer,
  processing_time interval
)
LANGUAGE plpgsql
AS $$
DECLARE
  start_time timestamptz;
  end_time timestamptz;
  update_count integer := 0;
BEGIN
  start_time := clock_timestamp();
  
  -- Update records without page numbers using content-based estimation
  WITH page_estimates AS (
    SELECT 
      id,
      CASE 
        -- Allergic disorders
        WHEN content ILIKE '%allergic%' OR content ILIKE '%allergy%' THEN 1100 + (chunk_index / 20)
        -- Behavioral/Psychiatric
        WHEN content ILIKE '%behavioral%' OR content ILIKE '%psychiatric%' THEN 200 + (chunk_index / 20)
        -- Cardiovascular
        WHEN content ILIKE '%cardiovascular%' OR content ILIKE '%cardiac%' OR content ILIKE '%heart%' THEN 2200 + (chunk_index / 20)
        -- Respiratory
        WHEN content ILIKE '%respiratory%' OR content ILIKE '%pulmonary%' OR content ILIKE '%lung%' THEN 2000 + (chunk_index / 20)
        -- Neurologic
        WHEN content ILIKE '%neurologic%' OR content ILIKE '%neurological%' OR content ILIKE '%brain%' THEN 2900 + (chunk_index / 20)
        -- Endocrine
        WHEN content ILIKE '%endocrine%' OR content ILIKE '%hormone%' THEN 2700 + (chunk_index / 20)
        -- Infectious diseases
        WHEN content ILIKE '%infectious%' OR content ILIKE '%infection%' THEN 1200 + (chunk_index / 20)
        -- Hematologic
        WHEN content ILIKE '%hematologic%' OR content ILIKE '%blood%' THEN 2400 + (chunk_index / 20)
        -- Oncologic
        WHEN content ILIKE '%oncologic%' OR content ILIKE '%cancer%' OR content ILIKE '%tumor%' THEN 2500 + (chunk_index / 20)
        -- Dermatologic
        WHEN content ILIKE '%dermatologic%' OR content ILIKE '%skin%' THEN 3100 + (chunk_index / 20)
        -- Urologic/Renal
        WHEN content ILIKE '%urologic%' OR content ILIKE '%renal%' OR content ILIKE '%kidney%' THEN 2600 + (chunk_index / 20)
        -- Default estimation
        ELSE GREATEST(1, chunk_index / 20 + 1)
      END AS estimated_page
    FROM nelson_textbook_chunks
    WHERE page_number IS NULL
    LIMIT 1000  -- Process in batches
  )
  UPDATE nelson_textbook_chunks 
  SET page_number = LEAST(page_estimates.estimated_page, 3500)  -- Cap at reasonable max
  FROM page_estimates
  WHERE nelson_textbook_chunks.id = page_estimates.id;
  
  GET DIAGNOSTICS update_count = ROW_COUNT;
  end_time := clock_timestamp();
  
  RETURN QUERY
  SELECT 
    update_count,
    end_time - start_time;
END;
$$;

-- =====================================================
-- 7. SEARCH ANALYTICS FUNCTION
-- =====================================================
-- Track and analyze search patterns
CREATE OR REPLACE FUNCTION log_search_analytics(
  search_term text,
  search_type text DEFAULT 'general',
  page_range_min integer DEFAULT NULL,
  page_range_max integer DEFAULT NULL,
  results_count integer DEFAULT 0
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  -- Create analytics table if it doesn't exist
  CREATE TABLE IF NOT EXISTS search_analytics (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    search_term text NOT NULL,
    search_type text NOT NULL,
    page_range_min integer,
    page_range_max integer,
    results_count integer DEFAULT 0,
    search_timestamp timestamptz DEFAULT now()
  );
  
  -- Insert search log
  INSERT INTO search_analytics (
    search_term, 
    search_type, 
    page_range_min, 
    page_range_max, 
    results_count
  ) VALUES (
    search_term, 
    search_type, 
    page_range_min, 
    page_range_max, 
    results_count
  );
END;
$$;

-- =====================================================
-- 8. DATABASE HEALTH CHECK FUNCTION
-- =====================================================
-- Comprehensive health check for the medical database
CREATE OR REPLACE FUNCTION database_health_check()
RETURNS TABLE (
  metric_name text,
  metric_value text,
  status text
)
LANGUAGE plpgsql
AS $$
DECLARE
  total_docs bigint;
  docs_with_pages bigint;
  coverage_pct numeric;
  min_page integer;
  max_page integer;
  avg_content_length numeric;
BEGIN
  -- Get basic statistics
  SELECT COUNT(*) INTO total_docs FROM nelson_textbook_chunks;
  SELECT COUNT(*) INTO docs_with_pages FROM nelson_textbook_chunks WHERE page_number IS NOT NULL;
  
  IF total_docs > 0 THEN
    coverage_pct := ROUND((docs_with_pages::numeric / total_docs::numeric) * 100, 2);
  ELSE
    coverage_pct := 0;
  END IF;
  
  SELECT MIN(page_number), MAX(page_number) 
  INTO min_page, max_page 
  FROM nelson_textbook_chunks 
  WHERE page_number IS NOT NULL;
  
  SELECT ROUND(AVG(LENGTH(content)), 2) 
  INTO avg_content_length 
  FROM nelson_textbook_chunks;
  
  -- Return health metrics
  RETURN QUERY VALUES
    ('Total Documents', total_docs::text, CASE WHEN total_docs > 0 THEN 'OK' ELSE 'WARNING' END),
    ('Documents with Pages', docs_with_pages::text, CASE WHEN docs_with_pages > 0 THEN 'OK' ELSE 'WARNING' END),
    ('Page Coverage %', coverage_pct::text || '%', CASE WHEN coverage_pct > 10 THEN 'OK' WHEN coverage_pct > 0 THEN 'WARNING' ELSE 'ERROR' END),
    ('Page Range', COALESCE(min_page::text || ' - ' || max_page::text, 'No pages'), CASE WHEN min_page IS NOT NULL THEN 'OK' ELSE 'WARNING' END),
    ('Avg Content Length', avg_content_length::text, CASE WHEN avg_content_length > 100 THEN 'OK' ELSE 'WARNING' END);
END;
$$;

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================
-- Create indexes to optimize the new functions

-- Index for page number queries
CREATE INDEX IF NOT EXISTS idx_nelson_page_number_not_null 
ON nelson_textbook_chunks (page_number) 
WHERE page_number IS NOT NULL;

-- Index for chapter and page combination
CREATE INDEX IF NOT EXISTS idx_nelson_chapter_page 
ON nelson_textbook_chunks (chapter_title, page_number) 
WHERE page_number IS NOT NULL;

-- Index for content search with page filtering
CREATE INDEX IF NOT EXISTS idx_nelson_content_gin 
ON nelson_textbook_chunks 
USING gin(to_tsvector('english', content));

-- Index for chunk ordering within pages
CREATE INDEX IF NOT EXISTS idx_nelson_page_chunk_order 
ON nelson_textbook_chunks (page_number, chunk_index) 
WHERE page_number IS NOT NULL;

-- =====================================================
-- USAGE EXAMPLES
-- =====================================================
/*
-- Example 1: Vector similarity search
SELECT * FROM search_embeddings(
  '[0.1, 0.2, 0.3, ...]'::vector(384),  -- Your query embedding
  0.1,  -- Similarity threshold
  5     -- Number of results
);

-- Example 2: Search within page range
SELECT * FROM search_page_range(
  'asthma treatment',  -- Search query
  1100,               -- Min page
  1200,               -- Max page
  10                  -- Result limit
);

-- Example 3: Get chapter statistics
SELECT * FROM get_chapter_page_stats('allergic%');

-- Example 4: Find specialty pages
SELECT * FROM find_specialty_pages(
  ARRAY['asthma', 'allergy', 'respiratory'],
  15
);

-- Example 5: Get complete page content
SELECT * FROM get_page_content(1101);

-- Example 6: Bulk update page numbers
SELECT * FROM update_page_numbers_bulk();

-- Example 7: Database health check
SELECT * FROM database_health_check();

-- Example 8: Log search analytics
SELECT log_search_analytics(
  'asthma treatment',
  'page_range',
  1100,
  1200,
  5
);
*/

-- =====================================================
-- GRANT PERMISSIONS
-- =====================================================
-- Grant execute permissions to authenticated users
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon;


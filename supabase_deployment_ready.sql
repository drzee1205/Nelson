
-- =====================================================
-- NELSON PEDIATRICS AI FUNCTIONS - PRODUCTION DEPLOYMENT
-- =====================================================
-- Copy and paste this entire script into Supabase SQL Editor

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- 1. VECTOR SIMILARITY SEARCH FUNCTION (1536D)
-- =====================================================
CREATE OR REPLACE FUNCTION search_embeddings(
  query_embedding vector(1536),
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
-- 2. ENHANCED SEMANTIC SEARCH WITH FILTERS
-- =====================================================
CREATE OR REPLACE FUNCTION search_embeddings_filtered(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.1,
  match_count int DEFAULT 5,
  min_page_filter int DEFAULT NULL,
  max_page_filter int DEFAULT NULL,
  section_filter text DEFAULT NULL,
  chapter_filter text DEFAULT NULL
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
    n.id,
    n.content,
    n.chapter_title,
    n.section_title,
    n.page_number,
    n.chunk_index,
    1 - (n.embedding <=> query_embedding) AS similarity,
    n.metadata,
    n.created_at
  FROM nelson_textbook_chunks n
  WHERE n.embedding IS NOT NULL
    AND 1 - (n.embedding <=> query_embedding) > match_threshold
    AND (min_page_filter IS NULL OR n.page_number >= min_page_filter)
    AND (max_page_filter IS NULL OR n.page_number <= max_page_filter)
    AND (section_filter IS NULL OR n.section_title ILIKE '%' || section_filter || '%')
    AND (chapter_filter IS NULL OR n.chapter_title ILIKE '%' || chapter_filter || '%')
  ORDER BY n.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- =====================================================
-- 3. MEDICAL SECTION SEARCH
-- =====================================================
CREATE OR REPLACE FUNCTION search_medical_sections(
  section_pattern text,
  search_query text DEFAULT NULL,
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
  IF search_query IS NOT NULL AND search_query != '' THEN
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
    WHERE n.section_title ILIKE '%' || section_pattern || '%'
      AND to_tsvector('english', n.content) @@ plainto_tsquery('english', search_query)
    ORDER BY relevance_score DESC, n.page_number ASC, n.chunk_index ASC
    LIMIT result_limit;
  ELSE
    RETURN QUERY
    SELECT
      n.id,
      n.content,
      n.chapter_title,
      n.section_title,
      n.page_number,
      n.chunk_index,
      0.0 AS relevance_score,
      n.created_at
    FROM nelson_textbook_chunks n
    WHERE n.section_title ILIKE '%' || section_pattern || '%'
    ORDER BY n.page_number ASC, n.chunk_index ASC
    LIMIT result_limit;
  END IF;
END;
$$;

-- =====================================================
-- 4. AI SYSTEM HEALTH CHECK
-- =====================================================
CREATE OR REPLACE FUNCTION ai_system_health_check()
RETURNS TABLE (
  metric_name text,
  metric_value text,
  status text,
  details jsonb
)
LANGUAGE plpgsql
AS $$
DECLARE
  total_docs bigint;
  docs_with_embeddings bigint;
  docs_with_sections bigint;
  docs_with_pages bigint;
  coverage_pct numeric;
  section_pct numeric;
  page_pct numeric;
BEGIN
  -- Get counts
  SELECT COUNT(*) INTO total_docs FROM nelson_textbook_chunks;
  SELECT COUNT(*) INTO docs_with_embeddings FROM nelson_textbook_chunks WHERE embedding IS NOT NULL;
  SELECT COUNT(*) INTO docs_with_sections FROM nelson_textbook_chunks WHERE section_title IS NOT NULL;
  SELECT COUNT(*) INTO docs_with_pages FROM nelson_textbook_chunks WHERE page_number IS NOT NULL;
  
  -- Calculate percentages
  coverage_pct := ROUND((docs_with_embeddings::numeric / NULLIF(total_docs, 0)) * 100, 2);
  section_pct := ROUND((docs_with_sections::numeric / NULLIF(total_docs, 0)) * 100, 2);
  page_pct := ROUND((docs_with_pages::numeric / NULLIF(total_docs, 0)) * 100, 2);
  
  -- Return health metrics
  RETURN QUERY VALUES
    ('Total Documents', total_docs::text, 
     CASE WHEN total_docs > 0 THEN 'OK' ELSE 'ERROR' END,
     jsonb_build_object('count', total_docs)),
    
    ('Embedding Coverage', coverage_pct::text || '%', 
     CASE WHEN coverage_pct > 50 THEN 'OK' WHEN coverage_pct > 10 THEN 'WARNING' ELSE 'NEEDS_PROCESSING' END,
     jsonb_build_object('count', docs_with_embeddings, 'percentage', coverage_pct)),
    
    ('Section Coverage', section_pct::text || '%',
     CASE WHEN section_pct > 30 THEN 'OK' WHEN section_pct > 5 THEN 'WARNING' ELSE 'NEEDS_PROCESSING' END,
     jsonb_build_object('count', docs_with_sections, 'percentage', section_pct)),
    
    ('Page Coverage', page_pct::text || '%',
     CASE WHEN page_pct > 50 THEN 'OK' WHEN page_pct > 10 THEN 'WARNING' ELSE 'NEEDS_PROCESSING' END,
     jsonb_build_object('count', docs_with_pages, 'percentage', page_pct)),
    
    ('AI Search Ready', 
     CASE WHEN docs_with_embeddings > 100 THEN 'YES' ELSE 'PROCESSING' END,
     CASE WHEN docs_with_embeddings > 100 THEN 'OK' ELSE 'WARNING' END,
     jsonb_build_object('min_embeddings_needed', 100, 'current_embeddings', docs_with_embeddings));
END;
$$;

-- =====================================================
-- 5. MEDICAL SPECIALTY FINDER
-- =====================================================
CREATE OR REPLACE FUNCTION find_medical_specialties(
  specialty_keywords text[],
  page_limit integer DEFAULT 20,
  include_embeddings boolean DEFAULT false
)
RETURNS TABLE (
  specialty_match text,
  page_number integer,
  chapter_title text,
  section_title text,
  content_preview text,
  match_count integer,
  similarity_score float
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
    n.section_title,
    LEFT(n.content, 200) || '...' AS content_preview,
    (
      SELECT COUNT(*)::integer
      FROM unnest(specialty_keywords) AS kw
      WHERE n.content ILIKE '%' || kw || '%'
    ) AS match_count,
    CASE 
      WHEN include_embeddings AND n.embedding IS NOT NULL THEN 
        -- Placeholder similarity score (would need actual query embedding)
        0.5::float
      ELSE 
        ts_rank(to_tsvector('english', n.content), to_tsquery('english', query_text))
    END AS similarity_score
  FROM nelson_textbook_chunks n
  WHERE n.page_number IS NOT NULL
    AND to_tsvector('english', n.content) @@ to_tsquery('english', query_text)
  ORDER BY match_count DESC, similarity_score DESC, n.page_number ASC
  LIMIT page_limit;
END;
$$;

-- =====================================================
-- 6. PERFORMANCE INDEXES
-- =====================================================

-- Vector similarity index for 1536D embeddings
CREATE INDEX IF NOT EXISTS idx_nelson_embedding_1536_hnsw 
ON nelson_textbook_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding IS NOT NULL;

-- Alternative IVFFlat index
CREATE INDEX IF NOT EXISTS idx_nelson_embedding_1536_ivfflat 
ON nelson_textbook_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100)
WHERE embedding IS NOT NULL;

-- Section-based search optimization
CREATE INDEX IF NOT EXISTS idx_nelson_section_search 
ON nelson_textbook_chunks (section_title, page_number) 
WHERE section_title IS NOT NULL;

-- Medical content full-text search
CREATE INDEX IF NOT EXISTS idx_nelson_medical_content_gin 
ON nelson_textbook_chunks 
USING gin(to_tsvector('english', content));

-- Combined medical search index
CREATE INDEX IF NOT EXISTS idx_nelson_medical_combined 
ON nelson_textbook_chunks (chapter_title, section_title, page_number) 
WHERE section_title IS NOT NULL AND page_number IS NOT NULL;

-- =====================================================
-- 7. GRANT PERMISSIONS
-- =====================================================
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon;

-- =====================================================
-- 8. VERIFICATION QUERIES
-- =====================================================

-- Test the health check function
SELECT * FROM ai_system_health_check();

-- Test medical specialty search
SELECT * FROM find_medical_specialties(ARRAY['asthma', 'allergy'], 5);

-- Check if indexes were created
SELECT indexname, tablename 
FROM pg_indexes 
WHERE tablename = 'nelson_textbook_chunks' 
AND indexname LIKE '%nelson%'
ORDER BY indexname;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================
DO $$ 
BEGIN
    RAISE NOTICE '🎉 NELSON PEDIATRICS AI FUNCTIONS DEPLOYED SUCCESSFULLY!';
    RAISE NOTICE '🤖 Vector similarity search ready (1536D embeddings)';
    RAISE NOTICE '📝 Medical section search enabled';
    RAISE NOTICE '🏥 Medical specialty finder active';
    RAISE NOTICE '📊 AI system health monitoring deployed';
    RAISE NOTICE '⚡ Performance indexes created';
    RAISE NOTICE '🚀 Your NelsonGPT is now AI-powered!';
END $$;

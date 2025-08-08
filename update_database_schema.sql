-- =====================================================
-- DATABASE SCHEMA UPDATES FOR EMBEDDINGS & SECTIONS
-- =====================================================
-- Run these commands in your Supabase SQL Editor to add
-- embedding and section title support

-- =====================================================
-- 1. ENABLE VECTOR EXTENSION (if not already enabled)
-- =====================================================
-- Enable the pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- 2. ADD EMBEDDING COLUMN (if not exists)
-- =====================================================
-- Add vector column for embeddings (384 dimensions for all-MiniLM-L6-v2)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'nelson_textbook_chunks' 
        AND column_name = 'embedding'
    ) THEN
        ALTER TABLE nelson_textbook_chunks 
        ADD COLUMN embedding vector(384);
        
        RAISE NOTICE 'Added embedding column (384 dimensions)';
    ELSE
        RAISE NOTICE 'Embedding column already exists';
    END IF;
END $$;

-- =====================================================
-- 3. ADD SECTION_TITLE COLUMN (if not exists)
-- =====================================================
-- Add section title column for better content organization
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'nelson_textbook_chunks' 
        AND column_name = 'section_title'
    ) THEN
        ALTER TABLE nelson_textbook_chunks 
        ADD COLUMN section_title text;
        
        RAISE NOTICE 'Added section_title column';
    ELSE
        RAISE NOTICE 'Section_title column already exists';
    END IF;
END $$;

-- =====================================================
-- 4. CREATE VECTOR SIMILARITY SEARCH FUNCTION
-- =====================================================
-- Enhanced function for vector similarity search with filtering
CREATE OR REPLACE FUNCTION search_embeddings_enhanced(
  query_embedding vector(384),
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
-- 5. CREATE SECTION-BASED SEARCH FUNCTION
-- =====================================================
-- Search within specific sections with optional text query
CREATE OR REPLACE FUNCTION search_by_section(
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
    -- Search with text query within section
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
    -- Just return all content from section
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
-- 6. CREATE SECTION STATISTICS FUNCTION
-- =====================================================
-- Get statistics about section titles and their distribution
CREATE OR REPLACE FUNCTION get_section_statistics()
RETURNS TABLE (
  section_title text,
  document_count bigint,
  avg_page_number float,
  min_page integer,
  max_page integer,
  chapters_covered bigint,
  avg_content_length float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    n.section_title,
    COUNT(*) AS document_count,
    AVG(n.page_number::float) AS avg_page_number,
    MIN(n.page_number) AS min_page,
    MAX(n.page_number) AS max_page,
    COUNT(DISTINCT n.chapter_title) AS chapters_covered,
    AVG(LENGTH(n.content)) AS avg_content_length
  FROM nelson_textbook_chunks n
  WHERE n.section_title IS NOT NULL
  GROUP BY n.section_title
  ORDER BY document_count DESC;
END;
$$;

-- =====================================================
-- 7. CREATE EMBEDDING STATISTICS FUNCTION
-- =====================================================
-- Get statistics about embedding coverage and quality
CREATE OR REPLACE FUNCTION get_embedding_statistics()
RETURNS TABLE (
  total_documents bigint,
  documents_with_embeddings bigint,
  embedding_coverage_percent numeric,
  documents_with_sections bigint,
  section_coverage_percent numeric,
  avg_embedding_dimension float,
  last_embedding_update timestamptz
)
LANGUAGE plpgsql
AS $$
DECLARE
  total_docs bigint;
  docs_with_embeddings bigint;
  docs_with_sections bigint;
BEGIN
  -- Get counts
  SELECT COUNT(*) INTO total_docs FROM nelson_textbook_chunks;
  SELECT COUNT(*) INTO docs_with_embeddings FROM nelson_textbook_chunks WHERE embedding IS NOT NULL;
  SELECT COUNT(*) INTO docs_with_sections FROM nelson_textbook_chunks WHERE section_title IS NOT NULL;
  
  RETURN QUERY
  SELECT
    total_docs,
    docs_with_embeddings,
    ROUND((docs_with_embeddings::numeric / NULLIF(total_docs, 0)) * 100, 2) AS embedding_coverage_percent,
    docs_with_sections,
    ROUND((docs_with_sections::numeric / NULLIF(total_docs, 0)) * 100, 2) AS section_coverage_percent,
    384.0 AS avg_embedding_dimension,  -- Known dimension for all-MiniLM-L6-v2
    (SELECT MAX(created_at) FROM nelson_textbook_chunks WHERE embedding IS NOT NULL) AS last_embedding_update;
END;
$$;

-- =====================================================
-- 8. CREATE HYBRID SEARCH FUNCTION
-- =====================================================
-- Combine vector similarity and text search for best results
CREATE OR REPLACE FUNCTION hybrid_search(
  search_query text,
  query_embedding vector(384) DEFAULT NULL,
  match_count int DEFAULT 10,
  similarity_weight float DEFAULT 0.7,
  text_weight float DEFAULT 0.3,
  min_page_filter int DEFAULT NULL,
  max_page_filter int DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  content text,
  chapter_title text,
  section_title text,
  page_number integer,
  chunk_index integer,
  combined_score float,
  similarity_score float,
  text_score float,
  created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
  IF query_embedding IS NOT NULL THEN
    -- Hybrid search with both vector and text
    RETURN QUERY
    SELECT
      n.id,
      n.content,
      n.chapter_title,
      n.section_title,
      n.page_number,
      n.chunk_index,
      (similarity_weight * (1 - (n.embedding <=> query_embedding)) + 
       text_weight * ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', search_query))) AS combined_score,
      (1 - (n.embedding <=> query_embedding)) AS similarity_score,
      ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', search_query)) AS text_score,
      n.created_at
    FROM nelson_textbook_chunks n
    WHERE n.embedding IS NOT NULL
      AND to_tsvector('english', n.content) @@ plainto_tsquery('english', search_query)
      AND (min_page_filter IS NULL OR n.page_number >= min_page_filter)
      AND (max_page_filter IS NULL OR n.page_number <= max_page_filter)
    ORDER BY combined_score DESC
    LIMIT match_count;
  ELSE
    -- Text-only search
    RETURN QUERY
    SELECT
      n.id,
      n.content,
      n.chapter_title,
      n.section_title,
      n.page_number,
      n.chunk_index,
      ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', search_query)) AS combined_score,
      0.0 AS similarity_score,
      ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', search_query)) AS text_score,
      n.created_at
    FROM nelson_textbook_chunks n
    WHERE to_tsvector('english', n.content) @@ plainto_tsquery('english', search_query)
      AND (min_page_filter IS NULL OR n.page_number >= min_page_filter)
      AND (max_page_filter IS NULL OR n.page_number <= max_page_filter)
    ORDER BY combined_score DESC
    LIMIT match_count;
  END IF;
END;
$$;

-- =====================================================
-- 9. CREATE PERFORMANCE INDEXES
-- =====================================================
-- Indexes for optimal performance with embeddings and sections

-- Vector similarity index (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_nelson_embedding_hnsw 
ON nelson_textbook_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat index for exact vector search (alternative)
CREATE INDEX IF NOT EXISTS idx_nelson_embedding_ivfflat 
ON nelson_textbook_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Section title index for fast section-based queries
CREATE INDEX IF NOT EXISTS idx_nelson_section_title 
ON nelson_textbook_chunks (section_title) 
WHERE section_title IS NOT NULL;

-- Combined section and page index
CREATE INDEX IF NOT EXISTS idx_nelson_section_page 
ON nelson_textbook_chunks (section_title, page_number) 
WHERE section_title IS NOT NULL AND page_number IS NOT NULL;

-- Embedding existence index
CREATE INDEX IF NOT EXISTS idx_nelson_has_embedding 
ON nelson_textbook_chunks (id) 
WHERE embedding IS NOT NULL;

-- =====================================================
-- 10. UPDATE TABLE COMMENTS
-- =====================================================
-- Add helpful comments to the table and columns
COMMENT ON TABLE nelson_textbook_chunks IS 'Medical content from Nelson Textbook of Pediatrics with AI embeddings and section organization';

COMMENT ON COLUMN nelson_textbook_chunks.embedding IS 'AI-generated vector embedding (384D) using sentence-transformers/all-MiniLM-L6-v2 for semantic search';

COMMENT ON COLUMN nelson_textbook_chunks.section_title IS 'Extracted section title from medical content for better organization and filtering';

COMMENT ON COLUMN nelson_textbook_chunks.page_number IS 'Page number reference from Nelson Textbook of Pediatrics for accurate citations';

-- =====================================================
-- 11. GRANT PERMISSIONS
-- =====================================================
-- Ensure proper permissions for the new functions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon;

-- =====================================================
-- 12. VERIFICATION QUERIES
-- =====================================================
-- Run these to verify the schema updates

-- Check if columns exist
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'nelson_textbook_chunks' 
AND column_name IN ('embedding', 'section_title')
ORDER BY column_name;

-- Check embedding statistics
SELECT * FROM get_embedding_statistics();

-- Check section statistics (top 10)
SELECT * FROM get_section_statistics() LIMIT 10;

-- Test vector search function (if embeddings exist)
-- SELECT * FROM search_embeddings_enhanced(
--   '[0.1, 0.2, ...]'::vector(384),  -- Replace with actual embedding
--   0.1, 5
-- );

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================
DO $$ 
BEGIN
    RAISE NOTICE '✅ Database schema updated successfully!';
    RAISE NOTICE '🤖 Vector embeddings support added (384D)';
    RAISE NOTICE '📝 Section titles support added';
    RAISE NOTICE '🔍 Enhanced search functions created';
    RAISE NOTICE '📊 Performance indexes created';
    RAISE NOTICE '🚀 Ready for AI-powered semantic search!';
END $$;


#!/usr/bin/env python3
"""
Fix Supabase Schema for 384-dimensional Embeddings

This script updates your Supabase table to work with 384-dimensional embeddings
from sentence-transformers instead of 1536-dimensional OpenAI embeddings.
"""

import os
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def main():
    """Main function to fix the schema"""
    
    print("🔧 Fixing Supabase Schema for 384-dimensional Embeddings")
    print("=" * 60)
    
    try:
        # Connect to Supabase
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("✅ Connected to Supabase")
        
        # Check current table structure
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        current_count = result.count if result.count else 0
        
        print(f"📊 Current records in table: {current_count}")
        
        if current_count > 0:
            response = input("⚠️ Table has existing data. Do you want to clear it and recreate? (y/N): ")
            if response.lower() != 'y':
                print("❌ Aborting. Please manually update the schema or clear the table.")
                return
        
        print("\n🗑️ Clearing existing table...")
        
        # Clear existing data
        if current_count > 0:
            supabase.table('nelson_textbook_chunks').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            logger.info("✅ Existing data cleared")
        
        print("\n📋 SQL Commands to run in Supabase SQL Editor:")
        print("=" * 60)
        
        # SQL to recreate table with correct embedding dimensions
        sql_commands = """
-- Drop existing indexes
DROP INDEX IF EXISTS idx_nelson_textbook_embedding;

-- Recreate the table with 384-dimensional embeddings
DROP TABLE IF EXISTS nelson_textbook_chunks;

CREATE TABLE public.nelson_textbook_chunks (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  chapter_title text NOT NULL,
  section_title text NULL,
  content text NOT NULL,
  page_number integer NULL,
  chunk_index integer NOT NULL,
  embedding public.vector(384) NULL,  -- Changed to 384 dimensions
  metadata jsonb NULL DEFAULT '{}'::jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT nelson_textbook_chunks_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Create indexes optimized for 384-dimensional vectors
CREATE INDEX IF NOT EXISTS idx_nelson_textbook_embedding 
ON public.nelson_textbook_chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100) 
TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_nelson_textbook_chapter 
ON public.nelson_textbook_chunks 
USING btree (chapter_title) 
TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_nelson_textbook_page 
ON public.nelson_textbook_chunks 
USING btree (page_number) 
TABLESPACE pg_default;

-- Create a function for vector similarity search
CREATE OR REPLACE FUNCTION search_embeddings(
  query_embedding vector(384),
  match_threshold float DEFAULT 0.1,
  match_count int DEFAULT 10
)
RETURNS TABLE (
  id uuid,
  chapter_title text,
  section_title text,
  content text,
  chunk_index int,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    nelson_textbook_chunks.id,
    nelson_textbook_chunks.chapter_title,
    nelson_textbook_chunks.section_title,
    nelson_textbook_chunks.content,
    nelson_textbook_chunks.chunk_index,
    nelson_textbook_chunks.metadata,
    1 - (nelson_textbook_chunks.embedding <=> query_embedding) AS similarity
  FROM nelson_textbook_chunks
  WHERE nelson_textbook_chunks.embedding IS NOT NULL
    AND 1 - (nelson_textbook_chunks.embedding <=> query_embedding) > match_threshold
  ORDER BY nelson_textbook_chunks.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create function to get unique chapters
CREATE OR REPLACE FUNCTION get_unique_chapters()
RETURNS TABLE(chapter_title TEXT, record_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ntc.chapter_title,
        COUNT(*) as record_count
    FROM nelson_textbook_chunks ntc
    GROUP BY ntc.chapter_title
    ORDER BY record_count DESC, ntc.chapter_title;
END;
$$ LANGUAGE plpgsql;
"""
        
        print(sql_commands)
        print("=" * 60)
        
        print("\n📝 Instructions:")
        print("1. Copy the SQL commands above")
        print("2. Go to your Supabase Dashboard > SQL Editor")
        print("3. Paste and run the SQL commands")
        print("4. Come back and run the upload script again")
        
        print("\n🔗 Supabase Dashboard: https://supabase.com/dashboard/project/nrtaztkewvbtzhbtkffc")
        
        print("\n✅ Schema fix prepared! Run the SQL commands in Supabase, then upload your data.")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()


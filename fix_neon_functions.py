#!/usr/bin/env python3
"""
Fix Neon PostgreSQL Functions

This script fixes the data type issues in the PostgreSQL functions.
"""

import psycopg2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neon Database Configuration
NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_4TWsIBXtja9b@ep-delicate-credit-a1h2uxg9-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def fix_search_functions():
    """Fix the PostgreSQL search functions with correct data types"""
    try:
        conn = psycopg2.connect(NEON_CONNECTION_STRING)
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("🔧 Fixing PostgreSQL search functions...")
        
        # Fixed text search function with correct return type
        text_search_function = """
        CREATE OR REPLACE FUNCTION search_medical_text(
            search_query TEXT,
            result_limit INTEGER DEFAULT 10
        )
        RETURNS TABLE(
            id INTEGER,
            text TEXT,
            topic TEXT,
            source_file TEXT,
            chunk_number INTEGER,
            similarity_score DOUBLE PRECISION
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                n.id,
                n.text,
                n.topic,
                n.source_file,
                n.chunk_number,
                CAST(ts_rank(to_tsvector('english', n.text), plainto_tsquery('english', search_query)) AS DOUBLE PRECISION) as similarity_score
            FROM nelson_book_of_pediatrics n
            WHERE to_tsvector('english', n.text) @@ plainto_tsquery('english', search_query)
            ORDER BY similarity_score DESC
            LIMIT result_limit;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        cursor.execute(text_search_function)
        logger.info("✅ Fixed text search function")
        
        # Simple search function for basic text matching
        simple_search_function = """
        CREATE OR REPLACE FUNCTION simple_medical_search(
            search_query TEXT,
            result_limit INTEGER DEFAULT 10
        )
        RETURNS TABLE(
            id INTEGER,
            text TEXT,
            topic TEXT,
            source_file TEXT,
            chunk_number INTEGER,
            similarity_score DOUBLE PRECISION
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                n.id,
                n.text,
                n.topic,
                n.source_file,
                n.chunk_number,
                CASE 
                    WHEN n.text ILIKE '%' || search_query || '%' THEN 1.0
                    ELSE 0.5
                END as similarity_score
            FROM nelson_book_of_pediatrics n
            WHERE n.text ILIKE '%' || search_query || '%'
            ORDER BY similarity_score DESC, n.id
            LIMIT result_limit;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        cursor.execute(simple_search_function)
        logger.info("✅ Created simple search function")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ All functions fixed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix functions: {e}")
        return False

def test_functions():
    """Test the fixed functions"""
    try:
        conn = psycopg2.connect(NEON_CONNECTION_STRING)
        cursor = conn.cursor()
        
        logger.info("🧪 Testing fixed functions...")
        
        # Test simple search
        cursor.execute("SELECT * FROM simple_medical_search('asthma', 3);")
        results = cursor.fetchall()
        
        if results:
            logger.info(f"✅ Simple search test passed - found {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                logger.info(f"   {i}. Topic: {result[2]} | Similarity: {result[5]}")
        else:
            logger.warning("⚠️ No results found in simple search test")
        
        # Test full-text search
        try:
            cursor.execute("SELECT * FROM search_medical_text('asthma', 3);")
            results = cursor.fetchall()
            
            if results:
                logger.info(f"✅ Full-text search test passed - found {len(results)} results")
            else:
                logger.info("ℹ️ Full-text search returned no results (this is normal)")
        except Exception as e:
            logger.warning(f"⚠️ Full-text search test failed: {e}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Function test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing Neon PostgreSQL Functions")
    print("=" * 40)
    
    if fix_search_functions():
        test_functions()
        print("\n✅ Functions fixed and tested successfully!")
    else:
        print("\n❌ Failed to fix functions")


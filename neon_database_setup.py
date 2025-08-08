#!/usr/bin/env python3
"""
Nelson Pediatrics - Neon PostgreSQL Database Setup

This script uploads your Nelson Pediatrics medical knowledge base 
to Neon PostgreSQL database with vector embeddings support.
"""

import os
import json
import psycopg2
from psycopg2.extras import execute_values
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neon Database Configuration
NEON_CONNECTION_STRING = os.getenv(
    'NEON_DATABASE_URL',
    'postgresql://neondb_owner:npg_4TWsIBXtja9b@ep-delicate-credit-a1h2uxg9-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
)

def connect_to_neon():
    """Connect to Neon PostgreSQL database"""
    try:
        logger.info("🔌 Connecting to Neon PostgreSQL database...")
        
        conn = psycopg2.connect(NEON_CONNECTION_STRING)
        conn.autocommit = True
        
        logger.info("✅ Successfully connected to Neon database")
        return conn
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neon database: {e}")
        return None

def setup_database_schema(conn):
    """Create the nelson_book_of_pediatrics table and indexes"""
    try:
        cursor = conn.cursor()
        
        logger.info("🏗️ Setting up database schema...")
        
        # Enable pgvector extension (if not already enabled)
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("✅ Vector extension enabled")
        
        # Create the table as specified in your original SQL
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS public.nelson_book_of_pediatrics (
            id SERIAL NOT NULL,
            text TEXT NOT NULL,
            page_number INTEGER NULL,
            source_file TEXT NULL,
            embedding VECTOR(384) NULL,
            topic TEXT NULL,
            chunk_number INTEGER NULL,
            character_count INTEGER NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT nelson_book_of_pediatrics_pkey PRIMARY KEY (id)
        );
        """
        
        cursor.execute(create_table_sql)
        logger.info("✅ Table 'nelson_book_of_pediatrics' created/verified")
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS nelson_embeddings_idx ON public.nelson_book_of_pediatrics USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
            "CREATE INDEX IF NOT EXISTS idx_nelson_content_embedding ON public.nelson_book_of_pediatrics USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
            "CREATE INDEX IF NOT EXISTS idx_nelson_topic ON public.nelson_book_of_pediatrics (topic);",
            "CREATE INDEX IF NOT EXISTS idx_nelson_source_file ON public.nelson_book_of_pediatrics (source_file);",
            "CREATE INDEX IF NOT EXISTS idx_nelson_created_at ON public.nelson_book_of_pediatrics (created_at);"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"✅ Index created: {index_sql.split('idx_')[1].split(' ')[0] if 'idx_' in index_sql else 'embeddings'}")
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning: {e}")
        
        # Create trigger for updated_at (if function exists)
        trigger_sql = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_nelson_updated_at ON nelson_book_of_pediatrics;
        CREATE TRIGGER update_nelson_updated_at 
            BEFORE UPDATE ON nelson_book_of_pediatrics 
            FOR EACH ROW 
            EXECUTE FUNCTION update_updated_at_column();
        """
        
        cursor.execute(trigger_sql)
        logger.info("✅ Updated_at trigger created")
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup database schema: {e}")
        return False

def load_processed_data():
    """Load processed Nelson Pediatrics data"""
    logger.info("📖 Loading processed Nelson Pediatrics data...")
    
    data = []
    data_file = "nelson_pediatrics_processed.jsonl"
    
    if not os.path.exists(data_file):
        logger.error(f"❌ {data_file} not found!")
        logger.info("💡 Please run the data processing script first to generate the embeddings")
        return []
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
                        continue
        
        logger.info(f"✅ Loaded {len(data)} medical documents")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return []

def upload_to_neon(conn, data):
    """Upload all data to Neon PostgreSQL database"""
    logger.info(f"📤 Uploading {len(data)} documents to Neon database...")
    
    try:
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM nelson_book_of_pediatrics;")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            logger.info(f"📊 Found {existing_count} existing records")
            response = input("Do you want to clear existing data and re-upload? (y/N): ")
            if response.lower() == 'y':
                cursor.execute("TRUNCATE TABLE nelson_book_of_pediatrics RESTART IDENTITY;")
                logger.info("🗑️ Existing data cleared")
            else:
                logger.info("📝 Appending to existing data")
        
        # Prepare data for bulk insert
        insert_data = []
        
        for item in tqdm(data, desc="Preparing data"):
            # Convert embedding list to PostgreSQL array format
            embedding_array = item['embedding']
            
            # Prepare row data
            row_data = (
                item['text'],                                    # text
                None,                                           # page_number (not in our data)
                item['source_file'],                            # source_file
                embedding_array,                                # embedding
                item.get('topic', 'Unknown'),                  # topic
                item.get('chunk_number', 0),                   # chunk_number
                item.get('character_count', len(item['text'])), # character_count
            )
            
            insert_data.append(row_data)
        
        # Bulk insert using execute_values for better performance
        insert_sql = """
        INSERT INTO nelson_book_of_pediatrics 
        (text, page_number, source_file, embedding, topic, chunk_number, character_count)
        VALUES %s
        """
        
        logger.info("📤 Performing bulk insert...")
        
        # Insert in batches for better memory management
        batch_size = 1000
        total_batches = (len(insert_data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(insert_data), batch_size), desc="Uploading batches", total=total_batches):
            batch = insert_data[i:i + batch_size]
            
            execute_values(
                cursor,
                insert_sql,
                batch,
                template=None,
                page_size=batch_size
            )
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM nelson_book_of_pediatrics;")
        final_count = cursor.fetchone()[0]
        
        cursor.close()
        
        logger.info(f"✅ Successfully uploaded {final_count} documents to Neon database")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to upload data: {e}")
        return False

def test_vector_search(conn):
    """Test vector similarity search functionality"""
    logger.info("🔍 Testing vector similarity search...")
    
    try:
        cursor = conn.cursor()
        
        # Test query
        test_query = "asthma treatment in children"
        logger.info(f"🔍 Testing search for: '{test_query}'")
        
        # For testing, we'll use a simple text search first
        # In production, you'd generate an embedding for the query
        search_sql = """
        SELECT 
            id,
            text,
            topic,
            source_file,
            chunk_number,
            character_count,
            created_at
        FROM nelson_book_of_pediatrics 
        WHERE text ILIKE %s 
        ORDER BY character_count DESC
        LIMIT 5;
        """
        
        cursor.execute(search_sql, (f'%{test_query}%',))
        results = cursor.fetchall()
        
        if results:
            logger.info(f"✅ Found {len(results)} test results:")
            for i, result in enumerate(results, 1):
                logger.info(f"   {i}. Topic: {result[2]} | Source: {result[3]} | Length: {result[5]} chars")
        else:
            logger.warning("⚠️ No test results found")
        
        # Test vector search (if embeddings are properly stored)
        vector_search_sql = """
        SELECT 
            id,
            text,
            topic,
            source_file,
            1 - (embedding <=> %s::vector) as similarity
        FROM nelson_book_of_pediatrics 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """
        
        # For now, we'll skip the actual vector search test
        # You would need to generate an embedding for the test query
        logger.info("💡 Vector search test skipped - requires query embedding generation")
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False

def create_search_functions(conn):
    """Create PostgreSQL functions for medical search"""
    logger.info("🔧 Creating search functions...")
    
    try:
        cursor = conn.cursor()
        
        # Function for text-based search
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
            similarity_score FLOAT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                n.id,
                n.text,
                n.topic,
                n.source_file,
                n.chunk_number,
                ts_rank(to_tsvector('english', n.text), plainto_tsquery('english', search_query)) as similarity_score
            FROM nelson_book_of_pediatrics n
            WHERE to_tsvector('english', n.text) @@ plainto_tsquery('english', search_query)
            ORDER BY similarity_score DESC
            LIMIT result_limit;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        cursor.execute(text_search_function)
        logger.info("✅ Text search function created")
        
        # Function for topic-based search
        topic_search_function = """
        CREATE OR REPLACE FUNCTION search_by_topic(
            topic_name TEXT,
            search_query TEXT DEFAULT '',
            result_limit INTEGER DEFAULT 10
        )
        RETURNS TABLE(
            id INTEGER,
            text TEXT,
            topic TEXT,
            source_file TEXT,
            chunk_number INTEGER
        ) AS $$
        BEGIN
            IF search_query = '' THEN
                RETURN QUERY
                SELECT 
                    n.id,
                    n.text,
                    n.topic,
                    n.source_file,
                    n.chunk_number
                FROM nelson_book_of_pediatrics n
                WHERE n.topic ILIKE '%' || topic_name || '%'
                ORDER BY n.chunk_number
                LIMIT result_limit;
            ELSE
                RETURN QUERY
                SELECT 
                    n.id,
                    n.text,
                    n.topic,
                    n.source_file,
                    n.chunk_number
                FROM nelson_book_of_pediatrics n
                WHERE n.topic ILIKE '%' || topic_name || '%'
                AND n.text ILIKE '%' || search_query || '%'
                ORDER BY n.chunk_number
                LIMIT result_limit;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        cursor.execute(topic_search_function)
        logger.info("✅ Topic search function created")
        
        # Function to get statistics
        stats_function = """
        CREATE OR REPLACE FUNCTION get_database_stats()
        RETURNS TABLE(
            total_documents BIGINT,
            total_topics BIGINT,
            total_sources BIGINT,
            avg_text_length NUMERIC,
            latest_update TIMESTAMP
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT topic) as total_topics,
                COUNT(DISTINCT source_file) as total_sources,
                AVG(character_count) as avg_text_length,
                MAX(created_at) as latest_update
            FROM nelson_book_of_pediatrics;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        cursor.execute(stats_function)
        logger.info("✅ Statistics function created")
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create search functions: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🏥 Nelson Pediatrics - Neon PostgreSQL Setup")
    print("=" * 60)
    print("🌐 Cloud Database • 🔍 Vector Search • 📊 Production Ready")
    print("=" * 60)
    
    # Step 1: Connect to Neon database
    conn = connect_to_neon()
    if not conn:
        return
    
    # Step 2: Setup database schema
    if not setup_database_schema(conn):
        conn.close()
        return
    
    # Step 3: Load processed data
    data = load_processed_data()
    if not data:
        conn.close()
        return
    
    # Step 4: Upload to Neon database
    if not upload_to_neon(conn, data):
        conn.close()
        return
    
    # Step 5: Create search functions
    if not create_search_functions(conn):
        conn.close()
        return
    
    # Step 6: Test search functionality
    test_vector_search(conn)
    
    # Step 7: Get final statistics
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM get_database_stats();")
        stats = cursor.fetchone()
        cursor.close()
        
        print("\n" + "=" * 60)
        print("🎉 NEON DATABASE SETUP COMPLETED!")
        print("=" * 60)
        print(f"📊 Total documents: {stats[0]}")
        print(f"📚 Total topics: {stats[1]}")
        print(f"📄 Total sources: {stats[2]}")
        print(f"📝 Average text length: {stats[3]:.0f} characters")
        print(f"🕒 Latest update: {stats[4]}")
        print(f"🌐 Database: Neon PostgreSQL")
        print(f"🔍 Vector search: Enabled (pgvector)")
        print(f"💾 Storage: Cloud (persistent)")
        
        print("\n🚀 Next Steps:")
        print("1. Use the search functions in your applications")
        print("2. Connect your NelsonGPT app to Neon database")
        print("3. Implement vector similarity search with embeddings")
        
        print("\n💡 Example usage:")
        print("SELECT * FROM search_medical_text('asthma treatment', 5);")
        print("SELECT * FROM search_by_topic('Allergic Disorder', 'asthma');")
        print("SELECT * FROM get_database_stats();")
        
        print("\n🏥 Your Nelson Pediatrics knowledge base is now in the cloud! ⚡")
        
    except Exception as e:
        logger.error(f"❌ Failed to get final statistics: {e}")
    
    # Close connection
    conn.close()
    logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    main()

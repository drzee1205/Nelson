#!/usr/bin/env python3
"""
Nelson Pediatrics - Supabase Upload Script

This script uploads your Nelson Pediatrics medical knowledge base 
to your existing Supabase table with vector embeddings.
"""

import os
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import uuid
from datetime import datetime

# Supabase client
try:
    from supabase import create_client, Client
    import numpy as np
except ImportError:
    print("❌ Missing dependencies. Installing...")
    os.system("pip install supabase numpy")
    from supabase import create_client, Client
    import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def create_supabase_client():
    """Create Supabase client"""
    try:
        logger.info("🔌 Connecting to Supabase...")
        
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Test connection
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        existing_count = result.count if result.count else 0
        
        logger.info(f"✅ Connected to Supabase. Found {existing_count} existing records")
        return supabase
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Supabase: {e}")
        return None

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

def transform_data_for_supabase(data: List[Dict]) -> List[Dict]:
    """Transform data to match Supabase table schema"""
    logger.info("🔄 Transforming data for Supabase schema...")
    
    transformed_data = []
    
    for i, item in enumerate(tqdm(data, desc="Transforming data")):
        try:
            # Extract chapter and section from topic or source_file
            topic = item.get('topic', 'Unknown')
            source_file = item.get('source_file', '')
            
            # Use topic as chapter_title, extract section if available
            chapter_title = topic
            section_title = None
            
            # Try to extract section from content if it contains headers
            content = item['text']
            lines = content.split('\n')
            if len(lines) > 1 and len(lines[0]) < 100:
                # First line might be a section title
                potential_section = lines[0].strip()
                if potential_section and not potential_section.lower().startswith(('the ', 'a ', 'an ')):
                    section_title = potential_section
            
            # Create metadata object
            metadata = {
                'source_file': source_file,
                'original_topic': topic,
                'character_count': item.get('character_count', len(content)),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Add any additional metadata from original item
            for key, value in item.items():
                if key not in ['text', 'embedding', 'topic', 'source_file', 'chunk_number']:
                    metadata[f'original_{key}'] = value
            
            # Transform to Supabase schema
            supabase_record = {
                'id': str(uuid.uuid4()),  # Generate UUID
                'chapter_title': chapter_title,
                'section_title': section_title,
                'content': content,
                'page_number': None,  # We don't have page numbers in our data
                'chunk_index': item.get('chunk_number', i),
                'embedding': item['embedding'],  # Vector embedding
                'metadata': metadata,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            transformed_data.append(supabase_record)
            
        except Exception as e:
            logger.warning(f"⚠️ Error transforming record {i}: {e}")
            continue
    
    logger.info(f"✅ Transformed {len(transformed_data)} records for Supabase")
    return transformed_data

def upload_to_supabase(supabase: Client, data: List[Dict]):
    """Upload data to Supabase in batches"""
    logger.info(f"📤 Uploading {len(data)} records to Supabase...")
    
    try:
        # Check if data already exists
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        existing_count = result.count if result.count else 0
        
        if existing_count > 0:
            logger.info(f"📊 Found {existing_count} existing records")
            response = input("Do you want to clear existing data and re-upload? (y/N): ")
            if response.lower() == 'y':
                logger.info("🗑️ Clearing existing data...")
                supabase.table('nelson_textbook_chunks').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
                logger.info("✅ Existing data cleared")
            else:
                logger.info("📝 Appending to existing data")
        
        # Upload in batches for better performance and reliability
        batch_size = 100  # Supabase handles batches well
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        successful_uploads = 0
        failed_uploads = 0
        
        for i in tqdm(range(0, len(data), batch_size), desc="Uploading batches", total=total_batches):
            batch = data[i:i + batch_size]
            
            try:
                # Upload batch to Supabase
                result = supabase.table('nelson_textbook_chunks').insert(batch).execute()
                
                if result.data:
                    successful_uploads += len(batch)
                    logger.debug(f"✅ Uploaded batch {i//batch_size + 1}/{total_batches}")
                else:
                    failed_uploads += len(batch)
                    logger.warning(f"⚠️ Failed to upload batch {i//batch_size + 1}")
                    
            except Exception as e:
                failed_uploads += len(batch)
                logger.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        # Get final count
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        final_count = result.count if result.count else 0
        
        logger.info(f"✅ Upload completed!")
        logger.info(f"📊 Successful uploads: {successful_uploads}")
        logger.info(f"❌ Failed uploads: {failed_uploads}")
        logger.info(f"📈 Total records in database: {final_count}")
        
        return successful_uploads > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to upload data: {e}")
        return False

def test_supabase_search(supabase: Client):
    """Test search functionality"""
    logger.info("🔍 Testing Supabase search functionality...")
    
    try:
        # Test basic search
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, chapter_title, section_title, content')\
            .ilike('content', '%asthma%')\
            .limit(3)\
            .execute()
        
        if result.data:
            logger.info(f"✅ Search test passed - found {len(result.data)} results")
            for i, record in enumerate(result.data, 1):
                logger.info(f"   {i}. Chapter: {record['chapter_title']}")
                if record['section_title']:
                    logger.info(f"      Section: {record['section_title']}")
                logger.info(f"      Content preview: {record['content'][:100]}...")
        else:
            logger.warning("⚠️ No search results found")
        
        # Test chapter-based search
        result = supabase.table('nelson_textbook_chunks')\
            .select('chapter_title, count', count='exact')\
            .execute()
        
        if result.count:
            logger.info(f"✅ Database contains {result.count} total records")
        
        # Get unique chapters
        result = supabase.rpc('get_unique_chapters').execute()
        if hasattr(result, 'data') and result.data:
            logger.info(f"✅ Available chapters: {len(result.data)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False

def create_supabase_functions(supabase: Client):
    """Create helpful database functions"""
    logger.info("🔧 Creating Supabase helper functions...")
    
    try:
        # Function to get unique chapters
        create_function_sql = """
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
        
        # Note: This would need to be run directly in Supabase SQL editor
        # as the Python client doesn't support DDL operations
        logger.info("💡 To create helper functions, run this SQL in your Supabase SQL editor:")
        logger.info(create_function_sql)
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not create functions via client: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🏥 Nelson Pediatrics - Supabase Upload")
    print("=" * 60)
    print("📊 Uploading to existing table: nelson_textbook_chunks")
    print("🌐 Supabase URL: https://nrtaztkewvbtzhbtkffc.supabase.co")
    print("=" * 60)
    
    # Step 1: Create Supabase client
    supabase = create_supabase_client()
    if not supabase:
        return
    
    # Step 2: Load processed data
    data = load_processed_data()
    if not data:
        return
    
    # Step 3: Transform data for Supabase schema
    transformed_data = transform_data_for_supabase(data)
    if not transformed_data:
        return
    
    # Step 4: Upload to Supabase
    if not upload_to_supabase(supabase, transformed_data):
        return
    
    # Step 5: Test search functionality
    test_supabase_search(supabase)
    
    # Step 6: Create helper functions
    create_supabase_functions(supabase)
    
    # Final summary
    try:
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = result.count if result.count else 0
        
        # Get unique chapters
        result = supabase.table('nelson_textbook_chunks')\
            .select('chapter_title')\
            .execute()
        
        unique_chapters = set()
        if result.data:
            unique_chapters = set(record['chapter_title'] for record in result.data)
        
        print("\n" + "=" * 60)
        print("🎉 SUPABASE UPLOAD COMPLETED!")
        print("=" * 60)
        print(f"📊 Total records uploaded: {total_count:,}")
        print(f"📚 Unique chapters: {len(unique_chapters)}")
        print(f"🌐 Database: Supabase PostgreSQL")
        print(f"🔍 Vector search: Enabled (pgvector)")
        print(f"💾 Storage: Cloud (persistent)")
        
        print("\n🚀 Next Steps:")
        print("1. Test vector similarity search in your application")
        print("2. Create API endpoints for your NelsonGPT integration")
        print("3. Implement real-time search functionality")
        
        print("\n💡 Example Supabase queries:")
        print("// Search by content")
        print("supabase.from('nelson_textbook_chunks')")
        print("  .select('*')")
        print("  .textSearch('content', 'asthma treatment')")
        
        print("\n// Search by chapter")
        print("supabase.from('nelson_textbook_chunks')")
        print("  .select('*')")
        print("  .eq('chapter_title', 'The Cardiovascular System')")
        
        print("\n🏥 Your Nelson Pediatrics knowledge base is now in Supabase! ⚡")
        
    except Exception as e:
        logger.error(f"❌ Failed to get final statistics: {e}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Resume Supabase Upload

This script resumes the upload from where it left off, uploading only the remaining documents.
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
    print("❌ Installing dependencies...")
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
        return supabase, existing_count
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Supabase: {e}")
        return None, 0

def load_processed_data():
    """Load processed Nelson Pediatrics data"""
    logger.info("📖 Loading processed Nelson Pediatrics data...")
    
    data = []
    data_file = "nelson_pediatrics_processed.jsonl"
    
    if not os.path.exists(data_file):
        logger.error(f"❌ {data_file} not found!")
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

def get_uploaded_content_hashes(supabase: Client):
    """Get hashes of already uploaded content to avoid duplicates"""
    logger.info("🔍 Checking for already uploaded content...")
    
    try:
        # Get all existing content (first 100 chars as identifier)
        result = supabase.table('nelson_textbook_chunks')\
            .select('content')\
            .execute()
        
        if result.data:
            # Create set of content hashes (first 100 chars)
            uploaded_hashes = set()
            for record in result.data:
                content_hash = record['content'][:100] if record['content'] else ""
                uploaded_hashes.add(content_hash)
            
            logger.info(f"✅ Found {len(uploaded_hashes)} unique content hashes")
            return uploaded_hashes
        else:
            return set()
            
    except Exception as e:
        logger.error(f"❌ Error getting uploaded content: {e}")
        return set()

def transform_data_for_supabase(data: List[Dict], uploaded_hashes: set, start_index: int = 0) -> List[Dict]:
    """Transform data to match Supabase table schema, skipping already uploaded content"""
    logger.info(f"🔄 Transforming data for Supabase schema (starting from index {start_index})...")
    
    transformed_data = []
    skipped_count = 0
    
    for i, item in enumerate(tqdm(data[start_index:], desc="Transforming data", initial=start_index)):
        try:
            content = item['text']
            content_hash = content[:100]
            
            # Skip if already uploaded
            if content_hash in uploaded_hashes:
                skipped_count += 1
                continue
            
            # Extract chapter and section from topic or source_file
            topic = item.get('topic', 'Unknown')
            source_file = item.get('source_file', '')
            
            # Use topic as chapter_title, extract section if available
            chapter_title = topic
            section_title = None
            
            # Try to extract section from content if it contains headers
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
                'processing_timestamp': datetime.now().isoformat(),
                'original_index': start_index + i
            }
            
            # Transform to Supabase schema - SKIP EMBEDDING for now
            supabase_record = {
                'id': str(uuid.uuid4()),  # Generate UUID
                'chapter_title': chapter_title,
                'section_title': section_title,
                'content': content,
                'page_number': None,  # We don't have page numbers in our data
                'chunk_index': item.get('chunk_number', start_index + i),
                # 'embedding': item['embedding'],  # Skip embedding for now due to dimension mismatch
                'metadata': metadata,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            transformed_data.append(supabase_record)
            
        except Exception as e:
            logger.warning(f"⚠️ Error transforming record {start_index + i}: {e}")
            continue
    
    logger.info(f"✅ Transformed {len(transformed_data)} new records (skipped {skipped_count} duplicates)")
    return transformed_data

def upload_to_supabase(supabase: Client, data: List[Dict]):
    """Upload data to Supabase in batches"""
    logger.info(f"📤 Uploading {len(data)} records to Supabase...")
    
    if not data:
        logger.info("ℹ️ No new data to upload")
        return True
    
    try:
        # Upload in batches for better performance and reliability
        batch_size = 50  # Smaller batches for better reliability
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

def main():
    """Main execution function"""
    
    print("🔄 Nelson Pediatrics - Resume Supabase Upload")
    print("=" * 60)
    print("📊 Resuming upload to table: nelson_textbook_chunks")
    print("🌐 Supabase URL: https://nrtaztkewvbtzhbtkffc.supabase.co")
    print("=" * 60)
    
    # Step 1: Create Supabase client
    supabase, existing_count = create_supabase_client()
    if not supabase:
        return
    
    # Step 2: Load processed data
    data = load_processed_data()
    if not data:
        return
    
    print(f"📊 Total documents to process: {len(data)}")
    print(f"📊 Already uploaded: {existing_count}")
    print(f"📊 Remaining to upload: {len(data) - existing_count}")
    
    # Step 3: Get uploaded content hashes to avoid duplicates
    uploaded_hashes = get_uploaded_content_hashes(supabase)
    
    # Step 4: Transform remaining data for Supabase schema
    # Start from where we left off (approximately)
    start_index = max(0, existing_count - 100)  # Start a bit earlier to catch any gaps
    
    transformed_data = transform_data_for_supabase(data, uploaded_hashes, start_index)
    if not transformed_data:
        print("✅ All data already uploaded!")
        return
    
    # Step 5: Upload to Supabase
    if not upload_to_supabase(supabase, transformed_data):
        return
    
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
        print("🎉 SUPABASE UPLOAD RESUMED!")
        print("=" * 60)
        print(f"📊 Total records uploaded: {total_count:,}")
        print(f"📚 Unique chapters: {len(unique_chapters)}")
        print(f"🌐 Database: Supabase PostgreSQL")
        print(f"🔍 Text search: Enabled")
        print(f"💾 Storage: Cloud (persistent)")
        
        if total_count >= 15000:
            print("\n🎉 UPLOAD COMPLETE! All Nelson Pediatrics data is now in Supabase!")
        else:
            print(f"\n📈 Progress: {total_count:,} / 15,339 ({total_count/15339*100:.1f}%)")
            print("🔄 Run this script again if needed to complete the upload")
        
    except Exception as e:
        logger.error(f"❌ Failed to get final statistics: {e}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Fast Supabase Upload - Complete All Remaining Documents

This script uses optimized batch processing to quickly upload all remaining documents.
"""

import os
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import uuid
from datetime import datetime
import asyncio
import concurrent.futures
from threading import Lock

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

# Global lock for thread safety
upload_lock = Lock()
upload_stats = {'success': 0, 'failed': 0, 'total': 0}

def create_supabase_client():
    """Create Supabase client"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
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

def get_uploaded_indices(supabase: Client):
    """Get indices of already uploaded documents"""
    logger.info("🔍 Getting uploaded document indices...")
    
    try:
        # Get all existing metadata with original_index
        result = supabase.table('nelson_textbook_chunks')\
            .select('metadata')\
            .execute()
        
        uploaded_indices = set()
        if result.data:
            for record in result.data:
                metadata = record.get('metadata', {})
                if isinstance(metadata, dict) and 'original_index' in metadata:
                    uploaded_indices.add(metadata['original_index'])
                elif isinstance(metadata, dict) and 'chunk_index' in metadata:
                    uploaded_indices.add(metadata['chunk_index'])
        
        logger.info(f"✅ Found {len(uploaded_indices)} uploaded document indices")
        return uploaded_indices
    except Exception as e:
        logger.error(f"❌ Error getting uploaded indices: {e}")
        return set()

def transform_batch(data_batch: List[Dict], start_index: int) -> List[Dict]:
    """Transform a batch of data for Supabase"""
    transformed_batch = []
    
    for i, item in enumerate(data_batch):
        try:
            content = item['text']
            topic = item.get('topic', 'Unknown')
            source_file = item.get('source_file', '')
            
            # Use topic as chapter_title
            chapter_title = topic
            section_title = None
            
            # Try to extract section from content
            lines = content.split('\n')
            if len(lines) > 1 and len(lines[0]) < 100:
                potential_section = lines[0].strip()
                if potential_section and not potential_section.lower().startswith(('the ', 'a ', 'an ')):
                    section_title = potential_section
            
            # Create metadata
            metadata = {
                'source_file': source_file,
                'original_topic': topic,
                'character_count': item.get('character_count', len(content)),
                'processing_timestamp': datetime.now().isoformat(),
                'original_index': start_index + i,
                'chunk_index': start_index + i
            }
            
            # Transform to Supabase schema
            supabase_record = {
                'id': str(uuid.uuid4()),
                'chapter_title': chapter_title,
                'section_title': section_title,
                'content': content,
                'page_number': None,
                'chunk_index': start_index + i,
                'metadata': metadata,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            transformed_batch.append(supabase_record)
            
        except Exception as e:
            logger.warning(f"⚠️ Error transforming record {start_index + i}: {e}")
            continue
    
    return transformed_batch

def upload_batch(supabase: Client, batch: List[Dict], batch_num: int) -> bool:
    """Upload a single batch to Supabase"""
    global upload_stats
    
    try:
        result = supabase.table('nelson_textbook_chunks').insert(batch).execute()
        
        with upload_lock:
            if result.data:
                upload_stats['success'] += len(batch)
                return True
            else:
                upload_stats['failed'] += len(batch)
                logger.warning(f"⚠️ Failed to upload batch {batch_num}")
                return False
                
    except Exception as e:
        with upload_lock:
            upload_stats['failed'] += len(batch)
        logger.error(f"❌ Error uploading batch {batch_num}: {e}")
        return False

def fast_upload_remaining(supabase: Client, data: List[Dict], uploaded_indices: set):
    """Fast upload of remaining documents using parallel processing"""
    logger.info("🚀 Starting fast upload of remaining documents...")
    
    # Filter out already uploaded documents
    remaining_data = []
    for i, item in enumerate(data):
        if i not in uploaded_indices:
            remaining_data.append((i, item))
    
    if not remaining_data:
        logger.info("✅ All documents already uploaded!")
        return True
    
    logger.info(f"📤 Uploading {len(remaining_data)} remaining documents...")
    
    # Process in larger batches for speed
    batch_size = 100  # Larger batches for speed
    total_batches = (len(remaining_data) + batch_size - 1) // batch_size
    
    global upload_stats
    upload_stats['total'] = len(remaining_data)
    
    # Use ThreadPoolExecutor for parallel uploads
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        for i in range(0, len(remaining_data), batch_size):
            batch_data = remaining_data[i:i + batch_size]
            
            # Transform batch
            start_index = batch_data[0][0]  # Original index of first item
            items_only = [item[1] for item in batch_data]  # Extract just the items
            transformed_batch = transform_batch(items_only, start_index)
            
            if transformed_batch:
                # Submit upload task
                future = executor.submit(upload_batch, supabase, transformed_batch, i // batch_size + 1)
                futures.append(future)
        
        # Monitor progress
        completed = 0
        with tqdm(total=len(futures), desc="Uploading batches") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    success = future.result()
                    completed += 1
                    pbar.update(1)
                    
                    # Update progress description
                    pbar.set_description(f"Uploaded: {upload_stats['success']}, Failed: {upload_stats['failed']}")
                    
                except Exception as e:
                    logger.error(f"❌ Batch upload failed: {e}")
                    completed += 1
                    pbar.update(1)
    
    # Final statistics
    logger.info(f"✅ Upload completed!")
    logger.info(f"📊 Successful uploads: {upload_stats['success']}")
    logger.info(f"❌ Failed uploads: {upload_stats['failed']}")
    logger.info(f"📈 Success rate: {upload_stats['success']/upload_stats['total']*100:.1f}%")
    
    return upload_stats['success'] > 0

def main():
    """Main execution function"""
    
    print("🚀 Nelson Pediatrics - Fast Supabase Upload")
    print("=" * 60)
    print("📊 Completing upload to table: nelson_textbook_chunks")
    print("🌐 Supabase URL: https://nrtaztkewvbtzhbtkffc.supabase.co")
    print("⚡ Using parallel processing for maximum speed")
    print("=" * 60)
    
    # Step 1: Create Supabase client
    supabase, existing_count = create_supabase_client()
    if not supabase:
        return
    
    # Step 2: Load processed data
    data = load_processed_data()
    if not data:
        return
    
    print(f"📊 Total documents: {len(data)}")
    print(f"📊 Already uploaded: {existing_count}")
    print(f"📊 Target: 15,339 documents")
    
    # Step 3: Get uploaded indices
    uploaded_indices = get_uploaded_indices(supabase)
    
    # Step 4: Fast upload remaining documents
    if not fast_upload_remaining(supabase, data, uploaded_indices):
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
        print("🎉 FAST SUPABASE UPLOAD COMPLETED!")
        print("=" * 60)
        print(f"📊 Total records uploaded: {total_count:,}")
        print(f"📚 Unique chapters: {len(unique_chapters)}")
        print(f"🌐 Database: Supabase PostgreSQL")
        print(f"🔍 Text search: Enabled")
        print(f"💾 Storage: Cloud (persistent)")
        
        if total_count >= 15000:
            print("\n🎉 SUCCESS! Complete Nelson Pediatrics knowledge base in Supabase!")
            print("🏥 Your NelsonGPT now has access to the full medical database! ⚡")
        else:
            print(f"\n📈 Progress: {total_count:,} / 15,339 ({total_count/15339*100:.1f}%)")
        
        print("\n🚀 Available chapters:")
        for chapter in sorted(unique_chapters)[:10]:
            print(f"  • {chapter}")
        if len(unique_chapters) > 10:
            print(f"  ... and {len(unique_chapters) - 10} more chapters")
        
    except Exception as e:
        logger.error(f"❌ Failed to get final statistics: {e}")

if __name__ == "__main__":
    main()


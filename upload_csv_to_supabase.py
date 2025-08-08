#!/usr/bin/env python3
"""
Upload CSV Dataset to Supabase

This script uploads the generated CSV dataset to Supabase in batches
with proper error handling and progress tracking.
"""

import csv
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import uuid
from datetime import datetime
import time

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing supabase dependency...")
    import os
    os.system("pip install supabase")
    from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

class SupabaseCSVUploader:
    """Upload CSV dataset to Supabase"""
    
    def __init__(self, csv_file: str = "nelson_textbook_chunks.csv", batch_size: int = 100):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.supabase = None
        self.upload_stats = {
            'total_records': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Test connection
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            existing_count = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Existing records in table: {existing_count:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def prepare_record(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Prepare a CSV row for Supabase upload"""
        try:
            # Parse metadata JSON
            metadata = {}
            if row['metadata']:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    metadata = {'raw_metadata': row['metadata']}
            
            # Convert page_number to integer
            page_number = None
            if row['page_number'] and row['page_number'].strip():
                try:
                    page_number = int(row['page_number'])
                except ValueError:
                    pass
            
            # Convert chunk_index to integer
            chunk_index = None
            if row['chunk_index'] and row['chunk_index'].strip():
                try:
                    chunk_index = int(row['chunk_index'])
                except ValueError:
                    pass
            
            # Prepare record
            record = {
                'id': row['id'],
                'content': row['content'],
                'chapter_title': row['chapter_title'],
                'section_title': row['section_title'] if row['section_title'] else None,
                'page_number': page_number,
                'chunk_index': chunk_index,
                'metadata': metadata,
                'created_at': row['created_at'],
                'embedding': None  # Will be added later by embedding scripts
            }
            
            return record
            
        except Exception as e:
            logger.warning(f"Error preparing record: {e}")
            return None
    
    def upload_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Upload a batch of records to Supabase"""
        try:
            result = self.supabase.table('nelson_textbook_chunks').insert(batch).execute()
            
            if result.data:
                self.upload_stats['successful_uploads'] += len(batch)
                return True
            else:
                logger.warning(f"Batch upload returned no data")
                self.upload_stats['failed_uploads'] += len(batch)
                return False
                
        except Exception as e:
            logger.error(f"❌ Batch upload failed: {e}")
            self.upload_stats['failed_uploads'] += len(batch)
            return False
    
    def upload_csv_dataset(self, skip_existing: bool = True) -> bool:
        """Upload the entire CSV dataset"""
        logger.info("🚀 Starting CSV dataset upload to Supabase")
        
        if not self.connect_to_supabase():
            return False
        
        self.upload_stats['start_time'] = datetime.now()
        
        try:
            # Get existing IDs if skip_existing is True
            existing_ids = set()
            if skip_existing:
                logger.info("🔍 Checking for existing records...")
                result = self.supabase.table('nelson_textbook_chunks').select('id').execute()
                if result.data:
                    existing_ids = {record['id'] for record in result.data}
                    logger.info(f"📊 Found {len(existing_ids):,} existing records to skip")
            
            # Read and process CSV
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                batch = []
                skipped_count = 0
                
                for row in reader:
                    self.upload_stats['total_records'] += 1
                    
                    # Skip existing records
                    if skip_existing and row['id'] in existing_ids:
                        skipped_count += 1
                        continue
                    
                    # Prepare record
                    record = self.prepare_record(row)
                    if record:
                        batch.append(record)
                    
                    # Upload batch when full
                    if len(batch) >= self.batch_size:
                        success = self.upload_batch(batch)
                        self.upload_stats['batches_processed'] += 1
                        
                        if success:
                            logger.info(f"✅ Uploaded batch {self.upload_stats['batches_processed']} ({len(batch)} records)")
                        else:
                            logger.error(f"❌ Failed to upload batch {self.upload_stats['batches_processed']}")
                        
                        batch = []
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                
                # Upload remaining records
                if batch:
                    success = self.upload_batch(batch)
                    self.upload_stats['batches_processed'] += 1
                    
                    if success:
                        logger.info(f"✅ Uploaded final batch ({len(batch)} records)")
                    else:
                        logger.error(f"❌ Failed to upload final batch")
                
                if skipped_count > 0:
                    logger.info(f"⏭️ Skipped {skipped_count:,} existing records")
            
            self.upload_stats['end_time'] = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during upload: {e}")
            self.upload_stats['end_time'] = datetime.now()
            return False
    
    def print_upload_summary(self):
        """Print upload summary statistics"""
        duration = None
        if self.upload_stats['start_time'] and self.upload_stats['end_time']:
            duration = self.upload_stats['end_time'] - self.upload_stats['start_time']
        
        print("\n📊 UPLOAD SUMMARY")
        print("=" * 50)
        print(f"📄 Total records processed: {self.upload_stats['total_records']:,}")
        print(f"✅ Successful uploads: {self.upload_stats['successful_uploads']:,}")
        print(f"❌ Failed uploads: {self.upload_stats['failed_uploads']:,}")
        print(f"📦 Batches processed: {self.upload_stats['batches_processed']:,}")
        
        if duration:
            print(f"⏱️ Total time: {duration}")
            if self.upload_stats['successful_uploads'] > 0:
                rate = self.upload_stats['successful_uploads'] / duration.total_seconds()
                print(f"🚀 Upload rate: {rate:.1f} records/second")
        
        success_rate = 0
        if self.upload_stats['total_records'] > 0:
            success_rate = (self.upload_stats['successful_uploads'] / self.upload_stats['total_records']) * 100
        
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if self.upload_stats['successful_uploads'] > 0:
            print("\n✅ UPLOAD SUCCESSFUL!")
            print("🚀 Next steps:")
            print("  1. Verify data in Supabase dashboard")
            print("  2. Run embedding generation scripts")
            print("  3. Test semantic search functionality")
        else:
            print("\n❌ UPLOAD FAILED!")
            print("🔍 Check the logs for error details")

def main():
    """Main execution function"""
    
    print("📤 NELSON PEDIATRICS CSV UPLOAD TO SUPABASE")
    print("=" * 60)
    print("🎯 Uploading CSV dataset to nelson_textbook_chunks table")
    print("📋 Batch processing with error handling")
    print("=" * 60)
    
    # Initialize uploader
    uploader = SupabaseCSVUploader(batch_size=50)  # Smaller batches for reliability
    
    # Upload dataset
    success = uploader.upload_csv_dataset(skip_existing=True)
    
    # Print summary
    uploader.print_upload_summary()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 CSV UPLOAD COMPLETE!")
        print("=" * 60)
        print("📊 Your Nelson Pediatrics dataset is now in Supabase")
        print("🤖 Ready for AI embedding generation")
    else:
        print("\n" + "=" * 60)
        print("❌ CSV UPLOAD FAILED!")
        print("=" * 60)
        print("🔍 Check the error logs above for details")

if __name__ == "__main__":
    main()


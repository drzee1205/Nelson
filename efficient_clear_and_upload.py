#!/usr/bin/env python3
"""
Efficient Clear and Upload - Fixed Version

This script uses a more efficient approach to clear existing data
by using SQL TRUNCATE instead of individual record deletion.
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

class EfficientDataUploader:
    """Efficiently clear existing data and upload fresh CSV dataset"""
    
    def __init__(self, csv_file: str = "nelson_textbook_chunks.csv", batch_size: int = 100):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.supabase = None
        self.operation_stats = {
            'existing_records_deleted': 0,
            'new_records_uploaded': 0,
            'failed_uploads': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase and verify connection"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Test connection by getting current record count
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            existing_count = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Current records in table: {existing_count:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def clear_table_efficiently(self) -> bool:
        """Clear all data using SQL TRUNCATE for efficiency"""
        logger.info("🗑️ Clearing table using efficient SQL method...")
        
        try:
            # Get current count before deletion
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            existing_count = result.count if result.count else 0
            
            if existing_count == 0:
                logger.info("✅ Table is already empty, no data to delete")
                return True
            
            logger.info(f"🗑️ Deleting {existing_count:,} existing records using SQL...")
            
            # Use SQL to truncate the table efficiently
            # This is much faster than individual record deletion
            sql_query = "TRUNCATE TABLE nelson_textbook_chunks RESTART IDENTITY CASCADE;"
            
            try:
                # Execute the truncate command
                result = self.supabase.rpc('exec_sql', {'sql': sql_query}).execute()
                logger.info("✅ SQL TRUNCATE executed successfully")
            except Exception as sql_error:
                logger.warning(f"SQL TRUNCATE failed: {sql_error}")
                logger.info("🔄 Falling back to batch deletion method...")
                return self.clear_table_batch_method()
            
            # Verify deletion
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            remaining_count = result.count if result.count else 0
            
            if remaining_count == 0:
                self.operation_stats['existing_records_deleted'] = existing_count
                logger.info(f"✅ Successfully deleted all {existing_count:,} existing records")
                return True
            else:
                logger.warning(f"⚠️ {remaining_count:,} records still remain after deletion")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error clearing table: {e}")
            logger.info("🔄 Falling back to batch deletion method...")
            return self.clear_table_batch_method()
    
    def clear_table_batch_method(self) -> bool:
        """Fallback method: Clear data in smaller batches"""
        logger.info("🗑️ Using batch deletion method...")
        
        try:
            deleted_total = 0
            batch_size = 100  # Much smaller batches to avoid URL length issues
            
            while True:
                # Get a small batch of IDs to delete
                result = self.supabase.table('nelson_textbook_chunks')\
                    .select('id')\
                    .limit(batch_size)\
                    .execute()
                
                if not result.data:
                    break
                
                # Delete records one by one to avoid URL length issues
                deleted_count = 0
                for record in result.data:
                    try:
                        delete_result = self.supabase.table('nelson_textbook_chunks')\
                            .delete()\
                            .eq('id', record['id'])\
                            .execute()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete record {record['id']}: {e}")
                
                deleted_total += deleted_count
                logger.info(f"🗑️ Deleted batch: {deleted_count} records (Total: {deleted_total:,})")
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
                
                # Break if we deleted fewer records than batch size (last batch)
                if deleted_count < batch_size:
                    break
            
            self.operation_stats['existing_records_deleted'] = deleted_total
            logger.info(f"✅ Batch deletion completed: {deleted_total:,} records deleted")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in batch deletion: {e}")
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
                self.operation_stats['new_records_uploaded'] += len(batch)
                return True
            else:
                logger.warning(f"Batch upload returned no data")
                self.operation_stats['failed_uploads'] += len(batch)
                return False
                
        except Exception as e:
            logger.error(f"❌ Batch upload failed: {e}")
            self.operation_stats['failed_uploads'] += len(batch)
            return False
    
    def upload_fresh_data(self) -> bool:
        """Upload the fresh CSV dataset"""
        logger.info("📤 Uploading fresh data from CSV...")
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                batch = []
                total_processed = 0
                
                for row in reader:
                    total_processed += 1
                    
                    # Prepare record
                    record = self.prepare_record(row)
                    if record:
                        batch.append(record)
                    
                    # Upload batch when full
                    if len(batch) >= self.batch_size:
                        success = self.upload_batch(batch)
                        self.operation_stats['batches_processed'] += 1
                        
                        if success:
                            logger.info(f"✅ Uploaded batch {self.operation_stats['batches_processed']} ({len(batch)} records)")
                        else:
                            logger.error(f"❌ Failed to upload batch {self.operation_stats['batches_processed']}")
                        
                        batch = []
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                
                # Upload remaining records
                if batch:
                    success = self.upload_batch(batch)
                    self.operation_stats['batches_processed'] += 1
                    
                    if success:
                        logger.info(f"✅ Uploaded final batch ({len(batch)} records)")
                    else:
                        logger.error(f"❌ Failed to upload final batch")
                
                logger.info(f"📊 Processed {total_processed:,} records from CSV")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error uploading fresh data: {e}")
            return False
    
    def verify_upload(self) -> bool:
        """Verify the upload was successful"""
        logger.info("🔍 Verifying upload success...")
        
        try:
            # Get final record count
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            final_count = result.count if result.count else 0
            
            logger.info(f"📊 Final record count: {final_count:,}")
            
            # Check if we have the expected number of records
            expected_count = self.operation_stats['new_records_uploaded']
            
            if final_count == expected_count:
                logger.info("✅ Upload verification successful!")
                return True
            else:
                logger.warning(f"⚠️ Record count mismatch: Expected {expected_count:,}, Found {final_count:,}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verifying upload: {e}")
            return False
    
    def execute_replacement(self) -> bool:
        """Execute the complete data replacement process"""
        logger.info("🚀 Starting efficient data replacement process")
        
        if not self.connect_to_supabase():
            return False
        
        self.operation_stats['start_time'] = datetime.now()
        
        # Step 1: Clear existing data efficiently
        if not self.clear_table_efficiently():
            logger.error("❌ Failed to clear existing data. Aborting upload.")
            return False
        
        # Step 2: Upload fresh data
        if not self.upload_fresh_data():
            logger.error("❌ Failed to upload fresh data.")
            return False
        
        # Step 3: Verify upload
        if not self.verify_upload():
            logger.warning("⚠️ Upload verification failed, but data may still be uploaded.")
        
        self.operation_stats['end_time'] = datetime.now()
        return True
    
    def print_operation_summary(self):
        """Print operation summary statistics"""
        duration = None
        if self.operation_stats['start_time'] and self.operation_stats['end_time']:
            duration = self.operation_stats['end_time'] - self.operation_stats['start_time']
        
        print("\n📊 EFFICIENT REPLACEMENT OPERATION SUMMARY")
        print("=" * 60)
        print(f"🗑️ Records deleted: {self.operation_stats['existing_records_deleted']:,}")
        print(f"📤 Records uploaded: {self.operation_stats['new_records_uploaded']:,}")
        print(f"❌ Failed uploads: {self.operation_stats['failed_uploads']:,}")
        print(f"📦 Batches processed: {self.operation_stats['batches_processed']:,}")
        
        if duration:
            print(f"⏱️ Total time: {duration}")
            if self.operation_stats['new_records_uploaded'] > 0:
                rate = self.operation_stats['new_records_uploaded'] / duration.total_seconds()
                print(f"🚀 Upload rate: {rate:.1f} records/second")
        
        success_rate = 100.0
        total_attempted = self.operation_stats['new_records_uploaded'] + self.operation_stats['failed_uploads']
        if total_attempted > 0:
            success_rate = (self.operation_stats['new_records_uploaded'] / total_attempted) * 100
        
        print(f"📈 Upload success rate: {success_rate:.1f}%")
        
        if self.operation_stats['new_records_uploaded'] > 0 and self.operation_stats['failed_uploads'] == 0:
            print("\n✅ DATA REPLACEMENT SUCCESSFUL!")
            print("🎉 Your Nelson Pediatrics dataset has been refreshed!")
            print("\n🚀 Next steps:")
            print("  1. Verify data in Supabase dashboard")
            print("  2. Run embedding generation scripts")
            print("  3. Test semantic search functionality")
        else:
            print("\n⚠️ REPLACEMENT COMPLETED WITH ISSUES!")
            print("🔍 Check the logs for error details")

def main():
    """Main execution function"""
    
    print("⚡ NELSON PEDIATRICS EFFICIENT DATA REPLACEMENT")
    print("=" * 60)
    print("🎯 Efficient deletion + fresh CSV upload")
    print("📋 Table: nelson_textbook_chunks")
    print("📄 Source: nelson_textbook_chunks.csv")
    print("⚡ Using optimized SQL methods")
    print("=" * 60)
    
    # Get user confirmation
    print("\n⚠️ This will DELETE ALL existing data and upload fresh CSV")
    response = input("❓ Type 'YES' to proceed: ").strip()
    
    if response != "YES":
        print("❌ Operation cancelled.")
        return
    
    # Initialize uploader
    uploader = EfficientDataUploader(batch_size=50)  # Smaller batches for reliability
    
    # Execute replacement
    success = uploader.execute_replacement()
    
    # Print summary
    uploader.print_operation_summary()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 EFFICIENT DATA REPLACEMENT COMPLETE!")
        print("=" * 60)
        print("📊 Your Nelson Pediatrics table has been refreshed")
        print("🤖 Ready for AI embedding generation")
    else:
        print("\n" + "=" * 60)
        print("❌ DATA REPLACEMENT FAILED!")
        print("=" * 60)
        print("🔍 Check the error logs above for details")

if __name__ == "__main__":
    main()


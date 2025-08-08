#!/usr/bin/env python3
"""
Create Table and Upload Fresh Data

This script creates the nelson_textbook_chunks table if it doesn't exist,
then uploads the fresh CSV data.
"""

import csv
import json
import logging
from typing import List, Dict, Any
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

class TableCreatorUploader:
    """Create table and upload fresh CSV data"""
    
    def __init__(self, csv_file: str = "nelson_textbook_chunks.csv", batch_size: int = 50):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.supabase = None
        self.upload_stats = {
            'total_processed': 0,
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
            logger.info(f"✅ Connected to Supabase")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def create_table_if_not_exists(self) -> bool:
        """Create the nelson_textbook_chunks table if it doesn't exist"""
        logger.info("🏗️ Creating nelson_textbook_chunks table...")
        
        # SQL to create the table with all required fields
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS public.nelson_textbook_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            chapter_title TEXT NOT NULL,
            section_title TEXT,
            page_number INTEGER,
            chunk_index INTEGER,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            embedding VECTOR(1536)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_nelson_chapter ON public.nelson_textbook_chunks(chapter_title);
        CREATE INDEX IF NOT EXISTS idx_nelson_page ON public.nelson_textbook_chunks(page_number);
        CREATE INDEX IF NOT EXISTS idx_nelson_created_at ON public.nelson_textbook_chunks(created_at);
        
        -- Create vector index for embeddings (if vector extension is available)
        CREATE INDEX IF NOT EXISTS nelson_embeddings_idx ON public.nelson_textbook_chunks 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """
        
        try:
            # Execute the table creation SQL
            # We'll use a direct SQL approach since the table doesn't exist
            import requests
            
            headers = {
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json'
            }
            
            # Use the PostgREST SQL endpoint
            sql_url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
            
            # Try to execute SQL via RPC (if available)
            try:
                response = requests.post(
                    sql_url,
                    headers=headers,
                    json={'sql': create_table_sql}
                )
                
                if response.status_code == 200:
                    logger.info("✅ Table created successfully via RPC")
                    return True
                else:
                    logger.warning(f"RPC failed: {response.status_code} - {response.text}")
            except Exception as rpc_error:
                logger.warning(f"RPC method failed: {rpc_error}")
            
            # Fallback: Try to create table by attempting an insert (will fail but give us info)
            logger.info("🔄 Attempting to verify table existence...")
            
            # Try a simple select to see if table exists
            try:
                result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
                logger.info("✅ Table already exists and is accessible")
                return True
            except Exception as select_error:
                logger.error(f"❌ Table does not exist and cannot be created automatically: {select_error}")
                
                print("\n🚨 MANUAL TABLE CREATION REQUIRED")
                print("=" * 60)
                print("The table needs to be created manually. Please:")
                print("1. Go to Supabase Dashboard → SQL Editor")
                print("2. Run this SQL command:")
                print("\n" + create_table_sql)
                print("\n3. Then run this script again")
                print("=" * 60)
                
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating table: {e}")
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
    
    def upload_csv_data(self) -> bool:
        """Upload the CSV dataset"""
        logger.info("📤 Uploading CSV data to Supabase...")
        
        self.upload_stats['start_time'] = datetime.now()
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                batch = []
                
                for row in reader:
                    self.upload_stats['total_processed'] += 1
                    
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
                        
                        # Progress update every 50 batches
                        if self.upload_stats['batches_processed'] % 50 == 0:
                            logger.info(f"📊 Progress: {self.upload_stats['successful_uploads']:,} records uploaded so far...")
                
                # Upload remaining records
                if batch:
                    success = self.upload_batch(batch)
                    self.upload_stats['batches_processed'] += 1
                    
                    if success:
                        logger.info(f"✅ Uploaded final batch ({len(batch)} records)")
                    else:
                        logger.error(f"❌ Failed to upload final batch")
                
                self.upload_stats['end_time'] = datetime.now()
                logger.info(f"📊 Processed {self.upload_stats['total_processed']:,} records from CSV")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error uploading CSV data: {e}")
            self.upload_stats['end_time'] = datetime.now()
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
            expected_count = self.upload_stats['successful_uploads']
            
            if final_count >= expected_count:
                logger.info("✅ Upload verification successful!")
                return True
            else:
                logger.warning(f"⚠️ Record count lower than expected: Expected {expected_count:,}, Found {final_count:,}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verifying upload: {e}")
            return False
    
    def print_upload_summary(self):
        """Print upload summary statistics"""
        duration = None
        if self.upload_stats['start_time'] and self.upload_stats['end_time']:
            duration = self.upload_stats['end_time'] - self.upload_stats['start_time']
        
        print("\n📊 UPLOAD SUMMARY")
        print("=" * 50)
        print(f"📄 Total records processed: {self.upload_stats['total_processed']:,}")
        print(f"✅ Successful uploads: {self.upload_stats['successful_uploads']:,}")
        print(f"❌ Failed uploads: {self.upload_stats['failed_uploads']:,}")
        print(f"📦 Batches processed: {self.upload_stats['batches_processed']:,}")
        
        if duration:
            print(f"⏱️ Total time: {duration}")
            if self.upload_stats['successful_uploads'] > 0:
                rate = self.upload_stats['successful_uploads'] / duration.total_seconds()
                print(f"🚀 Upload rate: {rate:.1f} records/second")
        
        success_rate = 0
        if self.upload_stats['total_processed'] > 0:
            success_rate = (self.upload_stats['successful_uploads'] / self.upload_stats['total_processed']) * 100
        
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
    
    print("🏗️ NELSON PEDIATRICS TABLE CREATION & UPLOAD")
    print("=" * 60)
    print("🎯 Create table and upload fresh CSV data")
    print("📋 Table: nelson_textbook_chunks")
    print("📄 Source: nelson_textbook_chunks.csv")
    print("=" * 60)
    
    # Initialize uploader
    uploader = TableCreatorUploader(batch_size=50)
    
    if not uploader.connect_to_supabase():
        print("❌ Cannot connect to Supabase. Check your connection.")
        return
    
    # Create table if needed
    if not uploader.create_table_if_not_exists():
        print("❌ Cannot create table. Manual intervention required.")
        return
    
    print("\n🚀 Starting upload process...")
    
    # Upload data
    success = uploader.upload_csv_data()
    
    # Verify upload
    if success:
        uploader.verify_upload()
    
    # Print summary
    uploader.print_upload_summary()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 TABLE CREATION & UPLOAD COMPLETE!")
        print("=" * 60)
        print("📊 Your Nelson Pediatrics data is now in Supabase")
        print("🤖 Ready for AI embedding generation")
    else:
        print("\n" + "=" * 60)
        print("❌ TABLE CREATION & UPLOAD FAILED!")
        print("=" * 60)
        print("🔍 Check the error logs above for details")

if __name__ == "__main__":
    main()


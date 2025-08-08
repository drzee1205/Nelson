#!/usr/bin/env python3
"""
Monitor Supabase Upload Progress

This script monitors the ongoing upload to Supabase and provides real-time updates.
"""

import time
import logging
from supabase import create_client
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

TARGET_COUNT = 15339

def monitor_upload():
    """Monitor the upload progress"""
    
    print("🔍 Nelson Pediatrics - Supabase Upload Monitor")
    print("=" * 60)
    
    try:
        # Connect to Supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        last_count = 0
        start_time = datetime.now()
        
        while True:
            try:
                # Get current count
                result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
                current_count = result.count if result.count else 0
                
                # Calculate progress
                progress_percent = (current_count / TARGET_COUNT) * 100
                
                # Calculate upload rate
                time_elapsed = (datetime.now() - start_time).total_seconds()
                if time_elapsed > 0:
                    upload_rate = current_count / time_elapsed
                    eta_seconds = (TARGET_COUNT - current_count) / upload_rate if upload_rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                else:
                    upload_rate = 0
                    eta_minutes = 0
                
                # Display progress
                progress_bar = "█" * int(progress_percent / 2) + "░" * (50 - int(progress_percent / 2))
                
                print(f"\r📊 Progress: [{progress_bar}] {progress_percent:.1f}% ({current_count:,}/{TARGET_COUNT:,})", end="")
                
                if current_count != last_count:
                    print(f"\n⚡ Upload rate: {upload_rate:.1f} docs/sec | ETA: {eta_minutes:.1f} minutes")
                    last_count = current_count
                
                # Check if complete
                if current_count >= TARGET_COUNT:
                    print(f"\n🎉 Upload completed! {current_count:,} documents uploaded")
                    
                    # Get final statistics
                    result = supabase.table('nelson_textbook_chunks').select('chapter_title').execute()
                    unique_chapters = set()
                    if result.data:
                        unique_chapters = set(record['chapter_title'] for record in result.data)
                    
                    print(f"📚 Total chapters: {len(unique_chapters)}")
                    print("📋 Available chapters:")
                    for chapter in sorted(unique_chapters):
                        print(f"  • {chapter}")
                    
                    break
                
                # Wait before next check
                time.sleep(10)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Monitoring stopped by user")
                print(f"📊 Current progress: {current_count:,}/{TARGET_COUNT:,} ({progress_percent:.1f}%)")
                break
                
            except Exception as e:
                logger.error(f"❌ Error checking progress: {e}")
                time.sleep(5)
                continue
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Supabase: {e}")

if __name__ == "__main__":
    monitor_upload()


#!/usr/bin/env python3
"""
Preview and Replace Data - Safe Version

This script first shows you exactly what data will be deleted,
then allows you to proceed with the replacement if you confirm.
"""

import csv
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

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

class DataPreviewReplacer:
    """Preview existing data and safely replace with new CSV data"""
    
    def __init__(self, csv_file: str = "nelson_textbook_chunks.csv"):
        self.csv_file = csv_file
        self.supabase = None
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Test connection
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            logger.info("✅ Connected to Supabase successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def preview_existing_data(self) -> Dict[str, Any]:
        """Preview what data currently exists in the table"""
        logger.info("🔍 Analyzing existing data in nelson_textbook_chunks table...")
        
        preview_data = {
            'total_records': 0,
            'chapters': {},
            'sample_records': [],
            'date_range': {'earliest': None, 'latest': None},
            'has_embeddings': 0,
            'has_sections': 0
        }
        
        try:
            # Get total count
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            preview_data['total_records'] = result.count if result.count else 0
            
            if preview_data['total_records'] == 0:
                logger.info("✅ Table is empty - no existing data to delete")
                return preview_data
            
            # Get sample records and statistics
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('id, chapter_title, section_title, page_number, created_at, embedding, content')\
                .limit(100)\
                .execute()
            
            if result.data:
                for record in result.data:
                    # Chapter statistics
                    chapter = record.get('chapter_title', 'Unknown')
                    if chapter not in preview_data['chapters']:
                        preview_data['chapters'][chapter] = 0
                    preview_data['chapters'][chapter] += 1
                    
                    # Date range
                    created_at = record.get('created_at')
                    if created_at:
                        if not preview_data['date_range']['earliest'] or created_at < preview_data['date_range']['earliest']:
                            preview_data['date_range']['earliest'] = created_at
                        if not preview_data['date_range']['latest'] or created_at > preview_data['date_range']['latest']:
                            preview_data['date_range']['latest'] = created_at
                    
                    # Feature counts
                    if record.get('embedding'):
                        preview_data['has_embeddings'] += 1
                    if record.get('section_title'):
                        preview_data['has_sections'] += 1
                    
                    # Sample records (first 5)
                    if len(preview_data['sample_records']) < 5:
                        sample = {
                            'id': record.get('id', '')[:36],
                            'chapter': record.get('chapter_title', ''),
                            'section': record.get('section_title', ''),
                            'page': record.get('page_number', ''),
                            'content_preview': record.get('content', '')[:100] + '...' if record.get('content') and len(record.get('content', '')) > 100 else record.get('content', ''),
                            'has_embedding': bool(record.get('embedding'))
                        }
                        preview_data['sample_records'].append(sample)
            
            return preview_data
            
        except Exception as e:
            logger.error(f"❌ Error previewing existing data: {e}")
            return preview_data
    
    def preview_new_data(self) -> Dict[str, Any]:
        """Preview the new CSV data that will be uploaded"""
        logger.info("📄 Analyzing new CSV data...")
        
        new_data = {
            'total_records': 0,
            'chapters': {},
            'sample_records': [],
            'has_sections': 0,
            'file_size_mb': 0
        }
        
        try:
            import os
            if os.path.exists(self.csv_file):
                new_data['file_size_mb'] = os.path.getsize(self.csv_file) / 1024 / 1024
            
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    new_data['total_records'] += 1
                    
                    # Chapter statistics
                    chapter = row.get('chapter_title', 'Unknown')
                    if chapter not in new_data['chapters']:
                        new_data['chapters'][chapter] = 0
                    new_data['chapters'][chapter] += 1
                    
                    # Feature counts
                    if row.get('section_title'):
                        new_data['has_sections'] += 1
                    
                    # Sample records (first 5)
                    if len(new_data['sample_records']) < 5:
                        sample = {
                            'id': row.get('id', '')[:36],
                            'chapter': row.get('chapter_title', ''),
                            'section': row.get('section_title', ''),
                            'page': row.get('page_number', ''),
                            'content_preview': row.get('content', '')[:100] + '...' if row.get('content') and len(row.get('content', '')) > 100 else row.get('content', ''),
                            'chunk_index': row.get('chunk_index', '')
                        }
                        new_data['sample_records'].append(sample)
                    
                    # Only analyze first 1000 records for performance
                    if i >= 999:
                        break
            
            return new_data
            
        except Exception as e:
            logger.error(f"❌ Error previewing new CSV data: {e}")
            return new_data
    
    def print_comparison_report(self):
        """Print detailed comparison between existing and new data"""
        print("📊 DATA REPLACEMENT PREVIEW REPORT")
        print("=" * 70)
        
        # Preview existing data
        existing_data = self.preview_existing_data()
        new_data = self.preview_new_data()
        
        print(f"\n🗑️ EXISTING DATA (TO BE DELETED):")
        print(f"  📊 Total Records: {existing_data['total_records']:,}")
        
        if existing_data['total_records'] > 0:
            print(f"  📚 Chapters: {len(existing_data['chapters'])}")
            print(f"  🤖 Records with Embeddings: {existing_data['has_embeddings']:,}")
            print(f"  📝 Records with Sections: {existing_data['has_sections']:,}")
            
            if existing_data['date_range']['earliest']:
                print(f"  📅 Date Range: {existing_data['date_range']['earliest'][:10]} to {existing_data['date_range']['latest'][:10]}")
            
            print(f"\n  📚 Existing Chapters:")
            for chapter, count in sorted(existing_data['chapters'].items()):
                print(f"    • {chapter}: {count:,} records")
            
            print(f"\n  📄 Sample Existing Records:")
            for i, sample in enumerate(existing_data['sample_records'], 1):
                print(f"    {i}. ID: {sample['id']}")
                print(f"       Chapter: {sample['chapter']}")
                print(f"       Section: {sample['section'] or 'None'}")
                print(f"       Has Embedding: {'Yes' if sample['has_embedding'] else 'No'}")
                print(f"       Content: {sample['content_preview']}")
        else:
            print("  ✅ Table is currently empty")
        
        print(f"\n📤 NEW DATA (TO BE UPLOADED):")
        print(f"  📊 Total Records: {new_data['total_records']:,}")
        print(f"  📁 File Size: {new_data['file_size_mb']:.1f} MB")
        print(f"  📚 Chapters: {len(new_data['chapters'])}")
        print(f"  📝 Records with Sections: {new_data['has_sections']:,}")
        
        print(f"\n  📚 New Chapters:")
        for chapter, count in sorted(new_data['chapters'].items()):
            print(f"    • {chapter}: {count:,} records")
        
        print(f"\n  📄 Sample New Records:")
        for i, sample in enumerate(new_data['sample_records'], 1):
            print(f"    {i}. ID: {sample['id']}")
            print(f"       Chapter: {sample['chapter']}")
            print(f"       Section: {sample['section'] or 'None'}")
            print(f"       Page: {sample['page']}")
            print(f"       Chunk Index: {sample['chunk_index']}")
            print(f"       Content: {sample['content_preview']}")
        
        print(f"\n📈 COMPARISON SUMMARY:")
        print(f"  🔄 Records Change: {existing_data['total_records']:,} → {new_data['total_records']:,}")
        
        if existing_data['total_records'] > 0:
            change = new_data['total_records'] - existing_data['total_records']
            change_pct = (change / existing_data['total_records']) * 100
            print(f"  📊 Net Change: {change:+,} records ({change_pct:+.1f}%)")
        
        print(f"  📚 Chapters Change: {len(existing_data['chapters'])} → {len(new_data['chapters'])}")
        
        # Show what will happen
        print(f"\n⚠️  WHAT WILL HAPPEN:")
        if existing_data['total_records'] > 0:
            print(f"  🗑️ DELETE: All {existing_data['total_records']:,} existing records")
            if existing_data['has_embeddings'] > 0:
                print(f"  ⚠️  WARNING: {existing_data['has_embeddings']:,} embeddings will be lost!")
        print(f"  📤 UPLOAD: {new_data['total_records']:,} new records from CSV")
        print(f"  🤖 EMBEDDINGS: Will be NULL (need to regenerate)")
        
        return existing_data['total_records'], new_data['total_records']

def main():
    """Main execution function"""
    
    print("🔍 NELSON PEDIATRICS DATA REPLACEMENT PREVIEW")
    print("=" * 70)
    print("📋 This tool shows you exactly what will be replaced")
    print("🔒 Safe preview before making any changes")
    print("=" * 70)
    
    previewer = DataPreviewReplacer()
    
    if not previewer.connect_to_supabase():
        print("❌ Cannot connect to Supabase. Check your connection.")
        return
    
    # Show comparison report
    existing_count, new_count = previewer.print_comparison_report()
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    
    if existing_count == 0:
        print("✅ Table is empty - you can safely upload new data")
        print("🚀 Run: python upload_csv_to_supabase.py")
    else:
        print("⚠️  To proceed with data replacement:")
        print("🗑️ Run: python clear_and_upload_fresh.py")
        print("   (This will ask for confirmation before deleting)")
        
        if existing_count > 0:
            print(f"\n💾 BACKUP RECOMMENDATION:")
            print(f"   Consider backing up your {existing_count:,} existing records")
            print(f"   especially the {previewer.preview_existing_data()['has_embeddings']:,} embeddings!")
    
    print("\n🔍 This preview is complete - no data was modified.")

if __name__ == "__main__":
    main()


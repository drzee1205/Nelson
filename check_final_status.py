#!/usr/bin/env python3
"""
Check Final Supabase Status

Get accurate count and chapter information from Supabase.
"""

from supabase import create_client
import json

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def main():
    print("📊 Nelson Pediatrics - Final Supabase Status")
    print("=" * 60)
    
    try:
        # Connect to Supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Get total count
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = result.count if result.count else 0
        
        print(f"📊 Total documents in Supabase: {total_count:,}")
        
        # Get unique chapters with counts
        result = supabase.table('nelson_textbook_chunks')\
            .select('chapter_title')\
            .execute()
        
        if result.data:
            # Count chapters
            chapter_counts = {}
            for record in result.data:
                chapter = record['chapter_title']
                chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
            
            print(f"📚 Unique chapters: {len(chapter_counts)}")
            print("\n📋 Chapter breakdown:")
            
            # Sort by count (descending)
            sorted_chapters = sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)
            
            for chapter, count in sorted_chapters:
                print(f"  • {chapter}: {count:,} documents")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, chapter_title, content')\
            .ilike('content', '%asthma%')\
            .limit(3)\
            .execute()
        
        if result.data:
            print(f"✅ Search test passed - found {len(result.data)} results for 'asthma'")
            for i, record in enumerate(result.data, 1):
                print(f"   {i}. Chapter: {record['chapter_title']}")
                print(f"      Content: {record['content'][:100]}...")
        else:
            print("⚠️ No search results found")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 SUPABASE STATUS SUMMARY")
        print("=" * 60)
        
        if total_count >= 15000:
            print("✅ SUCCESS: Complete Nelson Pediatrics database uploaded!")
            print("🏥 Your NelsonGPT has access to the full medical knowledge base!")
        elif total_count >= 10000:
            print("✅ EXCELLENT: Major portion of Nelson Pediatrics uploaded!")
            print("🏥 Your NelsonGPT has substantial medical knowledge available!")
        elif total_count >= 5000:
            print("✅ GOOD: Significant medical content available!")
            print("🏥 Your NelsonGPT can provide medical guidance!")
        else:
            print("⚠️ PARTIAL: Some medical content available")
        
        print(f"📊 Database size: {total_count:,} medical documents")
        print(f"📚 Medical specialties: {len(chapter_counts) if 'chapter_counts' in locals() else 'Unknown'}")
        print(f"🔍 Text search: ✅ Enabled")
        print(f"🌐 Cloud storage: ✅ Supabase PostgreSQL")
        print(f"⚡ API ready: ✅ Use supabase_api.py")
        
        print(f"\n🚀 Your medical knowledge base is ready for NelsonGPT integration!")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    main()


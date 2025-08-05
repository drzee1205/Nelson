#!/usr/bin/env python3
"""
Simple Embedding Test - Quick Solution

Test if we can store embeddings in a working format.
"""

import json
from supabase import create_client
from sentence_transformers import SentenceTransformer
import numpy as np

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def main():
    print("🧪 SIMPLE EMBEDDING TEST")
    print("=" * 40)
    
    # Connect to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    # Clear all existing embeddings first
    print("🧹 Clearing existing embeddings...")
    result = supabase.table('nelson_textbook_chunks')\
        .select('id')\
        .not_.is_('embedding', 'null')\
        .execute()
    
    if result.data:
        for record in result.data:
            supabase.table('nelson_textbook_chunks')\
                .update({'embedding': None})\
                .eq('id', record['id'])\
                .execute()
        print(f"✅ Cleared {len(result.data)} embeddings")
    
    # Load model
    print("🤖 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
    
    # Get 10 records to test
    print("📄 Getting test records...")
    result = supabase.table('nelson_textbook_chunks')\
        .select('id, content')\
        .limit(10)\
        .execute()
    
    if not result.data:
        print("❌ No records found")
        return
    
    print(f"✅ Found {len(result.data)} records")
    
    # Process each record
    successful = 0
    
    for i, record in enumerate(result.data, 1):
        try:
            content = record['content']
            if not content or len(content.strip()) < 10:
                continue
            
            print(f"Processing record {i}: {content[:50]}...")
            
            # Generate embedding
            embedding = model.encode(content, convert_to_tensor=False)
            
            # Convert to list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            print(f"Generated {len(embedding)}D embedding")
            
            # Store in metadata field as JSON string
            metadata = {
                'embedding': embedding,
                'embedding_dim': len(embedding),
                'model': 'all-MiniLM-L6-v2'
            }
            
            # Update record
            update_result = supabase.table('nelson_textbook_chunks')\
                .update({'metadata': metadata})\
                .eq('id', record['id'])\
                .execute()
            
            if update_result.data:
                print(f"✅ Successfully stored embedding for record {i}")
                successful += 1
            else:
                print(f"❌ Failed to store embedding for record {i}")
                
        except Exception as e:
            print(f"❌ Error processing record {i}: {e}")
    
    print(f"\n📊 Results: {successful}/{len(result.data)} records processed successfully")
    
    # Verify the stored embeddings
    print("\n🔍 Verifying stored embeddings...")
    verify_result = supabase.table('nelson_textbook_chunks')\
        .select('id, metadata, content')\
        .not_.is_('metadata', 'null')\
        .limit(5)\
        .execute()
    
    if verify_result.data:
        for i, record in enumerate(verify_result.data, 1):
            metadata = record.get('metadata', {})
            if 'embedding' in metadata:
                embedding = metadata['embedding']
                embedding_dim = len(embedding) if embedding else 0
                content_preview = record['content'][:40] + "..." if record['content'] else "No content"
                print(f"✅ Record {i}: {embedding_dim}D embedding - {content_preview}")
            else:
                print(f"❌ Record {i}: No embedding found")
    
    if successful > 0:
        print("\n🎉 SUCCESS! Embeddings are working!")
        print("💾 Embeddings stored in metadata field")
        print("🔍 Ready for semantic search implementation")
    else:
        print("\n❌ No embeddings were successfully stored")

if __name__ == "__main__":
    main()


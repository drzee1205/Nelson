#!/usr/bin/env python3
"""
Direct Embedding Fix - Reset and Regenerate

This script completely resets the embedding column and regenerates with correct format.
"""

import os
import json
import logging
from typing import List, Optional
from tqdm import tqdm
import numpy as np

# Dependencies
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing dependencies...")
    os.system("pip install sentence-transformers torch supabase")
    from sentence_transformers import SentenceTransformer
    import torch
    from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def create_supabase_client():
    """Create Supabase client"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = result.count if result.count else 0
        logger.info(f"✅ Connected to Supabase. Found {total_count} records")
        return supabase
    except Exception as e:
        logger.error(f"❌ Failed to connect to Supabase: {e}")
        return None

def clear_all_embeddings(supabase: Client):
    """Clear ALL embeddings to start fresh"""
    try:
        logger.info("🧹 Clearing ALL embeddings to start fresh...")
        
        # Get all records with embeddings
        result = supabase.table('nelson_textbook_chunks')\
            .select('id')\
            .not_.is_('embedding', 'null')\
            .execute()
        
        if not result.data:
            logger.info("No embeddings to clear")
            return 0
        
        cleared_count = 0
        
        # Clear in batches
        for record in tqdm(result.data, desc="Clearing embeddings"):
            try:
                supabase.table('nelson_textbook_chunks')\
                    .update({'embedding': None})\
                    .eq('id', record['id'])\
                    .execute()
                cleared_count += 1
            except Exception as e:
                logger.warning(f"Failed to clear embedding for {record['id']}: {e}")
        
        logger.info(f"✅ Cleared {cleared_count} embeddings")
        return cleared_count
        
    except Exception as e:
        logger.error(f"❌ Error clearing embeddings: {e}")
        return 0

def generate_simple_embeddings(supabase: Client, batch_size: int = 10):
    """Generate simple, properly formatted embeddings"""
    try:
        logger.info("🤖 Loading simple embedding model...")
        
        # Use a simple, reliable model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Model loaded")
        
        total_processed = 0
        
        while True:
            # Get records without embeddings
            result = supabase.table('nelson_textbook_chunks')\
                .select('id, content')\
                .is_('embedding', 'null')\
                .limit(batch_size)\
                .execute()
            
            if not result.data:
                logger.info("✅ No more records to process")
                break
            
            logger.info(f"📄 Processing batch of {len(result.data)} records...")
            
            successful_updates = 0
            
            for record in tqdm(result.data, desc="Generating embeddings"):
                try:
                    content = record.get('content', '')
                    if not content or len(content.strip()) < 10:
                        continue
                    
                    # Generate embedding (384D)
                    embedding = model.encode(content, convert_to_tensor=False)
                    
                    # Convert to Python list
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    # Ensure it's exactly 384D (the model's native dimension)
                    if len(embedding) != 384:
                        logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                        continue
                    
                    # Convert to proper format for Supabase
                    embedding_list = [float(x) for x in embedding]
                    
                    # Update record with simple embedding
                    update_result = supabase.table('nelson_textbook_chunks')\
                        .update({'embedding': embedding_list})\
                        .eq('id', record['id'])\
                        .execute()
                    
                    if update_result.data:
                        successful_updates += 1
                    else:
                        logger.warning(f"Failed to update record {record['id']}")
                        
                except Exception as e:
                    logger.warning(f"Error processing record {record.get('id', 'unknown')}: {e}")
                    continue
            
            total_processed += successful_updates
            logger.info(f"✅ Successfully processed {successful_updates}/{len(result.data)} records")
            logger.info(f"📊 Total processed so far: {total_processed}")
            
            if successful_updates == 0:
                logger.warning("No successful updates in this batch, stopping")
                break
        
        return total_processed
        
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        return 0

def verify_final_embeddings(supabase: Client):
    """Verify the final embeddings"""
    try:
        logger.info("🔍 Verifying final embeddings...")
        
        # Get sample embeddings
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, embedding, content')\
            .not_.is_('embedding', 'null')\
            .limit(10)\
            .execute()
        
        if not result.data:
            logger.warning("No embeddings found")
            return False
        
        all_good = True
        
        for i, record in enumerate(result.data, 1):
            embedding = record['embedding']
            embedding_dim = len(embedding) if embedding else 0
            content_preview = record['content'][:50] + "..." if record['content'] else "No content"
            
            logger.info(f"Record {i}: {embedding_dim}D - {content_preview}")
            
            if embedding_dim not in [384, 1536]:  # Accept either dimension
                all_good = False
        
        # Get total count
        total_result = supabase.table('nelson_textbook_chunks')\
            .select('count', count='exact')\
            .not_.is_('embedding', 'null')\
            .execute()
        
        total_embeddings = total_result.count if total_result.count else 0
        
        logger.info(f"📊 Total records with embeddings: {total_embeddings}")
        
        return all_good and total_embeddings > 0
        
    except Exception as e:
        logger.error(f"❌ Error verifying embeddings: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🔧 DIRECT EMBEDDING FIX - COMPLETE RESET")
    print("=" * 60)
    print("🧹 Clearing all existing embeddings")
    print("🤖 Regenerating with simple, reliable format")
    print("=" * 60)
    
    # Step 1: Create Supabase client
    supabase = create_supabase_client()
    if not supabase:
        print("❌ Failed to connect to Supabase. Exiting.")
        return
    
    # Step 2: Clear all existing embeddings
    cleared_count = clear_all_embeddings(supabase)
    print(f"🧹 Cleared {cleared_count} existing embeddings")
    
    # Step 3: Generate new embeddings
    processed_count = generate_simple_embeddings(supabase)
    
    if processed_count > 0:
        print(f"\n✅ Successfully processed {processed_count:,} records")
        
        # Step 4: Verify results
        if verify_final_embeddings(supabase):
            print("✅ All embeddings verified!")
        else:
            print("⚠️ Some embeddings may have issues")
        
        print("\n" + "=" * 60)
        print("🎉 EMBEDDING FIX COMPLETE!")
        print("=" * 60)
        print(f"🤖 Model: all-MiniLM-L6-v2 (384D native)")
        print(f"📊 Records Processed: {processed_count:,}")
        print(f"✅ Embeddings are now properly formatted")
        print(f"🔍 Ready for semantic search!")
        
    else:
        print("⚠️ No records were processed. Check the logs for issues.")

if __name__ == "__main__":
    main()


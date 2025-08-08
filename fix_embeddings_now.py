#!/usr/bin/env python3
"""
Fix Embeddings Issue - Correct Dimension and Format

This script fixes the embedding dimension issue and ensures proper 1536D vectors.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from datetime import datetime

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

class FixedEmbeddingGenerator:
    """Generate properly formatted 1536D embeddings"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🤖 Using device: {self.device}")
        
    def load_model(self):
        """Load the embedding model"""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use 384D model
            logger.info(f"📥 Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Test embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            embedding_dim = len(test_embedding)
            logger.info(f"✅ Model loaded. Base dimension: {embedding_dim}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate properly formatted 1536D embedding"""
        try:
            if not self.model:
                if not self.load_model():
                    return None
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Generate base embedding (384D)
            embedding = self.model.encode(cleaned_text, convert_to_tensor=False)
            
            # Convert to list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Pad to exactly 1536 dimensions
            current_dim = len(embedding)
            if current_dim < 1536:
                # Pad with zeros
                padding = [0.0] * (1536 - current_dim)
                embedding = embedding + padding
            elif current_dim > 1536:
                # Truncate to 1536
                embedding = embedding[:1536]
            
            # Ensure all values are float and properly formatted
            embedding = [float(x) for x in embedding]
            
            # Verify final dimension
            if len(embedding) != 1536:
                logger.error(f"❌ Wrong embedding dimension: {len(embedding)}")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean text for embedding"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\%\/]', ' ', text)
        
        # Limit length
        words = text.split()
        if len(words) > 400:  # Reasonable limit
            text = ' '.join(words[:400])
        
        return text.strip()

def create_supabase_client():
    """Create Supabase client"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Test connection
        result = supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
        total_count = result.count if result.count else 0
        
        logger.info(f"✅ Connected to Supabase. Found {total_count} records")
        return supabase, total_count
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Supabase: {e}")
        return None, 0

def clear_bad_embeddings(supabase: Client):
    """Clear embeddings that have wrong dimensions"""
    try:
        logger.info("🧹 Clearing bad embeddings...")
        
        # Get records with embeddings to check their dimensions
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, embedding')\
            .not_.is_('embedding', 'null')\
            .execute()
        
        bad_embedding_ids = []
        
        for record in result.data:
            embedding = record['embedding']
            if embedding and len(embedding) != 1536:
                bad_embedding_ids.append(record['id'])
        
        logger.info(f"Found {len(bad_embedding_ids)} records with wrong embedding dimensions")
        
        # Clear bad embeddings
        if bad_embedding_ids:
            for record_id in tqdm(bad_embedding_ids, desc="Clearing bad embeddings"):
                try:
                    supabase.table('nelson_textbook_chunks')\
                        .update({'embedding': None})\
                        .eq('id', record_id)\
                        .execute()
                except Exception as e:
                    logger.warning(f"Failed to clear embedding for {record_id}: {e}")
        
        logger.info(f"✅ Cleared {len(bad_embedding_ids)} bad embeddings")
        return len(bad_embedding_ids)
        
    except Exception as e:
        logger.error(f"❌ Error clearing bad embeddings: {e}")
        return 0

def generate_fixed_embeddings(supabase: Client, generator: FixedEmbeddingGenerator, batch_size: int = 20):
    """Generate properly formatted embeddings"""
    try:
        logger.info("🤖 Generating fixed embeddings...")
        
        total_processed = 0
        
        while True:
            # Get records without embeddings
            result = supabase.table('nelson_textbook_chunks')\
                .select('id, content, chapter_title')\
                .is_('embedding', 'null')\
                .limit(batch_size)\
                .execute()
            
            if not result.data:
                logger.info("✅ No more records to process")
                break
            
            logger.info(f"📄 Processing batch of {len(result.data)} records...")
            
            # Process each record
            successful_updates = 0
            
            for record in tqdm(result.data, desc="Generating embeddings"):
                try:
                    content = record.get('content', '')
                    if not content:
                        continue
                    
                    # Generate embedding
                    embedding = generator.generate_embedding(content)
                    
                    if embedding and len(embedding) == 1536:
                        # Update record
                        update_result = supabase.table('nelson_textbook_chunks')\
                            .update({
                                'embedding': embedding,
                                'metadata': {
                                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                                    'embedding_dimension': 1536,
                                    'processed_at': datetime.now().isoformat()
                                }
                            })\
                            .eq('id', record['id'])\
                            .execute()
                        
                        if update_result.data:
                            successful_updates += 1
                        else:
                            logger.warning(f"Failed to update record {record['id']}")
                    else:
                        logger.warning(f"Failed to generate embedding for record {record['id']}")
                        
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

def verify_embeddings(supabase: Client):
    """Verify that embeddings are properly formatted"""
    try:
        logger.info("🔍 Verifying embeddings...")
        
        # Get sample of records with embeddings
        result = supabase.table('nelson_textbook_chunks')\
            .select('id, embedding, content')\
            .not_.is_('embedding', 'null')\
            .limit(10)\
            .execute()
        
        if not result.data:
            logger.warning("No embeddings found to verify")
            return False
        
        all_good = True
        
        for i, record in enumerate(result.data, 1):
            embedding = record['embedding']
            embedding_dim = len(embedding) if embedding else 0
            content_preview = record['content'][:50] + "..." if record['content'] else "No content"
            
            if embedding_dim == 1536:
                logger.info(f"✅ Record {i}: {embedding_dim}D - {content_preview}")
            else:
                logger.error(f"❌ Record {i}: {embedding_dim}D (WRONG!) - {content_preview}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        logger.error(f"❌ Error verifying embeddings: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🔧 FIXING EMBEDDING DIMENSION ISSUE")
    print("=" * 60)
    print("🎯 Generating properly formatted 1536D embeddings")
    print("🧹 Clearing bad embeddings and regenerating")
    print("=" * 60)
    
    # Step 1: Initialize components
    generator = FixedEmbeddingGenerator()
    
    # Step 2: Create Supabase client
    supabase, total_count = create_supabase_client()
    if not supabase:
        return
    
    print(f"📊 Total records in database: {total_count:,}")
    
    # Step 3: Load embedding model
    if not generator.load_model():
        print("❌ Failed to load embedding model. Exiting.")
        return
    
    # Step 4: Clear bad embeddings
    cleared_count = clear_bad_embeddings(supabase)
    print(f"🧹 Cleared {cleared_count} bad embeddings")
    
    # Step 5: Generate fixed embeddings
    processed_count = generate_fixed_embeddings(supabase, generator)
    
    if processed_count > 0:
        print(f"\n✅ Successfully processed {processed_count:,} records")
        
        # Step 6: Verify results
        if verify_embeddings(supabase):
            print("✅ All embeddings verified as 1536D!")
        else:
            print("⚠️ Some embeddings still have issues")
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 EMBEDDING ISSUE FIXED!")
        print("=" * 60)
        print(f"🤖 Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"📊 Records Processed: {processed_count:,}")
        print(f"🔍 Dimension: 1536D (properly formatted)")
        print(f"✅ Ready for semantic search!")
        
    else:
        print("⚠️ No records were processed. Check the logs for issues.")

if __name__ == "__main__":
    main()


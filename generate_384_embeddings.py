#!/usr/bin/env python3
"""
Generate 384-Dimension Embeddings

This script generates embeddings using the all-MiniLM-L6-v2 model (384 dimensions)
to match the updated table schema.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import numpy as np

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing supabase dependency...")
    import os
    os.system("pip install supabase")
    from supabase import create_client, Client

# Hugging Face dependencies
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("❌ Installing Hugging Face dependencies...")
    import os
    os.system("pip install sentence-transformers torch")
    from sentence_transformers import SentenceTransformer
    import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

class Fast384EmbeddingGenerator:
    """Generate 384-dimension embeddings using all-MiniLM-L6-v2"""
    
    def __init__(self, batch_size: int = 32):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.batch_size = batch_size
        self.model = None
        self.supabase = None
        
        # Statistics tracking
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'failed_records': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None,
            'model_load_time': None
        }
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase and verify table schema"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Get total record count
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            self.stats['total_records'] = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Total records: {self.stats['total_records']:,}")
            
            # Test embedding dimensions
            test_embedding = [0.1] * 384
            result = self.supabase.table('nelson_textbook_chunks').select('id').limit(1).execute()
            
            if result.data:
                test_id = result.data[0]['id']
                try:
                    self.supabase.table('nelson_textbook_chunks')\
                        .update({'embedding': test_embedding})\
                        .eq('id', test_id)\
                        .execute()
                    
                    # Clean up test
                    self.supabase.table('nelson_textbook_chunks')\
                        .update({'embedding': None})\
                        .eq('id', test_id)\
                        .execute()
                    
                    logger.info("✅ Table accepts 384-dimension embeddings")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Table dimension mismatch: {e}")
                    return False
            else:
                logger.warning("⚠️ No records found to test with")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the all-MiniLM-L6-v2 model"""
        logger.info(f"🤖 Loading fast embedding model: {self.model_name}")
        
        start_time = time.time()
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🖥️ Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"📐 Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
            
            # Verify dimensions
            if self.model.get_sentence_embedding_dimension() != 384:
                logger.error(f"❌ Model produces {self.model.get_sentence_embedding_dimension()} dimensions, expected 384")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_records_batch(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get a batch of records without embeddings"""
        try:
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('id, content, chapter_title, section_title')\
                .is_('embedding', 'null')\
                .range(offset, offset + limit - 1)\
                .order('chapter_title, chunk_index')\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"❌ Error fetching records: {e}")
            return []
    
    def preprocess_text(self, content: str, chapter: str = "", section: str = "") -> str:
        """Preprocess text for embedding generation"""
        try:
            # Add medical context
            context_parts = []
            if chapter:
                context_parts.append(f"Medical Chapter: {chapter}")
            if section:
                context_parts.append(f"Section: {section}")
            
            if context_parts:
                processed_text = " | ".join(context_parts) + " | " + content
            else:
                processed_text = content
            
            # Clean up text
            processed_text = processed_text.strip()
            
            # Truncate if too long (rough token estimate)
            if len(processed_text) > 2000:  # Conservative limit for fast model
                processed_text = processed_text[:2000]
            
            return processed_text
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return content
    
    def generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=min(self.batch_size, len(texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            return None
    
    def update_embeddings_batch(self, records: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Update records with their embeddings"""
        try:
            updates = []
            
            for i, record in enumerate(records):
                embedding_list = embeddings[i].tolist()
                updates.append({
                    'id': record['id'],
                    'embedding': embedding_list
                })
            
            result = self.supabase.table('nelson_textbook_chunks').upsert(updates).execute()
            
            if result.data:
                self.stats['processed_records'] += len(updates)
                return True
            else:
                self.stats['failed_records'] += len(updates)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating embeddings: {e}")
            self.stats['failed_records'] += len(records)
            return False
    
    def process_all_embeddings(self) -> bool:
        """Process all records and generate embeddings"""
        logger.info("🚀 Starting fast embedding generation (384D)")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            offset = 0
            batch_limit = 100
            
            while True:
                # Get batch of records
                records = self.get_records_batch(limit=batch_limit, offset=offset)
                
                if not records:
                    logger.info("✅ No more records to process")
                    break
                
                # Preprocess texts
                texts = []
                for record in records:
                    processed_text = self.preprocess_text(
                        content=record['content'],
                        chapter=record.get('chapter_title', ''),
                        section=record.get('section_title', '')
                    )
                    texts.append(processed_text)
                
                # Generate embeddings
                embeddings = self.generate_embeddings_batch(texts)
                
                if embeddings is not None:
                    # Update records
                    success = self.update_embeddings_batch(records, embeddings)
                    
                    if success:
                        self.stats['batches_processed'] += 1
                        logger.info(f"✅ Batch {self.stats['batches_processed']} complete "
                                  f"({len(records)} records, {self.stats['processed_records']:,} total)")
                        
                        # Progress update every 25 batches
                        if self.stats['batches_processed'] % 25 == 0:
                            progress = (self.stats['processed_records'] / self.stats['total_records']) * 100
                            logger.info(f"📊 Progress: {progress:.1f}% complete "
                                      f"({self.stats['processed_records']:,} / {self.stats['total_records']:,})")
                    else:
                        logger.error(f"❌ Failed to update batch {self.stats['batches_processed'] + 1}")
                else:
                    logger.error(f"❌ Failed to generate embeddings for batch")
                    self.stats['failed_records'] += len(records)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Rate limiting
                time.sleep(0.05)  # Faster processing for smaller model
                
                # Move to next batch
                offset += batch_limit
                
                # Safety check
                if offset > self.stats['total_records'] * 2:
                    logger.warning("⚠️ Safety limit reached, stopping")
                    break
            
            self.stats['end_time'] = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in embedding process: {e}")
            self.stats['end_time'] = datetime.now()
            return False
    
    def verify_embeddings(self) -> Dict[str, Any]:
        """Verify embedding generation results"""
        logger.info("🔍 Verifying embeddings...")
        
        try:
            # Count records with embeddings
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('count', count='exact')\
                .not_.is_('embedding', 'null')\
                .execute()
            
            records_with_embeddings = result.count if result.count else 0
            
            # Get sample embeddings
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('embedding, chapter_title, content')\
                .not_.is_('embedding', 'null')\
                .limit(3)\
                .execute()
            
            sample_embeddings = []
            if result.data:
                for record in result.data:
                    if record['embedding']:
                        sample_embeddings.append({
                            'chapter': record['chapter_title'],
                            'content_preview': record['content'][:100] + '...',
                            'embedding_dim': len(record['embedding']),
                            'embedding_norm': np.linalg.norm(record['embedding'])
                        })
            
            verification = {
                'total_records': self.stats['total_records'],
                'records_with_embeddings': records_with_embeddings,
                'records_without_embeddings': self.stats['total_records'] - records_with_embeddings,
                'completion_rate': (records_with_embeddings / self.stats['total_records'] * 100) if self.stats['total_records'] > 0 else 0,
                'sample_embeddings': sample_embeddings
            }
            
            return verification
            
        except Exception as e:
            logger.error(f"❌ Error verifying embeddings: {e}")
            return {}
    
    def print_summary(self):
        """Print comprehensive summary"""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n📊 FAST EMBEDDING GENERATION SUMMARY")
        print("=" * 60)
        print(f"🤖 Model: {self.model_name}")
        print(f"📐 Dimensions: 384")
        print(f"📊 Total records: {self.stats['total_records']:,}")
        print(f"✅ Successfully processed: {self.stats['processed_records']:,}")
        print(f"❌ Failed: {self.stats['failed_records']:,}")
        print(f"📦 Batches processed: {self.stats['batches_processed']:,}")
        
        if self.stats['model_load_time']:
            print(f"🤖 Model load time: {self.stats['model_load_time']:.2f} seconds")
        
        if duration:
            print(f"⏱️ Total processing time: {duration}")
            if self.stats['processed_records'] > 0:
                rate = self.stats['processed_records'] / duration.total_seconds()
                print(f"🚀 Processing rate: {rate:.1f} records/second")
        
        # Verification results
        verification = self.verify_embeddings()
        if verification:
            print(f"\n🔍 VERIFICATION RESULTS:")
            print(f"  ✅ Records with embeddings: {verification['records_with_embeddings']:,}")
            print(f"  ❌ Records without embeddings: {verification['records_without_embeddings']:,}")
            print(f"  📈 Completion rate: {verification['completion_rate']:.1f}%")
            
            if verification['sample_embeddings']:
                print(f"\n📋 SAMPLE EMBEDDINGS:")
                for i, sample in enumerate(verification['sample_embeddings'], 1):
                    print(f"  {i}. Chapter: {sample['chapter']}")
                    print(f"     Dimensions: {sample['embedding_dim']}")
                    print(f"     Norm: {sample['embedding_norm']:.3f}")
            
            if verification['completion_rate'] >= 99:
                print(f"\n🎉 FAST EMBEDDING GENERATION SUCCESSFUL!")
                print(f"⚡ Nelson Pediatrics database ready for fast semantic search!")
            else:
                print(f"\n⚠️ PARTIAL COMPLETION")
                print(f"🔄 {verification['records_without_embeddings']:,} records need reprocessing")

def main():
    """Main execution function"""
    
    print("⚡ NELSON PEDIATRICS FAST EMBEDDING GENERATION")
    print("=" * 60)
    print("🤖 Using all-MiniLM-L6-v2 model (384 dimensions)")
    print("🚀 Optimized for speed and efficiency")
    print("=" * 60)
    
    # Initialize generator
    generator = Fast384EmbeddingGenerator(batch_size=32)
    
    print(f"\n🤖 CONFIGURATION:")
    print(f"  📋 Model: {generator.model_name}")
    print(f"  📐 Dimensions: 384")
    print(f"  📦 Batch size: {generator.batch_size}")
    print(f"  ⚡ Optimized for: Speed and efficiency")
    
    # Connect to Supabase
    if not generator.connect_to_supabase():
        print("❌ Cannot connect to Supabase or dimension mismatch.")
        return
    
    # Load model
    if not generator.load_model():
        print("❌ Cannot load embedding model.")
        return
    
    # Process embeddings
    success = generator.process_all_embeddings()
    
    # Print summary
    generator.print_summary()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 FAST EMBEDDING GENERATION COMPLETE!")
        print("=" * 60)
        print("⚡ Your Nelson Pediatrics database supports fast semantic search")
        print("🔍 Ready for AI-powered medical queries")
    else:
        print("\n" + "=" * 60)
        print("❌ FAST EMBEDDING GENERATION FAILED!")
        print("=" * 60)
        print("🔍 Check the error logs above for details")

if __name__ == "__main__":
    main()


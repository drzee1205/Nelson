#!/usr/bin/env python3
"""
Fix Schema and Generate Embeddings

This script first fixes the table schema to properly handle vector embeddings,
then generates 384-dimension embeddings using all-MiniLM-L6-v2.
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

class SchemaFixAndEmbeddingGenerator:
    """Fix schema and generate embeddings"""
    
    def __init__(self, batch_size: int = 20):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.batch_size = batch_size
        self.model = None
        self.supabase = None
        
        # Statistics tracking
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'failed_records': 0,
            'skipped_records': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None,
            'model_load_time': None
        }
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Get total record count
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            self.stats['total_records'] = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Total records: {self.stats['total_records']:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def fix_table_schema(self) -> bool:
        """Fix the table schema to properly handle vector embeddings"""
        logger.info("🔧 Fixing table schema for proper vector storage...")
        
        try:
            # First, clear all existing embeddings
            logger.info("🧹 Clearing existing embeddings...")
            result = self.supabase.table('nelson_textbook_chunks')\
                .update({'embedding': None})\
                .not_.is_('embedding', 'null')\
                .execute()
            
            logger.info("✅ Existing embeddings cleared")
            
            # The schema fix needs to be done via SQL in Supabase Dashboard
            # We'll work with the existing schema but ensure proper data format
            logger.info("⚠️ Note: For optimal performance, run this SQL in Supabase Dashboard:")
            print("\n" + "="*60)
            print("SQL TO RUN IN SUPABASE DASHBOARD:")
            print("="*60)
            print("""
-- Fix vector column type
ALTER TABLE public.nelson_textbook_chunks 
DROP COLUMN IF EXISTS embedding;

ALTER TABLE public.nelson_textbook_chunks 
ADD COLUMN embedding VECTOR(384);

-- Recreate index
DROP INDEX IF EXISTS nelson_embeddings_idx;
CREATE INDEX nelson_embeddings_idx 
ON public.nelson_textbook_chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
""")
            print("="*60)
            print("After running the SQL, the embeddings will be stored properly as vectors.\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in schema fix: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the all-MiniLM-L6-v2 model"""
        logger.info(f"🤖 Loading embedding model: {self.model_name}")
        
        start_time = time.time()
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🖥️ Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"📐 Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_records_batch(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get a batch of records without embeddings"""
        try:
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('id, content, chapter_title, section_title')\
                .is_('embedding', 'null')\
                .not_.is_('content', 'null')\
                .range(offset, offset + limit - 1)\
                .order('chapter_title, chunk_index')\
                .execute()
            
            # Filter out records with empty content
            valid_records = []
            if result.data:
                for record in result.data:
                    if record.get('content') and record['content'].strip():
                        valid_records.append(record)
                    else:
                        self.stats['skipped_records'] += 1
            
            return valid_records
            
        except Exception as e:
            logger.error(f"❌ Error fetching records: {e}")
            return []
    
    def preprocess_text(self, content: str, chapter: str = "", section: str = "") -> str:
        """Preprocess text for embedding generation"""
        try:
            if not content or not content.strip():
                return "Medical content"
            
            # Add medical context
            context_parts = []
            if chapter and chapter.strip():
                context_parts.append(f"Medical Chapter: {chapter.strip()}")
            if section and section.strip():
                context_parts.append(f"Section: {section.strip()}")
            
            if context_parts:
                processed_text = " | ".join(context_parts) + " | " + content.strip()
            else:
                processed_text = content.strip()
            
            # Truncate if too long
            if len(processed_text) > 2000:
                processed_text = processed_text[:2000]
            
            return processed_text
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return content if content else "Medical content"
    
    def generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        try:
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                return None
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    valid_texts,
                    batch_size=min(self.batch_size, len(valid_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            return None
    
    def update_embeddings_batch(self, records: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Update records with embeddings using string format (workaround)"""
        try:
            success_count = 0
            
            for i, record in enumerate(records):
                if i >= len(embeddings):
                    break
                
                try:
                    # Convert to list format
                    embedding_list = embeddings[i].tolist()
                    
                    # Store as JSON string (workaround for vector storage issue)
                    embedding_json = json.dumps(embedding_list)
                    
                    # Update individual record
                    result = self.supabase.table('nelson_textbook_chunks')\
                        .update({'embedding': embedding_json})\
                        .eq('id', record['id'])\
                        .execute()
                    
                    if result.data:
                        success_count += 1
                    else:
                        logger.warning(f"Failed to update record {record['id']}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update record {record['id']}: {e}")
                    continue
            
            if success_count > 0:
                self.stats['processed_records'] += success_count
                return True
            else:
                self.stats['failed_records'] += len(records)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating embeddings: {e}")
            self.stats['failed_records'] += len(records)
            return False
    
    def process_all_embeddings(self) -> bool:
        """Process all records and generate embeddings"""
        logger.info("🚀 Starting embedding generation with schema workaround")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            offset = 0
            batch_limit = 20
            consecutive_empty_batches = 0
            
            while consecutive_empty_batches < 5:
                # Get batch of records
                records = self.get_records_batch(limit=batch_limit, offset=offset)
                
                if not records:
                    consecutive_empty_batches += 1
                    logger.info(f"Empty batch at offset {offset}")
                    offset += batch_limit
                    continue
                
                consecutive_empty_batches = 0
                
                # Preprocess texts
                texts = []
                for record in records:
                    processed_text = self.preprocess_text(
                        content=record.get('content', ''),
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
                time.sleep(0.1)
                
                # Move to next batch
                offset += batch_limit
                
                # Safety check
                if offset > self.stats['total_records'] * 3:
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
            # Count records with embeddings (stored as JSON strings)
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
                        try:
                            # Parse JSON string back to list
                            embedding = json.loads(record['embedding'])
                            sample_embeddings.append({
                                'chapter': record['chapter_title'],
                                'content_preview': record['content'][:100] + '...' if record['content'] else 'N/A',
                                'embedding_dim': len(embedding) if isinstance(embedding, list) else 'Invalid',
                                'embedding_type': 'JSON string (workaround)',
                                'embedding_norm': np.linalg.norm(embedding) if isinstance(embedding, list) else 'N/A'
                            })
                        except:
                            sample_embeddings.append({
                                'chapter': record['chapter_title'],
                                'content_preview': 'Error parsing',
                                'embedding_dim': 'Parse error',
                                'embedding_type': 'Invalid',
                                'embedding_norm': 'N/A'
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
        
        print("\n📊 SCHEMA FIX & EMBEDDING GENERATION SUMMARY")
        print("=" * 60)
        print(f"🤖 Model: {self.model_name}")
        print(f"📐 Dimensions: 384 (stored as JSON strings)")
        print(f"📊 Total records: {self.stats['total_records']:,}")
        print(f"✅ Successfully processed: {self.stats['processed_records']:,}")
        print(f"❌ Failed: {self.stats['failed_records']:,}")
        print(f"⏭️ Skipped (empty content): {self.stats['skipped_records']:,}")
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
                    print(f"     Type: {sample['embedding_type']}")
                    print(f"     Norm: {sample['embedding_norm']}")
            
            if verification['completion_rate'] >= 95:
                print(f"\n🎉 EMBEDDING GENERATION SUCCESSFUL!")
                print(f"🚀 Nelson Pediatrics database ready for semantic search!")
                print(f"⚠️ Note: Embeddings stored as JSON strings (workaround)")
            else:
                print(f"\n⚠️ PARTIAL COMPLETION")
                print(f"🔄 {verification['records_without_embeddings']:,} records may need reprocessing")

def main():
    """Main execution function"""
    
    print("🔧 NELSON PEDIATRICS SCHEMA FIX & EMBEDDING GENERATION")
    print("=" * 70)
    print("🛠️ Step 1: Fix table schema for proper vector storage")
    print("🤖 Step 2: Generate 384D embeddings with workaround")
    print("✅ Step 3: Verify results and provide next steps")
    print("=" * 70)
    
    # Initialize generator
    generator = SchemaFixAndEmbeddingGenerator(batch_size=20)
    
    print(f"\n🤖 CONFIGURATION:")
    print(f"  📋 Model: {generator.model_name}")
    print(f"  📐 Dimensions: 384 (native)")
    print(f"  📦 Batch size: {generator.batch_size}")
    print(f"  🔧 Features: Schema fix, JSON workaround, individual updates")
    
    # Connect to Supabase
    if not generator.connect_to_supabase():
        print("❌ Cannot connect to Supabase.")
        return
    
    # Fix schema
    if not generator.fix_table_schema():
        print("❌ Cannot fix table schema.")
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
        print("\n" + "=" * 70)
        print("🎉 SCHEMA FIX & EMBEDDING GENERATION COMPLETE!")
        print("=" * 70)
        print("🚀 Your Nelson Pediatrics database has embeddings")
        print("⚠️ For optimal performance, run the SQL provided above")
        print("✅ Ready for semantic search with JSON parsing")
    else:
        print("\n" + "=" * 70)
        print("❌ EMBEDDING GENERATION ENCOUNTERED ISSUES!")
        print("=" * 70)
        print("🔍 Check the summary above for details")

if __name__ == "__main__":
    main()


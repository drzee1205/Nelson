#!/usr/bin/env python3
"""
Generate Embeddings using Hugging Face Models

This script generates embeddings for all records in the nelson_textbook_chunks table
using state-of-the-art Hugging Face sentence transformer models optimized for medical text.
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
    os.system("pip install sentence-transformers torch transformers")
    from sentence_transformers import SentenceTransformer
    import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

class HuggingFaceEmbeddingGenerator:
    """Generate embeddings using Hugging Face sentence transformers"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 max_seq_length: int = 512):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Hugging Face model name for embeddings
            batch_size: Number of texts to process in each batch
            max_seq_length: Maximum sequence length for the model
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
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
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available embedding models"""
        models = {
            "all-MiniLM-L6-v2": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "description": "Fast, lightweight model good for general text",
                "speed": "Very Fast",
                "quality": "Good",
                "size": "80MB"
            },
            "all-mpnet-base-v2": {
                "name": "sentence-transformers/all-mpnet-base-v2", 
                "dimensions": 768,
                "description": "High quality embeddings, slower but better results",
                "speed": "Medium",
                "quality": "Excellent",
                "size": "420MB"
            },
            "biobert-base": {
                "name": "dmis-lab/biobert-base-cased-v1.1",
                "dimensions": 768,
                "description": "Specialized for biomedical text (requires custom setup)",
                "speed": "Medium",
                "quality": "Excellent for Medical",
                "size": "420MB"
            },
            "clinical-bert": {
                "name": "emilyalsentzer/Bio_ClinicalBERT",
                "dimensions": 768,
                "description": "Specialized for clinical text (requires custom setup)",
                "speed": "Medium", 
                "quality": "Excellent for Clinical",
                "size": "420MB"
            }
        }
        return models
    
    def print_model_options(self):
        """Print available model options"""
        models = self.get_available_models()
        
        print("🤖 AVAILABLE HUGGING FACE EMBEDDING MODELS")
        print("=" * 70)
        
        for key, info in models.items():
            print(f"\n📋 {key.upper()}:")
            print(f"  🔗 Model: {info['name']}")
            print(f"  📐 Dimensions: {info['dimensions']}")
            print(f"  📝 Description: {info['description']}")
            print(f"  ⚡ Speed: {info['speed']}")
            print(f"  🎯 Quality: {info['quality']}")
            print(f"  💾 Size: {info['size']}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("  🚀 For speed: all-MiniLM-L6-v2 (default)")
        print("  🎯 For quality: all-mpnet-base-v2")
        print("  🏥 For medical: biobert-base or clinical-bert")
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Test connection and get record count
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            self.stats['total_records'] = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Total records to process: {self.stats['total_records']:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the Hugging Face sentence transformer model"""
        logger.info(f"🤖 Loading model: {self.model_name}")
        
        start_time = time.time()
        
        try:
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🖥️ Using device: {device}")
            
            # Load the model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Set max sequence length
            self.model.max_seq_length = self.max_seq_length
            
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"📐 Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
            logger.info(f"📏 Max sequence length: {self.max_seq_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_records_without_embeddings(self, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get records that don't have embeddings yet"""
        try:
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('id, content, chapter_title, section_title')\
                .is_('embedding', 'null')\
                .range(offset, offset + limit - 1)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"❌ Error fetching records: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=min(self.batch_size, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
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
                # Convert numpy array to list for JSON serialization
                embedding_list = embeddings[i].tolist()
                
                updates.append({
                    'id': record['id'],
                    'embedding': embedding_list
                })
            
            # Update records in Supabase
            result = self.supabase.table('nelson_textbook_chunks').upsert(updates).execute()
            
            if result.data:
                self.stats['processed_records'] += len(updates)
                return True
            else:
                logger.warning(f"⚠️ Batch update returned no data")
                self.stats['failed_records'] += len(updates)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating embeddings: {e}")
            self.stats['failed_records'] += len(records)
            return False
    
    def process_embeddings(self) -> bool:
        """Process all records and generate embeddings"""
        logger.info("🚀 Starting embedding generation process")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            offset = 0
            batch_limit = 100  # Process 100 records at a time
            
            while True:
                # Get batch of records without embeddings
                records = self.get_records_without_embeddings(limit=batch_limit, offset=offset)
                
                if not records:
                    logger.info("✅ No more records to process")
                    break
                
                # Prepare texts for embedding
                texts = []
                for record in records:
                    # Combine content with context for better embeddings
                    text = record['content']
                    
                    # Add chapter context
                    if record.get('chapter_title'):
                        text = f"Chapter: {record['chapter_title']}\n{text}"
                    
                    # Add section context if available
                    if record.get('section_title'):
                        text = f"Section: {record['section_title']}\n{text}"
                    
                    texts.append(text)
                
                # Generate embeddings for this batch
                embeddings = self.generate_embeddings_batch(texts)
                
                if embeddings is not None:
                    # Update records with embeddings
                    success = self.update_embeddings_batch(records, embeddings)
                    
                    if success:
                        self.stats['batches_processed'] += 1
                        logger.info(f"✅ Processed batch {self.stats['batches_processed']} "
                                  f"({len(records)} records, {self.stats['processed_records']:,} total)")
                        
                        # Progress update every 10 batches
                        if self.stats['batches_processed'] % 10 == 0:
                            progress = (self.stats['processed_records'] / self.stats['total_records']) * 100
                            logger.info(f"📊 Progress: {progress:.1f}% complete "
                                      f"({self.stats['processed_records']:,} / {self.stats['total_records']:,})")
                    else:
                        logger.error(f"❌ Failed to update batch {self.stats['batches_processed'] + 1}")
                else:
                    logger.error(f"❌ Failed to generate embeddings for batch")
                    self.stats['failed_records'] += len(records)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
                
                # Move to next batch
                offset += batch_limit
                
                # Safety check to avoid infinite loops
                if offset > self.stats['total_records'] * 2:
                    logger.warning("⚠️ Safety limit reached, stopping process")
                    break
            
            self.stats['end_time'] = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in embedding process: {e}")
            self.stats['end_time'] = datetime.now()
            return False
    
    def verify_embeddings(self) -> Dict[str, Any]:
        """Verify that embeddings were generated correctly"""
        logger.info("🔍 Verifying embedding generation...")
        
        try:
            # Count records with embeddings
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('count', count='exact')\
                .not_.is_('embedding', 'null')\
                .execute()
            
            records_with_embeddings = result.count if result.count else 0
            
            # Count total records
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('count', count='exact')\
                .execute()
            
            total_records = result.count if result.count else 0
            
            # Get a sample record to check embedding dimensions
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('embedding')\
                .not_.is_('embedding', 'null')\
                .limit(1)\
                .execute()
            
            embedding_dimensions = 0
            if result.data and result.data[0]['embedding']:
                embedding_dimensions = len(result.data[0]['embedding'])
            
            verification = {
                'total_records': total_records,
                'records_with_embeddings': records_with_embeddings,
                'records_without_embeddings': total_records - records_with_embeddings,
                'completion_rate': (records_with_embeddings / total_records * 100) if total_records > 0 else 0,
                'embedding_dimensions': embedding_dimensions,
                'expected_dimensions': self.model.get_sentence_embedding_dimension() if self.model else 0
            }
            
            return verification
            
        except Exception as e:
            logger.error(f"❌ Error verifying embeddings: {e}")
            return {}
    
    def print_summary(self):
        """Print embedding generation summary"""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n📊 EMBEDDING GENERATION SUMMARY")
        print("=" * 60)
        print(f"🤖 Model: {self.model_name}")
        print(f"📊 Total records: {self.stats['total_records']:,}")
        print(f"✅ Processed: {self.stats['processed_records']:,}")
        print(f"❌ Failed: {self.stats['failed_records']:,}")
        print(f"📦 Batches: {self.stats['batches_processed']:,}")
        
        if self.stats['model_load_time']:
            print(f"🤖 Model load time: {self.stats['model_load_time']:.2f} seconds")
        
        if duration:
            print(f"⏱️ Total time: {duration}")
            if self.stats['processed_records'] > 0:
                rate = self.stats['processed_records'] / duration.total_seconds()
                print(f"🚀 Processing rate: {rate:.1f} records/second")
        
        # Verification
        verification = self.verify_embeddings()
        if verification:
            print(f"\n🔍 VERIFICATION RESULTS:")
            print(f"  ✅ Records with embeddings: {verification['records_with_embeddings']:,}")
            print(f"  ❌ Records without embeddings: {verification['records_without_embeddings']:,}")
            print(f"  📈 Completion rate: {verification['completion_rate']:.1f}%")
            print(f"  📐 Embedding dimensions: {verification['embedding_dimensions']}")
            
            if verification['completion_rate'] >= 99:
                print(f"\n🎉 EMBEDDING GENERATION SUCCESSFUL!")
                print(f"🚀 Your Nelson Pediatrics database is ready for semantic search!")
            else:
                print(f"\n⚠️ PARTIAL COMPLETION")
                print(f"🔄 Some records may need reprocessing")

def main():
    """Main execution function"""
    
    print("🤖 NELSON PEDIATRICS EMBEDDING GENERATION")
    print("=" * 60)
    print("🎯 Generate embeddings using Hugging Face models")
    print("🔍 Enable semantic search for medical content")
    print("=" * 60)
    
    # Initialize generator
    generator = HuggingFaceEmbeddingGenerator()
    
    # Show model options
    generator.print_model_options()
    
    print(f"\n🚀 Using model: {generator.model_name}")
    print(f"📦 Batch size: {generator.batch_size}")
    print(f"📏 Max sequence length: {generator.max_seq_length}")
    
    # Get user confirmation
    response = input(f"\n❓ Generate embeddings? Type 'GENERATE' to proceed: ").strip()
    
    if response != "GENERATE":
        print("❌ Embedding generation cancelled.")
        return
    
    # Connect to Supabase
    if not generator.connect_to_supabase():
        print("❌ Cannot connect to Supabase. Check your connection.")
        return
    
    # Load model
    if not generator.load_model():
        print("❌ Cannot load embedding model.")
        return
    
    # Process embeddings
    success = generator.process_embeddings()
    
    # Print summary
    generator.print_summary()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 EMBEDDING GENERATION COMPLETE!")
        print("=" * 60)
        print("🔍 Your Nelson Pediatrics database now supports semantic search")
        print("🤖 Ready for AI-powered medical queries")
    else:
        print("\n" + "=" * 60)
        print("❌ EMBEDDING GENERATION FAILED!")
        print("=" * 60)
        print("🔍 Check the error logs above for details")

if __name__ == "__main__":
    main()


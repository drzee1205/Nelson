#!/usr/bin/env python3
"""
Fix Embedding Dimensions

This script fixes the embedding column dimensions to match Hugging Face models.
It updates the table schema from VECTOR(1536) to VECTOR(768) for compatibility.
"""

import logging
from typing import Dict, Any

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

class EmbeddingDimensionFixer:
    """Fix embedding column dimensions for Hugging Face compatibility"""
    
    def __init__(self):
        self.supabase = None
    
    def connect_to_supabase(self) -> bool:
        """Connect to Supabase"""
        try:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            
            # Test connection
            result = self.supabase.table('nelson_textbook_chunks').select('count', count='exact').execute()
            total_records = result.count if result.count else 0
            
            logger.info(f"✅ Connected to Supabase")
            logger.info(f"📊 Total records: {total_records:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def get_current_schema_info(self) -> Dict[str, Any]:
        """Get current table schema information"""
        try:
            # Get a sample record to check current embedding dimensions
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('embedding')\
                .not_.is_('embedding', 'null')\
                .limit(1)\
                .execute()
            
            current_embedding_dim = 0
            has_embeddings = False
            
            if result.data and result.data[0].get('embedding'):
                current_embedding_dim = len(result.data[0]['embedding'])
                has_embeddings = True
            
            # Count records with embeddings
            result = self.supabase.table('nelson_textbook_chunks')\
                .select('count', count='exact')\
                .not_.is_('embedding', 'null')\
                .execute()
            
            records_with_embeddings = result.count if result.count else 0
            
            return {
                'has_embeddings': has_embeddings,
                'current_embedding_dim': current_embedding_dim,
                'records_with_embeddings': records_with_embeddings,
                'expected_hf_dim': 768,  # Hugging Face all-mpnet-base-v2
                'expected_openai_dim': 1536  # OpenAI embeddings
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting schema info: {e}")
            return {}
    
    def print_schema_analysis(self, schema_info: Dict[str, Any]):
        """Print current schema analysis"""
        print("\n📋 CURRENT SCHEMA ANALYSIS")
        print("=" * 50)
        print(f"📊 Records with embeddings: {schema_info['records_with_embeddings']:,}")
        print(f"📐 Current embedding dimensions: {schema_info['current_embedding_dim']}")
        print(f"🤖 Expected HuggingFace dimensions: {schema_info['expected_hf_dim']}")
        print(f"🔮 Expected OpenAI dimensions: {schema_info['expected_openai_dim']}")
        
        if schema_info['current_embedding_dim'] == schema_info['expected_hf_dim']:
            print("✅ COMPATIBLE: Current dimensions match Hugging Face models")
        elif schema_info['current_embedding_dim'] == schema_info['expected_openai_dim']:
            print("⚠️ MISMATCH: Current dimensions are for OpenAI, need HuggingFace")
        elif schema_info['current_embedding_dim'] == 0:
            print("📝 NO EMBEDDINGS: Table is ready for embedding generation")
        else:
            print(f"❓ UNKNOWN: Unexpected dimension size {schema_info['current_embedding_dim']}")
    
    def fix_embedding_dimensions(self) -> bool:
        """Fix the embedding column dimensions"""
        logger.info("🔧 Fixing embedding column dimensions...")
        
        # SQL commands to fix the dimensions
        fix_sql_commands = [
            # Drop the existing vector index
            "DROP INDEX IF EXISTS nelson_embeddings_idx;",
            
            # Drop the existing embedding column
            "ALTER TABLE public.nelson_textbook_chunks DROP COLUMN IF EXISTS embedding;",
            
            # Add new embedding column with correct dimensions
            "ALTER TABLE public.nelson_textbook_chunks ADD COLUMN embedding VECTOR(768);",
            
            # Recreate the vector index with correct dimensions
            """CREATE INDEX IF NOT EXISTS nelson_embeddings_idx 
               ON public.nelson_textbook_chunks 
               USING ivfflat (embedding vector_cosine_ops) 
               WITH (lists = 100);"""
        ]
        
        print("\n🔧 DIMENSION FIX REQUIRED")
        print("=" * 50)
        print("The table needs to be updated for Hugging Face compatibility.")
        print("This will:")
        print("  1. Drop existing embedding column (if any)")
        print("  2. Create new embedding column with 768 dimensions")
        print("  3. Recreate vector indexes")
        print("  4. Clear any existing embeddings (they'll be regenerated)")
        
        print("\n📋 SQL COMMANDS TO RUN:")
        for i, command in enumerate(fix_sql_commands, 1):
            print(f"\n{i}. {command}")
        
        print("\n" + "=" * 50)
        print("🚨 MANUAL ACTION REQUIRED:")
        print("1. Go to Supabase Dashboard → SQL Editor")
        print("2. Run each SQL command above")
        print("3. Verify the commands execute successfully")
        print("4. Then run the embedding generation script")
        print("=" * 50)
        
        return True

def main():
    """Main execution function"""
    
    print("🔧 NELSON PEDIATRICS EMBEDDING DIMENSION FIX")
    print("=" * 60)
    print("🎯 Fix embedding dimensions for Hugging Face compatibility")
    print("📐 Update from VECTOR(1536) to VECTOR(768)")
    print("=" * 60)
    
    # Initialize fixer
    fixer = EmbeddingDimensionFixer()
    
    # Connect to Supabase
    if not fixer.connect_to_supabase():
        print("❌ Cannot connect to Supabase. Check your connection.")
        return
    
    # Get current schema info
    schema_info = fixer.get_current_schema_info()
    
    if not schema_info:
        print("❌ Cannot analyze current schema.")
        return
    
    # Print analysis
    fixer.print_schema_analysis(schema_info)
    
    # Check if fix is needed
    if schema_info['current_embedding_dim'] == 768:
        print("\n✅ NO FIX NEEDED!")
        print("🎉 Table is already compatible with Hugging Face models")
        print("🚀 You can proceed with embedding generation")
    else:
        # Provide fix instructions
        fixer.fix_embedding_dimensions()

if __name__ == "__main__":
    main()


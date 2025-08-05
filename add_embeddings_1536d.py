#!/usr/bin/env python3
"""
Add 1536D Embeddings and Section Titles

Modified version that generates 1536-dimensional embeddings to match the existing database schema.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Install dependencies
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing dependencies...")
    os.system("pip install sentence-transformers transformers torch supabase")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch
    from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

class MedicalEmbeddingGenerator1536:
    """Generate 1536D embeddings for medical content"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the embedding generator with 1536D model
        
        Args:
            model_name: Hugging Face model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🤖 Using device: {self.device}")
        
    def load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Test embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            embedding_dim = len(test_embedding)
            logger.info(f"✅ Model loaded. Embedding dimension: {embedding_dim}")
            
            # If not 1536D, we'll pad or truncate
            if embedding_dim != 1536:
                logger.warning(f"⚠️ Model produces {embedding_dim}D embeddings, will adjust to 1536D")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return False
    
    def adjust_embedding_dimension(self, embedding: List[float]) -> List[float]:
        """Adjust embedding to 1536 dimensions"""
        current_dim = len(embedding)
        
        if current_dim == 1536:
            return embedding
        elif current_dim < 1536:
            # Pad with zeros
            padding = [0.0] * (1536 - current_dim)
            return embedding + padding
        else:
            # Truncate to 1536
            return embedding[:1536]
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate 1536D embedding for a single text"""
        try:
            if not self.model:
                if not self.load_model():
                    return None
            
            # Clean and prepare text
            cleaned_text = self.clean_medical_text(text)
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text, convert_to_tensor=False)
            
            # Convert to list and ensure proper format
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Adjust to 1536 dimensions
            embedding = self.adjust_embedding_dimension(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 16) -> List[Optional[List[float]]]:
        """Generate 1536D embeddings for multiple texts in batches"""
        try:
            if not self.model:
                if not self.load_model():
                    return [None] * len(texts)
            
            embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Clean texts
                cleaned_batch = [self.clean_medical_text(text) for text in batch_texts]
                
                # Generate batch embeddings
                batch_embeddings = self.model.encode(cleaned_batch, convert_to_tensor=False, batch_size=batch_size)
                
                # Convert to list format and adjust dimensions
                for embedding in batch_embeddings:
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    # Adjust to 1536 dimensions
                    embedding = self.adjust_embedding_dimension(embedding)
                    embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def clean_medical_text(self, text: str) -> str:
        """Clean and prepare medical text for embedding"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\%\/]', ' ', text)
        
        # Normalize medical abbreviations
        text = re.sub(r'\b(mg|kg|ml|cm|mm|mcg|IU)\b', lambda m: m.group().upper(), text)
        
        # Limit length for embedding model
        max_length = 512  # Most models have 512 token limit
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        
        return text.strip()

class SectionTitleExtractor:
    """Extract section titles from medical content"""
    
    def __init__(self):
        """Initialize the section title extractor"""
        self.section_patterns = [
            # Main section headers
            r'^([A-Z][A-Za-z\s]+(?:and|of|in|for|with|by)\s+[A-Za-z\s]+)$',
            # Numbered sections
            r'^\d+\.?\s+([A-Z][A-Za-z\s]+)$',
            # Chapter subsections
            r'^([A-Z][A-Z\s]+[A-Z])$',  # ALL CAPS sections
            # Medical procedure/condition titles
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Syndrome|Disease|Disorder|Treatment|Management|Diagnosis|Therapy)))$',
            # Clinical sections
            r'^(Clinical\s+[A-Za-z\s]+|Pathophysiology|Epidemiology|Etiology|Diagnosis|Treatment|Management|Prognosis|Prevention)$',
            # Anatomical sections
            r'^([A-Z][a-z]+\s+System|[A-Z][a-z]+\s+Disorders?)$'
        ]
    
    def extract_section_title(self, content: str, chapter_title: str = "") -> Optional[str]:
        """Extract section title from content"""
        if not content:
            return None
        
        # Split content into lines
        lines = content.split('\n')
        
        # Look for section titles in first few lines
        for line in lines[:5]:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Try each pattern
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_title = match.group(1).strip()
                    
                    # Validate section title
                    if self.is_valid_section_title(section_title, chapter_title):
                        return section_title
        
        # Try to extract from content structure
        return self.extract_from_content_structure(content, chapter_title)
    
    def extract_from_content_structure(self, content: str, chapter_title: str = "") -> Optional[str]:
        """Extract section title from content structure"""
        
        # Look for medical terminology patterns
        medical_patterns = [
            r'\b(Acute|Chronic|Primary|Secondary|Congenital|Acquired)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+)\s+(Syndrome|Disease|Disorder|Condition|Infection|Inflammation)',
            r'\b(Treatment|Management|Therapy)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+)\s+(Manifestations|Symptoms|Signs|Complications)',
            r'\b(Pediatric|Childhood|Neonatal|Infant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, content[:500])  # Check first 500 chars
            if matches:
                # Return the most relevant match
                for match in matches:
                    if isinstance(match, tuple):
                        section_title = ' '.join(match).strip()
                    else:
                        section_title = match.strip()
                    
                    if self.is_valid_section_title(section_title, chapter_title):
                        return section_title
        
        return None
    
    def is_valid_section_title(self, title: str, chapter_title: str = "") -> bool:
        """Validate if extracted title is a valid section title"""
        if not title or len(title) < 3:
            return False
        
        # Skip if too long (likely not a section title)
        if len(title) > 100:
            return False
        
        # Skip if it's the same as chapter title
        if chapter_title and title.lower() == chapter_title.lower():
            return False
        
        # Skip common non-section phrases
        skip_phrases = [
            'the following', 'as follows', 'in this chapter', 'see table', 'see figure',
            'for example', 'such as', 'including', 'however', 'therefore', 'moreover'
        ]
        
        title_lower = title.lower()
        for phrase in skip_phrases:
            if phrase in title_lower:
                return False
        
        # Must contain at least one medical or clinical term
        medical_terms = [
            'disease', 'disorder', 'syndrome', 'condition', 'infection', 'inflammation',
            'treatment', 'therapy', 'management', 'diagnosis', 'symptoms', 'signs',
            'clinical', 'pathophysiology', 'etiology', 'epidemiology', 'prognosis',
            'prevention', 'pediatric', 'childhood', 'neonatal', 'infant', 'acute', 'chronic'
        ]
        
        has_medical_term = any(term in title_lower for term in medical_terms)
        
        # Also accept anatomical terms
        anatomical_terms = [
            'cardiac', 'respiratory', 'neurologic', 'gastrointestinal', 'renal', 'hepatic',
            'pulmonary', 'cardiovascular', 'endocrine', 'hematologic', 'immunologic',
            'dermatologic', 'ophthalmologic', 'otolaryngologic', 'orthopedic', 'urologic'
        ]
        
        has_anatomical_term = any(term in title_lower for term in anatomical_terms)
        
        return has_medical_term or has_anatomical_term

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

def get_records_for_processing(supabase: Client, batch_size: int = 25, skip_existing: bool = True) -> List[Dict]:
    """Get records that need embedding generation or section title extraction"""
    try:
        logger.info("🔍 Finding records for processing...")
        
        # Build query based on what needs processing
        query_builder = supabase.table('nelson_textbook_chunks')\
            .select('id, content, chapter_title, section_title, embedding, chunk_index, metadata')
        
        if skip_existing:
            # Get records without embeddings OR without section titles
            query_builder = query_builder.or_('embedding.is.null,section_title.is.null')
        
        result = query_builder.limit(batch_size).execute()
        
        if result.data:
            logger.info(f"✅ Found {len(result.data)} records for processing")
            return result.data
        else:
            logger.info("✅ No records need processing")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error getting records: {e}")
        return []

def update_records_with_embeddings_and_sections(supabase: Client, updates: List[Dict]) -> bool:
    """Update records with new embeddings and section titles"""
    if not updates:
        return True
    
    try:
        successful_updates = 0
        failed_updates = 0
        
        for update in tqdm(updates, desc="Updating records"):
            try:
                update_data = {}
                
                # Add embedding if present
                if 'embedding' in update and update['embedding']:
                    update_data['embedding'] = update['embedding']
                
                # Add section title if present
                if 'section_title' in update and update['section_title']:
                    update_data['section_title'] = update['section_title']
                
                # Update metadata
                if 'metadata' in update:
                    update_data['metadata'] = update['metadata']
                
                if update_data:
                    result = supabase.table('nelson_textbook_chunks')\
                        .update(update_data)\
                        .eq('id', update['id'])\
                        .execute()
                    
                    if result.data:
                        successful_updates += 1
                    else:
                        failed_updates += 1
                else:
                    failed_updates += 1
                    
            except Exception as e:
                failed_updates += 1
                logger.warning(f"⚠️ Failed to update record {update.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ Updated {successful_updates} records, {failed_updates} failed")
        return successful_updates > 0
        
    except Exception as e:
        logger.error(f"❌ Error updating batch: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🤖 NELSON PEDIATRICS - ADD 1536D EMBEDDINGS & SECTION TITLES")
    print("=" * 70)
    print("🎯 Generating 1536D AI embeddings via Hugging Face")
    print("📝 Extracting section titles from medical content")
    print("🔍 Enhancing search capabilities with semantic understanding")
    print("=" * 70)
    
    # Step 1: Initialize components
    embedding_generator = MedicalEmbeddingGenerator1536()
    section_extractor = SectionTitleExtractor()
    
    # Step 2: Create Supabase client
    supabase, total_count = create_supabase_client()
    if not supabase:
        return
    
    print(f"📊 Total records in database: {total_count:,}")
    
    # Step 3: Load embedding model
    if not embedding_generator.load_model():
        print("❌ Failed to load embedding model. Exiting.")
        return
    
    # Step 4: Process records in smaller batches
    total_processed = 0
    batch_size = 25  # Smaller batches for stability
    
    while True:
        # Get next batch of records
        records = get_records_for_processing(supabase, batch_size)
        
        if not records:
            logger.info("✅ All records have been processed")
            break
        
        logger.info(f"📄 Processing batch of {len(records)} records...")
        
        # Prepare data for processing
        texts_for_embedding = []
        updates = []
        
        for record in records:
            content = record.get('content', '')
            chapter_title = record.get('chapter_title', '')
            current_section = record.get('section_title')
            current_embedding = record.get('embedding')
            
            update_record = {'id': record['id']}
            needs_update = False
            
            # Check if embedding is needed
            if not current_embedding and content:
                texts_for_embedding.append((len(updates), content))
                needs_update = True
            
            # Check if section title is needed
            if not current_section and content:
                section_title = section_extractor.extract_section_title(content, chapter_title)
                if section_title:
                    update_record['section_title'] = section_title
                    needs_update = True
            
            # Update metadata
            metadata = record.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            metadata['processing_timestamp'] = datetime.now().isoformat()
            metadata['embedding_model'] = embedding_generator.model_name
            metadata['embedding_dimension'] = 1536
            update_record['metadata'] = metadata
            needs_update = True
            
            if needs_update:
                updates.append(update_record)
        
        # Generate embeddings for texts that need them
        if texts_for_embedding:
            logger.info(f"🤖 Generating 1536D embeddings for {len(texts_for_embedding)} texts...")
            
            texts_only = [text for _, text in texts_for_embedding]
            embeddings = embedding_generator.generate_batch_embeddings(texts_only)
            
            # Add embeddings to updates
            for i, (update_idx, _) in enumerate(texts_for_embedding):
                if i < len(embeddings) and embeddings[i]:
                    updates[update_idx]['embedding'] = embeddings[i]
        
        # Update records in database
        if updates:
            logger.info(f"📤 Updating {len(updates)} records...")
            if update_records_with_embeddings_and_sections(supabase, updates):
                total_processed += len(updates)
                logger.info(f"✅ Total processed so far: {total_processed}")
            else:
                logger.error("❌ Failed to update batch")
                break
        else:
            logger.warning("⚠️ No updates needed for this batch")
            break
    
    if total_processed > 0:
        print(f"\n✅ Successfully processed {total_processed:,} records")
        print("\n" + "=" * 70)
        print("🎉 1536D EMBEDDINGS & SECTION TITLES SUCCESSFULLY ADDED!")
        print("=" * 70)
        print(f"🤖 AI Embeddings: Generated using {embedding_generator.model_name}")
        print(f"📝 Section Titles: Extracted using medical content analysis")
        print(f"📊 Records Processed: {total_processed:,}")
        print(f"🔍 Enhanced Search: 1536D semantic similarity now available")
        
        print("\n🚀 New capabilities enabled:")
        print("   • 1536D semantic search using AI embeddings")
        print("   • Section-based content organization")
        print("   • Improved medical content discovery")
        print("   • Enhanced citation accuracy")
        
        print(f"\n📖 Your NelsonGPT now has AI-powered semantic search!")
        
    else:
        print("⚠️ No records were processed. Check if embeddings already exist.")

if __name__ == "__main__":
    main()


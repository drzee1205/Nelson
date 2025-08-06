#!/usr/bin/env python3
"""
Nelson Pediatrics CSV Dataset Preparation

This script processes all text files in the txt_files directory and creates a CSV dataset
that matches the nelson_textbook_chunks schema for Supabase upload.

Schema:
- id: UUID (generated)
- content: text (chunk content)
- chapter_title: text (extracted from file)
- section_title: text (extracted from content)
- page_number: integer (estimated/extracted)
- chunk_index: integer (sequential within chapter)
- metadata: jsonb (additional info)
- created_at: timestamp (current time)
- embedding: vector(384) (null for now, will be added later)
"""

import os
import csv
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NelsonCSVPreparer:
    """Prepare Nelson Pediatrics text files for CSV upload"""
    
    def __init__(self, txt_files_dir: str = "txt_files", output_file: str = "nelson_textbook_chunks.csv"):
        self.txt_files_dir = Path(txt_files_dir)
        self.output_file = output_file
        self.chunk_size = 1000  # Characters per chunk
        self.overlap_size = 200  # Overlap between chunks
        
        # Chapter mapping from filenames
        self.chapter_mapping = {
            "Allergic Disorder.txt": "Allergic Disorders",
            "Behavioural & pyschatrical disorder.txt": "Behavioral and Psychiatric Disorders",
            "Bone and Joint Disorders.txt": "Bone and Joint Disorders",
            "Digestive system.txt": "Digestive System",
            "Diseases of the Blood.txt": "Diseases of the Blood",
            "Ear .txt": "Ear Disorders",
            "Fluid &electrolyte disorder.txt": "Fluid and Electrolyte Disorders",
            "Growth development & behaviour.txt": "Growth, Development and Behavior",
            "Gynecologic History and  Physical Examination.txt": "Gynecologic History and Physical Examination",
            "Humangenetics.txt": "Human Genetics",
            "Rehabilitation Medicine.txt": "Rehabilitation Medicine",
            "Rheumatic Disease.txt": "Rheumatic Diseases",
            "Skin.txt": "Skin Disorders",
            "The Cardiovascular System.txt": "The Cardiovascular System",
            "The Endocrine System.txt": "The Endocrine System",
            "The Nervous System.txt": "The Nervous System",
            "The Respiratory System .txt": "The Respiratory System",
            "Urology.txt": "Urology",
            "aldocent medicine.txt": "Adolescent Medicine",
            "cancer & benign tumor.txt": "Cancer and Benign Tumors",
            "immunology.txt": "Immunology",
            "learning & developmental disorder.txt": "Learning and Developmental Disorders",
            "metabolic disorder.txt": "Metabolic Disorders"
        }
    
    def extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from text content"""
        lines = text.strip().split('\n')
        
        # Look for section patterns
        section_patterns = [
            r'^[A-Z][A-Z\s&-]{10,}$',  # ALL CAPS titles
            r'^Chapter \d+',  # Chapter titles
            r'^[A-Z][a-zA-Z\s&-]{5,}$',  # Title case
        ]
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 5 and len(line) < 100:
                for pattern in section_patterns:
                    if re.match(pattern, line):
                        return line
        
        return None
    
    def estimate_page_number(self, chunk_index: int, total_chunks: int, chapter_name: str) -> Optional[int]:
        """Estimate page number based on chunk position and chapter"""
        
        # Rough page estimates based on medical textbook structure
        chapter_page_estimates = {
            "Allergic Disorders": (1000, 1100),
            "Behavioral and Psychiatric Disorders": (200, 300),
            "Bone and Joint Disorders": (800, 1000),
            "Digestive System": (1200, 1500),
            "Diseases of the Blood": (1600, 1800),
            "Ear Disorders": (2500, 2600),
            "Fluid and Electrolyte Disorders": (200, 300),
            "Growth, Development and Behavior": (50, 150),
            "Gynecologic History and Physical Examination": (700, 800),
            "Human Genetics": (600, 700),
            "Rehabilitation Medicine": (3000, 3100),
            "Rheumatic Diseases": (1100, 1200),
            "Skin Disorders": (3100, 3300),
            "The Cardiovascular System": (1800, 2100),
            "The Endocrine System": (2400, 2700),
            "The Nervous System": (2700, 3000),
            "The Respiratory System": (2100, 2400),
            "Urology": (2600, 2700),
            "Adolescent Medicine": (100, 200),
            "Cancer and Benign Tumors": (2300, 2500),
            "Immunology": (900, 1000),
            "Learning and Developmental Disorders": (150, 250),
            "Metabolic Disorders": (500, 700)
        }
        
        if chapter_name in chapter_page_estimates:
            start_page, end_page = chapter_page_estimates[chapter_name]
            # Distribute chunks across the page range
            if total_chunks > 0:
                page_range = end_page - start_page
                page_offset = int((chunk_index / total_chunks) * page_range)
                return start_page + page_offset
        
        return None
    
    def chunk_text(self, text: str, chapter_title: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = []
        
        # Clean the text
        text = text.strip()
        if not text:
            return chunks
        
        # Split into chunks
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size - 200, start + 500), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) < 50:  # Skip very short chunks
                start = end
                continue
            
            # Extract section title from chunk
            section_title = self.extract_section_title(chunk_text)
            
            # Create chunk record
            chunk_record = {
                'id': str(uuid.uuid4()),
                'content': chunk_text,
                'chapter_title': chapter_title,
                'section_title': section_title,
                'page_number': None,  # Will be estimated later
                'chunk_index': chunk_index,
                'metadata': {
                    'source_file': f"{chapter_title}.txt",
                    'chunk_length': len(chunk_text),
                    'processing_timestamp': datetime.now().isoformat(),
                    'chunk_method': 'sentence_boundary'
                },
                'created_at': datetime.now().isoformat(),
                'embedding': None  # Will be added later
            }
            
            chunks.append(chunk_record)
            chunk_index += 1
            
            # Move start position with overlap
            start = end - self.overlap_size
            if start >= len(text):
                break
        
        # Estimate page numbers for all chunks
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk['page_number'] = self.estimate_page_number(i, total_chunks, chapter_title)
        
        return chunks
    
    def process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single text file"""
        logger.info(f"📖 Processing: {file_path.name}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get chapter title
            chapter_title = self.chapter_mapping.get(file_path.name, file_path.stem)
            
            # Chunk the content
            chunks = self.chunk_text(content, chapter_title)
            
            logger.info(f"✅ Created {len(chunks)} chunks from {file_path.name}")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return []
    
    def create_csv_dataset(self) -> bool:
        """Create CSV dataset from all text files"""
        logger.info("🚀 Starting Nelson Pediatrics CSV dataset creation")
        logger.info(f"📁 Source directory: {self.txt_files_dir}")
        logger.info(f"📄 Output file: {self.output_file}")
        
        if not self.txt_files_dir.exists():
            logger.error(f"❌ Directory {self.txt_files_dir} does not exist")
            return False
        
        # Get all text files
        txt_files = list(self.txt_files_dir.glob("*.txt"))
        if not txt_files:
            logger.error(f"❌ No .txt files found in {self.txt_files_dir}")
            return False
        
        logger.info(f"📚 Found {len(txt_files)} text files to process")
        
        # Process all files
        all_chunks = []
        
        for file_path in txt_files:
            chunks = self.process_text_file(file_path)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.error("❌ No chunks created from text files")
            return False
        
        logger.info(f"📊 Total chunks created: {len(all_chunks)}")
        
        # Write to CSV
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                # Define CSV columns matching Supabase schema
                fieldnames = [
                    'id',
                    'content',
                    'chapter_title',
                    'section_title',
                    'page_number',
                    'chunk_index',
                    'metadata',
                    'created_at',
                    'embedding'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write all chunks
                for chunk in all_chunks:
                    # Convert metadata to JSON string
                    chunk_copy = chunk.copy()
                    chunk_copy['metadata'] = json.dumps(chunk['metadata'])
                    writer.writerow(chunk_copy)
            
            logger.info(f"✅ CSV dataset created successfully: {self.output_file}")
            logger.info(f"📊 Total records: {len(all_chunks)}")
            
            # Print summary statistics
            self.print_dataset_summary(all_chunks)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error writing CSV file: {e}")
            return False
    
    def print_dataset_summary(self, chunks: List[Dict[str, Any]]):
        """Print summary statistics of the dataset"""
        logger.info("\n📊 DATASET SUMMARY")
        logger.info("=" * 50)
        
        # Chapter statistics
        chapter_stats = {}
        section_stats = {}
        page_stats = []
        
        for chunk in chunks:
            chapter = chunk['chapter_title']
            section = chunk['section_title']
            page = chunk['page_number']
            
            # Chapter stats
            if chapter not in chapter_stats:
                chapter_stats[chapter] = 0
            chapter_stats[chapter] += 1
            
            # Section stats
            if section and section not in section_stats:
                section_stats[section] = 0
            if section:
                section_stats[section] += 1
            
            # Page stats
            if page:
                page_stats.append(page)
        
        logger.info(f"📚 Chapters: {len(chapter_stats)}")
        logger.info(f"📝 Sections: {len(section_stats)}")
        logger.info(f"📄 Page range: {min(page_stats) if page_stats else 'N/A'} - {max(page_stats) if page_stats else 'N/A'}")
        logger.info(f"📊 Total chunks: {len(chunks)}")
        logger.info(f"📏 Avg chunk length: {sum(len(c['content']) for c in chunks) / len(chunks):.0f} chars")
        
        logger.info("\n📚 CHAPTERS:")
        for chapter, count in sorted(chapter_stats.items()):
            logger.info(f"  • {chapter}: {count} chunks")
        
        logger.info("\n📝 TOP SECTIONS:")
        top_sections = sorted(section_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for section, count in top_sections:
            logger.info(f"  • {section}: {count} chunks")

def main():
    """Main execution function"""
    
    print("📚 NELSON PEDIATRICS CSV DATASET PREPARATION")
    print("=" * 60)
    print("🎯 Creating CSV dataset for Supabase upload")
    print("📋 Schema: nelson_textbook_chunks")
    print("=" * 60)
    
    # Initialize preparer
    preparer = NelsonCSVPreparer()
    
    # Create dataset
    success = preparer.create_csv_dataset()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 CSV DATASET CREATION COMPLETE!")
        print("=" * 60)
        print(f"📄 Output file: {preparer.output_file}")
        print("📊 Ready for Supabase upload")
        print("\n🚀 Next steps:")
        print("1. Review the CSV file")
        print("2. Upload to Supabase using the web interface or API")
        print("3. Add embeddings using the embedding scripts")
        print("4. Test semantic search functionality")
        
        # Show file info
        if os.path.exists(preparer.output_file):
            file_size = os.path.getsize(preparer.output_file)
            print(f"\n📁 File size: {file_size / 1024 / 1024:.1f} MB")
            
            # Count lines
            with open(preparer.output_file, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            print(f"📊 Total records: {line_count:,}")
    else:
        print("\n❌ CSV dataset creation failed. Check the logs for details.")

if __name__ == "__main__":
    main()

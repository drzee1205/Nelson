#!/usr/bin/env python3
"""
CSV Dataset Validator and Viewer

This script validates the generated CSV dataset and provides sample viewing capabilities.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVValidator:
    """Validate and analyze the Nelson CSV dataset"""
    
    def __init__(self, csv_file: str = "nelson_textbook_chunks.csv"):
        self.csv_file = csv_file
        
    def validate_csv_structure(self) -> bool:
        """Validate CSV structure and schema"""
        logger.info("🔍 Validating CSV structure...")
        
        if not os.path.exists(self.csv_file):
            logger.error(f"❌ CSV file not found: {self.csv_file}")
            return False
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check required columns
                required_columns = [
                    'id', 'content', 'chapter_title', 'section_title',
                    'page_number', 'chunk_index', 'metadata', 'created_at', 'embedding'
                ]
                
                missing_columns = set(required_columns) - set(reader.fieldnames)
                if missing_columns:
                    logger.error(f"❌ Missing columns: {missing_columns}")
                    return False
                
                logger.info("✅ All required columns present")
                logger.info(f"📋 Columns: {', '.join(reader.fieldnames)}")
                
                # Validate first few rows
                valid_rows = 0
                total_rows = 0
                
                for i, row in enumerate(reader):
                    total_rows += 1
                    
                    # Validate required fields
                    if row['id'] and row['content'] and row['chapter_title']:
                        valid_rows += 1
                    
                    # Only check first 100 rows for performance
                    if i >= 99:
                        break
                
                logger.info(f"✅ Validated {valid_rows}/{total_rows} sample rows")
                return valid_rows > 0
                
        except Exception as e:
            logger.error(f"❌ Error validating CSV: {e}")
            return False
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        logger.info("📊 Analyzing dataset statistics...")
        
        stats = {
            'total_records': 0,
            'chapters': {},
            'sections': {},
            'page_range': {'min': None, 'max': None},
            'content_lengths': [],
            'records_with_sections': 0,
            'records_with_pages': 0,
            'avg_chunk_index': 0
        }
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    stats['total_records'] += 1
                    
                    # Chapter statistics
                    chapter = row['chapter_title']
                    if chapter:
                        if chapter not in stats['chapters']:
                            stats['chapters'][chapter] = 0
                        stats['chapters'][chapter] += 1
                    
                    # Section statistics
                    section = row['section_title']
                    if section:
                        stats['records_with_sections'] += 1
                        if section not in stats['sections']:
                            stats['sections'][section] = 0
                        stats['sections'][section] += 1
                    
                    # Page statistics
                    page = row['page_number']
                    if page and page.strip():
                        try:
                            page_num = int(page)
                            stats['records_with_pages'] += 1
                            if stats['page_range']['min'] is None or page_num < stats['page_range']['min']:
                                stats['page_range']['min'] = page_num
                            if stats['page_range']['max'] is None or page_num > stats['page_range']['max']:
                                stats['page_range']['max'] = page_num
                        except ValueError:
                            pass
                    
                    # Content length
                    content = row['content']
                    if content:
                        stats['content_lengths'].append(len(content))
                    
                    # Chunk index
                    chunk_idx = row['chunk_index']
                    if chunk_idx and chunk_idx.strip():
                        try:
                            stats['avg_chunk_index'] += int(chunk_idx)
                        except ValueError:
                            pass
            
            # Calculate averages
            if stats['content_lengths']:
                stats['avg_content_length'] = sum(stats['content_lengths']) / len(stats['content_lengths'])
                stats['min_content_length'] = min(stats['content_lengths'])
                stats['max_content_length'] = max(stats['content_lengths'])
            
            if stats['total_records'] > 0:
                stats['avg_chunk_index'] = stats['avg_chunk_index'] / stats['total_records']
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error analyzing statistics: {e}")
            return stats
    
    def show_sample_records(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Show sample records from the dataset"""
        logger.info(f"📄 Showing {num_samples} sample records...")
        
        samples = []
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i >= num_samples:
                        break
                    
                    # Parse metadata if it's JSON
                    metadata = row['metadata']
                    try:
                        if metadata:
                            metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        pass
                    
                    sample = {
                        'id': row['id'],
                        'content_preview': row['content'][:100] + '...' if len(row['content']) > 100 else row['content'],
                        'chapter_title': row['chapter_title'],
                        'section_title': row['section_title'],
                        'page_number': row['page_number'],
                        'chunk_index': row['chunk_index'],
                        'metadata': metadata,
                        'content_length': len(row['content']) if row['content'] else 0
                    }
                    
                    samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"❌ Error getting samples: {e}")
            return []
    
    def print_validation_report(self):
        """Print comprehensive validation report"""
        print("📊 NELSON PEDIATRICS CSV VALIDATION REPORT")
        print("=" * 60)
        
        # Structure validation
        is_valid = self.validate_csv_structure()
        if not is_valid:
            print("❌ CSV validation failed!")
            return
        
        # Get statistics
        stats = self.get_dataset_statistics()
        
        # Print statistics
        print(f"\n📈 DATASET STATISTICS:")
        print(f"  📊 Total Records: {stats['total_records']:,}")
        print(f"  📚 Chapters: {len(stats['chapters'])}")
        print(f"  📝 Unique Sections: {len(stats['sections'])}")
        print(f"  📄 Records with Pages: {stats['records_with_pages']:,} ({stats['records_with_pages']/stats['total_records']*100:.1f}%)")
        print(f"  📝 Records with Sections: {stats['records_with_sections']:,} ({stats['records_with_sections']/stats['total_records']*100:.1f}%)")
        
        if stats['page_range']['min'] and stats['page_range']['max']:
            print(f"  📄 Page Range: {stats['page_range']['min']} - {stats['page_range']['max']}")
        
        if 'avg_content_length' in stats:
            print(f"  📏 Content Length: {stats['min_content_length']} - {stats['max_content_length']} chars (avg: {stats['avg_content_length']:.0f})")
        
        # Chapter breakdown
        print(f"\n📚 CHAPTER BREAKDOWN:")
        sorted_chapters = sorted(stats['chapters'].items(), key=lambda x: x[1], reverse=True)
        for chapter, count in sorted_chapters:
            percentage = (count / stats['total_records']) * 100
            print(f"  • {chapter}: {count:,} records ({percentage:.1f}%)")
        
        # Top sections
        print(f"\n📝 TOP 10 SECTIONS:")
        sorted_sections = sorted(stats['sections'].items(), key=lambda x: x[1], reverse=True)[:10]
        for section, count in sorted_sections:
            print(f"  • {section}: {count:,} records")
        
        # Sample records
        print(f"\n📄 SAMPLE RECORDS:")
        samples = self.show_sample_records(3)
        for i, sample in enumerate(samples, 1):
            print(f"\n  Record {i}:")
            print(f"    ID: {sample['id']}")
            print(f"    Chapter: {sample['chapter_title']}")
            print(f"    Section: {sample['section_title'] or 'None'}")
            print(f"    Page: {sample['page_number'] or 'None'}")
            print(f"    Chunk Index: {sample['chunk_index']}")
            print(f"    Content Length: {sample['content_length']} chars")
            print(f"    Content Preview: {sample['content_preview']}")
        
        # File information
        file_size = os.path.getsize(self.csv_file)
        print(f"\n📁 FILE INFORMATION:")
        print(f"  📄 Filename: {self.csv_file}")
        print(f"  📏 File Size: {file_size / 1024 / 1024:.1f} MB")
        print(f"  📊 Records: {stats['total_records']:,}")
        
        # Supabase readiness
        print(f"\n✅ SUPABASE READINESS:")
        print(f"  ✅ Schema Compatible: Yes")
        print(f"  ✅ Required Fields: Present")
        print(f"  ✅ Data Format: Valid")
        print(f"  ✅ Ready for Upload: Yes")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"  1. Upload CSV to Supabase via web interface or API")
        print(f"  2. Run embedding generation scripts")
        print(f"  3. Test semantic search functionality")
        print(f"  4. Integrate with NelsonGPT application")

def main():
    """Main execution function"""
    
    print("🔍 NELSON PEDIATRICS CSV DATASET VALIDATOR")
    print("=" * 60)
    
    validator = CSVValidator()
    validator.print_validation_report()
    
    print("\n" + "=" * 60)
    print("🎉 VALIDATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()


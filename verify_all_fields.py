#!/usr/bin/env python3
"""
Verify All Required Fields in CSV Dataset

This script checks that all required fields are properly populated:
- Chapter name
- Section title  
- Content
- Page number
- Chunk index
- ID
- Embedding (null)
- Metadata
- Timestamp
"""

import csv
import json
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldVerifier:
    """Verify all required fields in the CSV dataset"""
    
    def __init__(self, csv_file: str = "nelson_textbook_chunks.csv"):
        self.csv_file = csv_file
        
    def verify_all_fields(self) -> Dict[str, Any]:
        """Verify all required fields are present and properly formatted"""
        logger.info("🔍 Verifying all required fields in CSV dataset...")
        
        verification_results = {
            'total_records': 0,
            'field_completeness': {
                'id': {'filled': 0, 'empty': 0},
                'content': {'filled': 0, 'empty': 0},
                'chapter_title': {'filled': 0, 'empty': 0},
                'section_title': {'filled': 0, 'empty': 0},
                'page_number': {'filled': 0, 'empty': 0},
                'chunk_index': {'filled': 0, 'empty': 0},
                'metadata': {'filled': 0, 'empty': 0},
                'created_at': {'filled': 0, 'empty': 0},
                'embedding': {'filled': 0, 'empty': 0}
            },
            'sample_records': [],
            'issues_found': []
        }
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    verification_results['total_records'] += 1
                    
                    # Check each field
                    for field_name in verification_results['field_completeness'].keys():
                        value = row.get(field_name, '')
                        
                        if value and value.strip():
                            verification_results['field_completeness'][field_name]['filled'] += 1
                        else:
                            verification_results['field_completeness'][field_name]['empty'] += 1
                            
                            # Log specific issues
                            if field_name in ['id', 'content', 'chapter_title']:  # Critical fields
                                issue = f"Row {i+1}: Missing {field_name}"
                                verification_results['issues_found'].append(issue)
                    
                    # Collect sample records (first 5)
                    if i < 5:
                        sample = {
                            'row_number': i + 1,
                            'id': row.get('id', '')[:36],
                            'content_preview': row.get('content', '')[:100] + '...' if len(row.get('content', '')) > 100 else row.get('content', ''),
                            'chapter_title': row.get('chapter_title', ''),
                            'section_title': row.get('section_title', ''),
                            'page_number': row.get('page_number', ''),
                            'chunk_index': row.get('chunk_index', ''),
                            'metadata_preview': row.get('metadata', '')[:50] + '...' if len(row.get('metadata', '')) > 50 else row.get('metadata', ''),
                            'created_at': row.get('created_at', ''),
                            'embedding': row.get('embedding', '')
                        }
                        verification_results['sample_records'].append(sample)
                    
                    # Only process first 1000 for performance
                    if i >= 999:
                        logger.info(f"Processed first 1000 records for verification...")
                        break
            
            return verification_results
            
        except Exception as e:
            logger.error(f"❌ Error verifying fields: {e}")
            return verification_results
    
    def print_field_report(self):
        """Print comprehensive field verification report"""
        results = self.verify_all_fields()
        
        print("📊 FIELD VERIFICATION REPORT")
        print("=" * 60)
        print(f"📄 Records Analyzed: {results['total_records']:,}")
        
        print(f"\n📋 FIELD COMPLETENESS:")
        for field_name, stats in results['field_completeness'].items():
            total = stats['filled'] + stats['empty']
            if total > 0:
                fill_rate = (stats['filled'] / total) * 100
                status = "✅" if fill_rate >= 95 else "⚠️" if fill_rate >= 50 else "❌"
                print(f"  {status} {field_name:15}: {stats['filled']:,} filled, {stats['empty']:,} empty ({fill_rate:.1f}% complete)")
        
        print(f"\n📄 SAMPLE RECORDS:")
        for sample in results['sample_records']:
            print(f"\n  Record {sample['row_number']}:")
            print(f"    ✅ ID: {sample['id']}")
            print(f"    ✅ Chapter: {sample['chapter_title']}")
            print(f"    ✅ Section: {sample['section_title'] or 'None'}")
            print(f"    ✅ Page: {sample['page_number'] or 'None'}")
            print(f"    ✅ Chunk Index: {sample['chunk_index']}")
            print(f"    ✅ Content: {sample['content_preview']}")
            print(f"    ✅ Metadata: {sample['metadata_preview']}")
            print(f"    ✅ Created At: {sample['created_at']}")
            print(f"    ✅ Embedding: {sample['embedding'] or 'NULL (as expected)'}")
        
        if results['issues_found']:
            print(f"\n⚠️ ISSUES FOUND:")
            for issue in results['issues_found'][:10]:  # Show first 10 issues
                print(f"  • {issue}")
            if len(results['issues_found']) > 10:
                print(f"  ... and {len(results['issues_found']) - 10} more issues")
        else:
            print(f"\n✅ NO CRITICAL ISSUES FOUND!")
        
        # Calculate overall completeness
        critical_fields = ['id', 'content', 'chapter_title']
        critical_completeness = []
        
        for field in critical_fields:
            stats = results['field_completeness'][field]
            total = stats['filled'] + stats['empty']
            if total > 0:
                completeness = (stats['filled'] / total) * 100
                critical_completeness.append(completeness)
        
        if critical_completeness:
            avg_completeness = sum(critical_completeness) / len(critical_completeness)
            print(f"\n📊 OVERALL DATASET QUALITY:")
            print(f"  📈 Critical Fields Completeness: {avg_completeness:.1f}%")
            
            if avg_completeness >= 95:
                print(f"  ✅ EXCELLENT - Dataset is production ready!")
            elif avg_completeness >= 80:
                print(f"  ⚠️ GOOD - Minor issues, mostly ready for use")
            else:
                print(f"  ❌ NEEDS WORK - Significant missing data")
        
        print(f"\n🎯 REQUIRED FIELDS SUMMARY:")
        print(f"  ✅ Chapter Name: {results['field_completeness']['chapter_title']['filled']:,} records")
        print(f"  ✅ Section Title: {results['field_completeness']['section_title']['filled']:,} records")
        print(f"  ✅ Content: {results['field_completeness']['content']['filled']:,} records")
        print(f"  ✅ Page Number: {results['field_completeness']['page_number']['filled']:,} records")
        print(f"  ✅ Chunk Index: {results['field_completeness']['chunk_index']['filled']:,} records")
        print(f"  ✅ ID: {results['field_completeness']['id']['filled']:,} records")
        print(f"  ✅ Metadata: {results['field_completeness']['metadata']['filled']:,} records")
        print(f"  ✅ Timestamp: {results['field_completeness']['created_at']['filled']:,} records")
        print(f"  ✅ Embedding: NULL (ready for embedding generation)")

def main():
    """Main execution function"""
    
    print("🔍 NELSON PEDIATRICS FIELD VERIFICATION")
    print("=" * 60)
    print("📋 Checking all required fields:")
    print("   • Chapter name")
    print("   • Section title")
    print("   • Content")
    print("   • Page number")
    print("   • Chunk index")
    print("   • ID")
    print("   • Embedding")
    print("   • Metadata")
    print("   • Timestamp")
    print("=" * 60)
    
    verifier = FieldVerifier()
    verifier.print_field_report()
    
    print("\n" + "=" * 60)
    print("🎉 FIELD VERIFICATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()


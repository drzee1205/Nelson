#!/usr/bin/env python3
"""
Deploy and Test Supabase Functions

This script helps deploy the SQL functions to Supabase and test their functionality.
"""

import os
import json
from datetime import datetime

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Installing dependencies...")
    os.system("pip install supabase")
    from supabase import create_client, Client

# Supabase Configuration
SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def create_supabase_client():
    """Create Supabase client"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        return supabase
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return None

def read_sql_functions():
    """Read the SQL functions file"""
    try:
        with open('supabase_functions.sql', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("❌ supabase_functions.sql file not found")
        return None
    except Exception as e:
        print(f"❌ Error reading SQL file: {e}")
        return None

def test_database_health_check(supabase: Client):
    """Test the database health check function"""
    try:
        print("🔍 Testing database health check function...")
        
        result = supabase.rpc('database_health_check').execute()
        
        if result.data:
            print("✅ Database Health Check Results:")
            print("-" * 50)
            for metric in result.data:
                status_emoji = "✅" if metric['status'] == 'OK' else "⚠️" if metric['status'] == 'WARNING' else "❌"
                print(f"{status_emoji} {metric['metric_name']}: {metric['metric_value']} ({metric['status']})")
            return True
        else:
            print("❌ No health check data returned")
            return False
            
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def test_page_range_search(supabase: Client):
    """Test the page range search function"""
    try:
        print("\n🔍 Testing page range search function...")
        
        result = supabase.rpc('search_page_range', {
            'search_query': 'asthma',
            'min_page_num': 1100,
            'max_page_num': 1200,
            'result_limit': 3
        }).execute()
        
        if result.data:
            print(f"✅ Page Range Search Results: {len(result.data)} results found")
            for i, record in enumerate(result.data[:2], 1):
                print(f"   {i}. Page {record['page_number']}: {record['content'][:80]}...")
            return True
        else:
            print("⚠️ No results from page range search")
            return False
            
    except Exception as e:
        print(f"❌ Page range search test failed: {e}")
        return False

def test_chapter_page_stats(supabase: Client):
    """Test the chapter page statistics function"""
    try:
        print("\n🔍 Testing chapter page statistics function...")
        
        result = supabase.rpc('get_chapter_page_stats', {
            'chapter_pattern': '%allergic%'
        }).execute()
        
        if result.data:
            print("✅ Chapter Page Statistics:")
            for chapter in result.data[:3]:
                print(f"   • {chapter['chapter_title']}: Pages {chapter['min_page']}-{chapter['max_page']} ({chapter['document_count']} docs)")
            return True
        else:
            print("⚠️ No chapter statistics returned")
            return False
            
    except Exception as e:
        print(f"❌ Chapter statistics test failed: {e}")
        return False

def test_page_content_aggregator(supabase: Client):
    """Test the page content aggregator function"""
    try:
        print("\n🔍 Testing page content aggregator function...")
        
        result = supabase.rpc('get_page_content', {
            'target_page': 1101
        }).execute()
        
        if result.data and len(result.data) > 0:
            page_data = result.data[0]
            print(f"✅ Page Content Results:")
            print(f"   • Page {page_data['page_number']}: {page_data['total_chunks']} chunks")
            print(f"   • Chapter: {page_data['chapter_title']}")
            print(f"   • Content length: {len(page_data['full_content']) if page_data['full_content'] else 0} characters")
            return True
        else:
            print("⚠️ No page content returned")
            return False
            
    except Exception as e:
        print(f"❌ Page content test failed: {e}")
        return False

def test_specialty_finder(supabase: Client):
    """Test the medical specialty page finder function"""
    try:
        print("\n🔍 Testing medical specialty finder function...")
        
        result = supabase.rpc('find_specialty_pages', {
            'specialty_keywords': ['asthma', 'allergy'],
            'page_limit': 5
        }).execute()
        
        if result.data:
            print(f"✅ Specialty Finder Results: {len(result.data)} pages found")
            for record in result.data[:2]:
                print(f"   • Page {record['page_number']}: {record['match_count']} matches")
            return True
        else:
            print("⚠️ No specialty pages found")
            return False
            
    except Exception as e:
        print(f"❌ Specialty finder test failed: {e}")
        return False

def test_bulk_page_update(supabase: Client):
    """Test the bulk page number update function"""
    try:
        print("\n🔍 Testing bulk page number update function...")
        
        result = supabase.rpc('update_page_numbers_bulk').execute()
        
        if result.data and len(result.data) > 0:
            update_result = result.data[0]
            print(f"✅ Bulk Update Results:")
            print(f"   • Updated records: {update_result['updated_count']}")
            print(f"   • Processing time: {update_result['processing_time']}")
            return True
        else:
            print("⚠️ No update results returned")
            return False
            
    except Exception as e:
        print(f"❌ Bulk update test failed: {e}")
        return False

def create_api_wrapper_functions():
    """Create Python wrapper functions for easy API integration"""
    
    wrapper_code = '''
# =====================================================
# SUPABASE FUNCTIONS API WRAPPERS
# =====================================================
# Add these functions to your supabase_api.py file

@app.route('/functions/health', methods=['GET'])
def api_database_health():
    """API endpoint for database health check"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        result = supabase.rpc('database_health_check').execute()
        
        if result.data:
            return jsonify({
                "status": "success",
                "health_metrics": result.data,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "No health data available"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/functions/search/advanced', methods=['POST'])
def api_advanced_page_search():
    """API endpoint for advanced page range search"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        result = supabase.rpc('search_page_range', {
            'search_query': data['query'],
            'min_page_num': data.get('min_page', 1),
            'max_page_num': data.get('max_page', 99999),
            'result_limit': data.get('limit', 10)
        }).execute()
        
        return jsonify({
            "status": "success",
            "query": data['query'],
            "results": result.data or [],
            "count": len(result.data) if result.data else 0
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/functions/chapters/stats', methods=['GET'])
def api_chapter_statistics():
    """API endpoint for chapter page statistics"""
    try:
        chapter_pattern = request.args.get('pattern', '%')
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        result = supabase.rpc('get_chapter_page_stats', {
            'chapter_pattern': chapter_pattern
        }).execute()
        
        return jsonify({
            "status": "success",
            "chapters": result.data or [],
            "count": len(result.data) if result.data else 0
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/functions/specialty/<specialty_name>', methods=['GET'])
def api_specialty_pages(specialty_name):
    """API endpoint for medical specialty page finder"""
    try:
        # Convert specialty name to keywords
        specialty_keywords = specialty_name.lower().split('-')
        limit = int(request.args.get('limit', 20))
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        result = supabase.rpc('find_specialty_pages', {
            'specialty_keywords': specialty_keywords,
            'page_limit': limit
        }).execute()
        
        return jsonify({
            "status": "success",
            "specialty": specialty_name,
            "keywords": specialty_keywords,
            "pages": result.data or [],
            "count": len(result.data) if result.data else 0
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
'''
    
    with open('api_function_wrappers.py', 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print("✅ API wrapper functions created in 'api_function_wrappers.py'")

def main():
    """Main execution function"""
    
    print("🚀 SUPABASE FUNCTIONS DEPLOYMENT & TESTING")
    print("=" * 60)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Connect to Supabase
    supabase = create_supabase_client()
    if not supabase:
        print("❌ Cannot connect to Supabase. Exiting.")
        return
    
    print("✅ Connected to Supabase successfully")
    
    # Step 2: Read SQL functions
    sql_content = read_sql_functions()
    if not sql_content:
        print("❌ Cannot read SQL functions file. Exiting.")
        return
    
    print("✅ SQL functions file loaded successfully")
    
    # Step 3: Display deployment instructions
    print("\n📋 DEPLOYMENT INSTRUCTIONS:")
    print("-" * 40)
    print("1. Open your Supabase Dashboard")
    print("2. Go to SQL Editor")
    print("3. Copy and paste the content from 'supabase_functions.sql'")
    print("4. Run the SQL script to create all functions")
    print("5. Come back here to test the functions")
    
    input("\n⏸️  Press Enter after you've deployed the functions in Supabase...")
    
    # Step 4: Test all functions
    print("\n🧪 TESTING DEPLOYED FUNCTIONS:")
    print("-" * 40)
    
    test_results = []
    
    # Test 1: Database health check
    test_results.append(test_database_health_check(supabase))
    
    # Test 2: Page range search
    test_results.append(test_page_range_search(supabase))
    
    # Test 3: Chapter statistics
    test_results.append(test_chapter_page_stats(supabase))
    
    # Test 4: Page content aggregator
    test_results.append(test_page_content_aggregator(supabase))
    
    # Test 5: Specialty finder
    test_results.append(test_specialty_finder(supabase))
    
    # Test 6: Bulk page update
    test_results.append(test_bulk_page_update(supabase))
    
    # Step 5: Create API wrappers
    print("\n🔧 CREATING API WRAPPERS:")
    print("-" * 40)
    create_api_wrapper_functions()
    
    # Step 6: Summary
    successful_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print("🎉 SUPABASE FUNCTIONS DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print(f"✅ Functions tested: {successful_tests}/{total_tests} successful")
    
    if successful_tests == total_tests:
        print("🎯 All functions are working perfectly!")
    elif successful_tests > 0:
        print("⚠️ Some functions need attention - check the logs above")
    else:
        print("❌ Functions may not be deployed correctly - check Supabase")
    
    print("\n🚀 ENHANCED CAPABILITIES:")
    print("• Vector similarity search")
    print("• Advanced page range search")
    print("• Chapter statistics and analytics")
    print("• Medical specialty page finder")
    print("• Bulk page number updates")
    print("• Database health monitoring")
    
    print(f"\n📖 Your NelsonGPT now has powerful database functions!")

if __name__ == "__main__":
    main()


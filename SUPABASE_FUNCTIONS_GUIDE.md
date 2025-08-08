# 🚀 Supabase Functions for Nelson Pediatrics Database

This guide explains how to deploy and use the advanced SQL functions for your Nelson Pediatrics medical database.

## 📋 Quick Start

### 1. Deploy Functions to Supabase

1. **Open Supabase Dashboard**: Go to your project dashboard
2. **Navigate to SQL Editor**: Click on "SQL Editor" in the sidebar
3. **Copy SQL Functions**: Open `supabase_functions.sql` and copy all content
4. **Run SQL Script**: Paste and execute in the SQL Editor
5. **Verify Deployment**: Run the test script below

### 2. Test Deployment

```bash
python deploy_supabase_functions.py
```

## 🔧 Available Functions

### 1. **Vector Similarity Search** 
```sql
SELECT * FROM search_embeddings(
  '[0.1, 0.2, 0.3, ...]'::vector(384),  -- Your query embedding
  0.1,  -- Similarity threshold
  5     -- Number of results
);
```

**Use Case**: Semantic search using AI embeddings for more accurate medical content retrieval.

### 2. **Page Range Search**
```sql
SELECT * FROM search_page_range(
  'asthma treatment',  -- Search query
  1100,               -- Min page
  1200,               -- Max page
  10                  -- Result limit
);
```

**Use Case**: Search within specific page ranges with full-text search ranking.

### 3. **Chapter Page Statistics**
```sql
SELECT * FROM get_chapter_page_stats('allergic%');
```

**Use Case**: Get comprehensive statistics for medical chapters including page ranges and document counts.

### 4. **Medical Specialty Page Finder**
```sql
SELECT * FROM find_specialty_pages(
  ARRAY['asthma', 'allergy', 'respiratory'],
  15
);
```

**Use Case**: Find pages related to specific medical specialties with keyword matching.

### 5. **Page Content Aggregator**
```sql
SELECT * FROM get_page_content(1101);
```

**Use Case**: Get all content from a specific page, properly ordered and aggregated.

### 6. **Bulk Page Number Update**
```sql
SELECT * FROM update_page_numbers_bulk();
```

**Use Case**: Efficiently update page numbers for documents using content-based analysis.

### 7. **Database Health Check**
```sql
SELECT * FROM database_health_check();
```

**Use Case**: Monitor database health with comprehensive metrics and status indicators.

### 8. **Search Analytics Logger**
```sql
SELECT log_search_analytics(
  'asthma treatment',
  'page_range',
  1100,
  1200,
  5
);
```

**Use Case**: Track and analyze search patterns for optimization.

## 🌐 API Integration

### Enhanced API Endpoints

Add these endpoints to your `supabase_api.py`:

#### 1. Database Health Check
```python
@app.route('/functions/health', methods=['GET'])
def api_database_health():
    result = supabase.rpc('database_health_check').execute()
    return jsonify({"health_metrics": result.data})
```

**Usage**:
```bash
curl -X GET http://localhost:5000/functions/health
```

#### 2. Advanced Page Search
```python
@app.route('/functions/search/advanced', methods=['POST'])
def api_advanced_page_search():
    data = request.get_json()
    result = supabase.rpc('search_page_range', {
        'search_query': data['query'],
        'min_page_num': data.get('min_page', 1),
        'max_page_num': data.get('max_page', 99999),
        'result_limit': data.get('limit', 10)
    }).execute()
    return jsonify({"results": result.data})
```

**Usage**:
```bash
curl -X POST http://localhost:5000/functions/search/advanced \
  -H 'Content-Type: application/json' \
  -d '{"query": "asthma", "min_page": 1100, "max_page": 1200}'
```

#### 3. Chapter Statistics
```python
@app.route('/functions/chapters/stats', methods=['GET'])
def api_chapter_statistics():
    pattern = request.args.get('pattern', '%')
    result = supabase.rpc('get_chapter_page_stats', {
        'chapter_pattern': pattern
    }).execute()
    return jsonify({"chapters": result.data})
```

**Usage**:
```bash
curl -X GET "http://localhost:5000/functions/chapters/stats?pattern=allergic%"
```

#### 4. Specialty Page Finder
```python
@app.route('/functions/specialty/<specialty_name>', methods=['GET'])
def api_specialty_pages(specialty_name):
    keywords = specialty_name.lower().split('-')
    result = supabase.rpc('find_specialty_pages', {
        'specialty_keywords': keywords,
        'page_limit': 20
    }).execute()
    return jsonify({"pages": result.data})
```

**Usage**:
```bash
curl -X GET http://localhost:5000/functions/specialty/asthma-allergy
```

## 📊 Performance Optimizations

### Indexes Created
- `idx_nelson_page_number_not_null` - Fast page number queries
- `idx_nelson_chapter_page` - Chapter and page combinations
- `idx_nelson_content_gin` - Full-text search optimization
- `idx_nelson_page_chunk_order` - Page content ordering

### Query Optimization Tips

1. **Use Page Filters**: Always specify page ranges when possible
2. **Limit Results**: Use reasonable limits (≤50) for better performance
3. **Cache Results**: Cache frequently accessed chapter statistics
4. **Batch Updates**: Use bulk functions for large operations

## 🏥 Medical Use Cases

### 1. **Clinical Reference Search**
```sql
-- Find all asthma treatments in pediatric allergy section
SELECT * FROM search_page_range('asthma treatment', 1100, 1300, 10);
```

### 2. **Textbook Navigation**
```sql
-- Get complete content from a specific page for citation
SELECT * FROM get_page_content(1150);
```

### 3. **Specialty Research**
```sql
-- Find all pages related to respiratory conditions
SELECT * FROM find_specialty_pages(
  ARRAY['respiratory', 'pulmonary', 'lung', 'breathing'], 
  25
);
```

### 4. **Chapter Analysis**
```sql
-- Get statistics for all cardiovascular chapters
SELECT * FROM get_chapter_page_stats('%cardiovascular%');
```

## 🔍 Troubleshooting

### Common Issues

1. **Function Not Found Error**
   - Ensure functions are deployed in Supabase SQL Editor
   - Check function names match exactly

2. **Permission Denied**
   - Verify RLS policies allow function execution
   - Check user authentication

3. **Vector Search Not Working**
   - Ensure pgvector extension is enabled
   - Verify embedding column exists and has data

4. **Slow Performance**
   - Check if indexes are created
   - Use appropriate page range limits
   - Consider query optimization

### Debug Commands

```sql
-- Check if functions exist
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_schema = 'public' 
AND routine_name LIKE '%search%';

-- Check index usage
EXPLAIN ANALYZE SELECT * FROM search_page_range('asthma', 1100, 1200, 5);

-- Monitor function performance
SELECT * FROM database_health_check();
```

## 📈 Analytics and Monitoring

### Search Analytics Table
The functions automatically create a `search_analytics` table to track:
- Search terms and patterns
- Page range preferences
- Result counts and performance
- Usage timestamps

### Health Monitoring
Regular health checks provide:
- Document count metrics
- Page coverage statistics
- Content quality indicators
- Performance benchmarks

## 🚀 Next Steps

1. **Deploy Functions**: Run the SQL script in Supabase
2. **Test Functionality**: Use the Python test script
3. **Integrate APIs**: Add wrapper endpoints to your API
4. **Monitor Performance**: Set up regular health checks
5. **Optimize Queries**: Use analytics to improve search patterns

## 💡 Advanced Features

### Custom Specialty Mapping
Extend the page estimation logic for new medical specialties:

```sql
-- Add new specialty in update_page_numbers_bulk function
WHEN content ILIKE '%dermatology%' THEN 3100 + (chunk_index / 20)
```

### Vector Search Integration
For AI-powered semantic search:

```python
# Generate embedding for query
embedding = generate_embedding("asthma treatment")

# Search using vector similarity
result = supabase.rpc('search_embeddings', {
    'query_embedding': embedding,
    'match_threshold': 0.1,
    'match_count': 5
}).execute()
```

---

**🏥 Your Nelson Pediatrics database is now equipped with professional-grade SQL functions for advanced medical search and analytics!**


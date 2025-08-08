#!/usr/bin/env python3
"""Fix embedding column dimension"""

from supabase import create_client, Client

SUPABASE_URL = "https://nrtaztkewvbtzhbtkffc.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ydGF6dGtld3ZidHpoYnRrZmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1NjM3NSwiZXhwIjoyMDY5ODMyMzc1fQ.qJas9ux_U-1V4lbx3XuIeEOIEx68so9kXbwRN7w5gXU"

def fix_embedding_column():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Drop existing embedding column and recreate with correct dimensions
        sql_commands = [
            "ALTER TABLE nelson_textbook_chunks DROP COLUMN IF EXISTS embedding;",
            "ALTER TABLE nelson_textbook_chunks ADD COLUMN embedding vector(384);",
            "CREATE INDEX IF NOT EXISTS idx_nelson_embedding_384 ON nelson_textbook_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        ]
        
        for sql in sql_commands:
            print(f"Executing: {sql}")
            result = supabase.rpc('exec_sql', {'sql': sql}).execute()
            print(f"✅ Success")
        
        print("✅ Embedding column fixed for 384 dimensions")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fix_embedding_column()

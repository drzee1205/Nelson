#!/bin/bash
# MongoDB Import Script for Nelson Pediatrics Data

echo "🚀 Starting MongoDB import..."

# Import using mongoimport (requires MongoDB tools)
mongoimport --uri "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson" \
    --collection nelson_book_content \
    --file nelson_pediatrics_processed.jsonl \
    --jsonArray

echo "✅ Import completed!"

# Alternative: Using Python script
echo "📝 Alternative: Run the following Python script:"
echo "python import_to_mongodb.py"

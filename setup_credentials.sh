#!/bin/bash

# Nelson Pediatrics - Pinecone Credentials Setup Script

echo "🏥 Nelson Pediatrics - Pinecone Setup"
echo "====================================="
echo ""

# Check if API key is already set
if [ ! -z "$PINECONE_API_KEY" ]; then
    echo "✅ PINECONE_API_KEY is already set"
    echo "🌐 Environment: ${PINECONE_ENVIRONMENT:-us-east-1-aws}"
    echo ""
    echo "🚀 Ready to proceed! Run:"
    echo "   python setup_pinecone_complete.py"
    exit 0
fi

echo "🔑 Setting up Pinecone credentials..."
echo ""
echo "📋 To get your Pinecone API key:"
echo "1. Go to https://app.pinecone.io/"
echo "2. Sign up or log in to your account"
echo "3. Create a new project (if needed)"
echo "4. Copy your API key from the dashboard"
echo ""

# Prompt for API key
read -p "🔐 Enter your Pinecone API key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ No API key provided. Exiting."
    exit 1
fi

# Prompt for environment (optional)
echo ""
echo "🌐 Pinecone Environment Options:"
echo "1. us-east-1-aws (default, free tier)"
echo "2. us-west1-gcp"
echo "3. asia-northeast1-gcp"
echo "4. eu-west1-gcp"
echo ""
read -p "🌍 Enter environment (press Enter for default): " environment

if [ -z "$environment" ]; then
    environment="us-east-1-aws"
fi

# Set environment variables
export PINECONE_API_KEY="$api_key"
export PINECONE_ENVIRONMENT="$environment"

echo ""
echo "✅ Credentials set successfully!"
echo "🔑 API Key: ${api_key:0:8}..."
echo "🌐 Environment: $environment"
echo ""

# Save to .env file for persistence
echo "💾 Saving credentials to .env file..."
cat > .env << EOF
# Pinecone Credentials for Nelson Pediatrics
PINECONE_API_KEY=$api_key
PINECONE_ENVIRONMENT=$environment
EOF

echo "✅ Credentials saved to .env file"
echo ""
echo "🚀 Now run the complete setup:"
echo "   source .env"
echo "   python setup_pinecone_complete.py"
echo ""
echo "💡 Or run individual steps:"
echo "   python upload_to_pinecone.py"
echo "   python demo_pinecone_search.py interactive"


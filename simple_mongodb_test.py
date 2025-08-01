#!/usr/bin/env python3
"""
Simple MongoDB connection test with different SSL configurations
"""

import pymongo
import ssl

def test_connection_methods():
    """Test different connection methods"""
    
    connection_string = "mongodb+srv://essaypaisa:P2s4word@nelson.pfga7bt.mongodb.net/?retryWrites=true&w=majority&appName=Nelson"
    
    print("🧪 Testing different MongoDB connection methods...")
    
    # Method 1: Basic connection
    print("\n1️⃣ Testing basic connection...")
    try:
        client = pymongo.MongoClient(connection_string)
        client.admin.command('ping')
        print("✅ Basic connection successful!")
        client.close()
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
    
    # Method 2: With TLS settings
    print("\n2️⃣ Testing with TLS settings...")
    try:
        client = pymongo.MongoClient(
            connection_string,
            tls=True,
            serverSelectionTimeoutMS=5000
        )
        client.admin.command('ping')
        print("✅ TLS connection successful!")
        client.close()
    except Exception as e:
        print(f"❌ TLS connection failed: {e}")
    
    # Method 3: With SSL context
    print("\n3️⃣ Testing with SSL context...")
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        client = pymongo.MongoClient(
            connection_string,
            ssl_context=ssl_context,
            serverSelectionTimeoutMS=5000
        )
        client.admin.command('ping')
        print("✅ SSL context connection successful!")
        client.close()
    except Exception as e:
        print(f"❌ SSL context connection failed: {e}")
    
    # Method 4: Minimal SSL
    print("\n4️⃣ Testing minimal SSL...")
    try:
        client = pymongo.MongoClient(
            connection_string,
            ssl=True,
            ssl_cert_reqs=ssl.CERT_NONE,
            serverSelectionTimeoutMS=5000
        )
        client.admin.command('ping')
        print("✅ Minimal SSL connection successful!")
        client.close()
    except Exception as e:
        print(f"❌ Minimal SSL connection failed: {e}")
    
    # Method 5: No SSL verification
    print("\n5️⃣ Testing without SSL verification...")
    try:
        client = pymongo.MongoClient(
            connection_string,
            tlsInsecure=True,
            serverSelectionTimeoutMS=5000
        )
        client.admin.command('ping')
        print("✅ Insecure connection successful!")
        return client  # Return working client
    except Exception as e:
        print(f"❌ Insecure connection failed: {e}")
    
    return None

if __name__ == "__main__":
    working_client = test_connection_methods()
    
    if working_client:
        print("\n🎉 Found a working connection method!")
        try:
            db = working_client.nelson_pediatrics
            collection = db.nelson_book_content
            count = collection.count_documents({})
            print(f"📊 Current document count: {count}")
        except Exception as e:
            print(f"❌ Error querying database: {e}")
        finally:
            working_client.close()
    else:
        print("\n😞 No working connection method found.")
        print("This might be due to:")
        print("- Network restrictions")
        print("- MongoDB Atlas IP whitelist")
        print("- Incorrect credentials")
        print("- SSL/TLS configuration issues")


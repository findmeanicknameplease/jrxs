#!/usr/bin/env python3
"""
Juraxis Setup Test Script
Tests the environment without downloading any large files
"""

import sys
import importlib
import subprocess
import time
from pathlib import Path

def test_python_imports():
    """Test all required Python imports"""
    print("🐍 Testing Python imports...")
    
    required_packages = [
        'torch',
        'transformers', 
        'sentence_transformers',
        'spacy',
        'nltk',
        'qdrant_client',
        'sqlite3',
        'requests',
        'pandas',
        'numpy',
        'tqdm',
        'psutil',
        'supabase',
        'beautifulsoup4',
        'PyPDF2',
        'python_docx',
        'prometheus_client'
    ]
    
    failed = []
    for package in required_packages:
        try:
            if package == 'python_docx':
                importlib.import_module('docx')
            elif package == 'PyPDF2':
                importlib.import_module('PyPDF2')
            elif package == 'beautifulsoup4':
                importlib.import_module('bs4')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed imports: {', '.join(failed)}")
        return False
    else:
        print("✅ All Python imports successful!")
        return True

def test_embedding_model():
    """Test embedding model loading"""
    print("\n🔢 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try primary model first
        try:
            model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
            print("  ✅ Qwen3-Embedding-8B loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Qwen3-Embedding-8B failed: {e}")
            print("  🔄 Trying fallback model...")
            model = SentenceTransformer('dunzhang/stella_en_400M_v5')
            print("  ✅ Fallback model loaded successfully")
        
        # Test embedding generation
        test_embeddings = model.encode(['This is a test legal sentence.'])
        print(f"  ✅ Generated embedding with shape: {test_embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Embedding model test failed: {e}")
        return False

def test_spacy():
    """Test SpaCy model"""
    print("\n📖 Testing SpaCy model...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp('This is a test legal document.')
        print(f"  ✅ SpaCy processed {len(doc)} tokens")
        return True
    except Exception as e:
        print(f"  ❌ SpaCy test failed: {e}")
        return False

def test_qdrant_connection():
    """Test Qdrant connection"""
    print("\n🔍 Testing Qdrant connection...")
    
    try:
        import requests
        response = requests.get('http://localhost:6333/cluster', timeout=5)
        if response.status_code == 200:
            print("  ✅ Qdrant is running and accessible")
            return True
        else:
            print(f"  ❌ Qdrant returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Qdrant connection failed: {e}")
        print("  💡 Make sure Qdrant is running: systemctl start qdrant")
        return False

def test_disk_space():
    """Test available disk space"""
    print("\n💾 Testing disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('/workspace' if Path('/workspace').exists() else '.')
        
        free_gb = free // (1024**3)
        print(f"  📊 Free disk space: {free_gb} GB")
        
        if free_gb < 10:
            print("  ⚠️  Warning: Less than 10GB free space")
            return False
        elif free_gb < 50:
            print("  ⚠️  Warning: Less than 50GB free space (recommended for full mode)")
            return True
        else:
            print("  ✅ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"  ❌ Disk space check failed: {e}")
        return False

def test_memory():
    """Test available memory"""
    print("\n🧠 Testing memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        total_gb = memory.total // (1024**3)
        available_gb = memory.available // (1024**3)
        
        print(f"  📊 Total memory: {total_gb} GB")
        print(f"  📊 Available memory: {available_gb} GB")
        
        if available_gb < 4:
            print("  ❌ Less than 4GB available memory")
            return False
        elif available_gb < 8:
            print("  ⚠️  Warning: Less than 8GB available memory")
            return True
        else:
            print("  ✅ Sufficient memory available")
            return True
            
    except Exception as e:
        print(f"  ❌ Memory check failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\n🎮 Testing GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"  ✅ GPU available: {gpu_name}")
            print(f"  📊 GPU memory: {memory} GB")
            print(f"  📊 GPU count: {gpu_count}")
            return True
        else:
            print("  ⚠️  No GPU available (CPU mode)")
            return True
    except Exception as e:
        print(f"  ❌ GPU test failed: {e}")
        return True  # GPU is optional

def test_internet_connection():
    """Test internet connection to CourtListener"""
    print("\n🌐 Testing internet connection...")
    
    try:
        import requests
        
        # Test basic internet
        response = requests.get('https://www.google.com', timeout=10)
        print("  ✅ Basic internet connection working")
        
        # Test CourtListener access
        response = requests.head('https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/courts/courts.json.gz', timeout=10)
        if response.status_code in [200, 302]:
            print("  ✅ CourtListener data accessible")
            return True
        else:
            print(f"  ⚠️  CourtListener returned status {response.status_code}")
            return True  # Still proceed
            
    except Exception as e:
        print(f"  ❌ Internet connection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Juraxis Setup Test")
    print("=" * 50)
    
    tests = [
        test_python_imports,
        test_embedding_model,
        test_spacy,
        test_qdrant_connection,
        test_disk_space,
        test_memory,
        test_gpu,
        test_internet_connection
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System ready for Juraxis pipeline.")
        print("\n🚀 Next steps:")
        print("  python runpod_juraxis_pipeline.py --mode test")
        return True
    elif passed >= total - 2:
        print("⚠️  Most tests passed. System should work with minor issues.")
        print("\n🚀 You can try:")
        print("  python runpod_juraxis_pipeline.py --mode test")
        return True
    else:
        print("❌ Multiple tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
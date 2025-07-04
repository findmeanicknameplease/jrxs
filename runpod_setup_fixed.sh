#!/bin/bash

# Juraxis RunPod Setup Script - FIXED VERSION
# Complete environment setup for legal AI data processing on RunPod

set -e  # Exit on any error

echo "🚀 Starting Juraxis RunPod Setup (Fixed Version)..."
echo "=================================================="

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    python3-pip \
    python3-venv \
    sqlite3 \
    libsqlite3-dev \
    postgresql-client \
    libpq-dev \
    unzip \
    p7zip-full \
    htop \
    tree \
    vim \
    nano

# Create workspace directory
echo "📁 Creating workspace directory..."
mkdir -p /workspace/juraxis_data
cd /workspace

# Create scripts directory and copy files
echo "📥 Setting up pipeline files..."
mkdir -p /workspace/scripts
echo "📋 IMPORTANT: Copy these files to /workspace/ before running setup:"
echo "   - runpod_juraxis_pipeline.py"
echo "   - runpod_requirements.txt"
echo ""
echo "⚠️  Setup will continue assuming files are present..."

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# Install core ML/NLP packages first (these take longer)
echo "🤖 Installing ML/NLP packages..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install sentence-transformers
pip3 install transformers
pip3 install spacy
pip3 install nltk
pip3 install scikit-learn

# Install other dependencies from requirements file
echo "📚 Installing other dependencies..."
if [ -f "/workspace/runpod_requirements.txt" ]; then
    pip3 install -r /workspace/runpod_requirements.txt
else
    echo "⚠️  runpod_requirements.txt not found, installing core dependencies manually..."
    pip3 install requests pandas numpy tqdm psutil qdrant-client supabase beautifulsoup4 PyPDF2 python-docx lxml prometheus-client aiohttp regex unidecode py7zr python-dotenv pyyaml colorlog python-dateutil pytest pytest-asyncio openai
fi

# Download SpaCy models
echo "📖 Downloading SpaCy models..."
python3 -m spacy download en_core_web_sm

# Download NLTK data
echo "📝 Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
"

# Install and setup Qdrant
echo "🔍 Setting up Qdrant vector database..."
cd /workspace
wget https://github.com/qdrant/qdrant/releases/download/v1.7.3/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
chmod +x qdrant
mv qdrant /usr/local/bin/

# Create Qdrant service
cat > /etc/systemd/system/qdrant.service << EOF
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace/juraxis_data
ExecStart=/usr/local/bin/qdrant --config-path /workspace/juraxis_data/qdrant_config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create Qdrant configuration
mkdir -p /workspace/juraxis_data/qdrant_storage
cat > /workspace/juraxis_data/qdrant_config.yaml << EOF
log_level: INFO
storage:
  storage_path: /workspace/juraxis_data/qdrant_storage
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
EOF

# Start Qdrant service
systemctl daemon-reload
systemctl enable qdrant
systemctl start qdrant

echo "⏳ Waiting for Qdrant to start..."
sleep 10

# Test Qdrant connection
echo "🧪 Testing Qdrant connection..."
curl -X GET "http://localhost:6333/cluster" || echo "Qdrant connection test failed"

# Install Ollama (optional for local models)
echo "🦙 Installing Ollama (optional)..."
curl -fsSL https://ollama.ai/install.sh | sh

# Download embedding models
echo "🔢 Pre-downloading embedding models..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading Qwen3-Embedding-8B...')
try:
    model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
    print('✅ Qwen3-Embedding-8B downloaded successfully')
except Exception as e:
    print(f'❌ Failed to download Qwen3-Embedding-8B: {e}')
    print('Downloading fallback model...')
    model = SentenceTransformer('dunzhang/stella_en_400M_v5')
    print('✅ Fallback model downloaded successfully')
"

# Setup monitoring
echo "📊 Setting up monitoring..."
pip3 install prometheus-client grafana-api

# Create monitoring directory
mkdir -p /workspace/juraxis_data/monitoring

# Create Prometheus config
cat > /workspace/juraxis_data/monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'juraxis-pipeline'
    static_configs:
      - targets: ['localhost:8000']
EOF

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat > /workspace/juraxis_data/.env << EOF
# Juraxis Pipeline Configuration
WORKSPACE_DIR=/workspace/juraxis_data
QDRANT_HOST=localhost
QDRANT_PORT=6333
SQLITE_DB_PATH=/workspace/juraxis_data/juraxis_cache.db
LOG_LEVEL=INFO
BATCH_SIZE=32
MAX_WORKERS=4
DEEPSEEK_API_KEY=your_deepseek_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
EOF

# Create FIXED run script
echo "🏃 Creating FIXED run script..."
cat > /workspace/juraxis_data/run_pipeline.sh << 'EOF'
#!/bin/bash

# Juraxis Pipeline Runner - FIXED VERSION
echo "🚀 Starting Juraxis Legal AI Pipeline..."

# Check if pipeline file exists
if [ ! -f "/workspace/runpod_juraxis_pipeline.py" ]; then
    echo "❌ Error: runpod_juraxis_pipeline.py not found in /workspace/"
    echo "📋 Please copy the following files to /workspace/:"
    echo "   - runpod_juraxis_pipeline.py"
    echo "   - runpod_requirements.txt"
    exit 1
fi

# Load environment variables
if [ -f "/workspace/juraxis_data/.env" ]; then
    source /workspace/juraxis_data/.env
else
    echo "⚠️  Environment file not found, using defaults"
fi

# Set Python path
export PYTHONPATH=/workspace:$PYTHONPATH

# Run pipeline from correct location
cd /workspace
python3 runpod_juraxis_pipeline.py \
    --mode full \
    --output-dir /workspace/juraxis_data \
    --qdrant-host localhost \
    --qdrant-port 6333 \
    --sample-size 1000

echo "✅ Pipeline completed!"
EOF

chmod +x /workspace/juraxis_data/run_pipeline.sh

# Create FIXED test script
echo "🧪 Creating FIXED test script..."
cat > /workspace/juraxis_data/test_setup.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing Juraxis Setup..."

# Check if pipeline file exists
if [ -f "/workspace/runpod_juraxis_pipeline.py" ]; then
    echo "✅ Pipeline file found"
else
    echo "❌ Pipeline file missing: /workspace/runpod_juraxis_pipeline.py"
fi

# Test Python imports
echo "Testing Python imports..."
python3 -c "
import torch
import sentence_transformers
import transformers
import spacy
import nltk
import qdrant_client
import sqlite3
import requests
import pandas as pd
import numpy as np
print('✅ All Python imports successful')
"

# Test Qdrant
echo "Testing Qdrant connection..."
curl -s -X GET "http://localhost:6333/cluster" > /dev/null && echo "✅ Qdrant is running" || echo "❌ Qdrant connection failed"

# Test embedding model
echo "Testing embedding model..."
python3 -c "
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
    embeddings = model.encode(['test sentence'])
    print(f'✅ Embedding model working, output shape: {embeddings.shape}')
except Exception as e:
    print(f'❌ Embedding model test failed: {e}')
"

# Test SpaCy
echo "Testing SpaCy..."
python3 -c "
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test sentence.')
print(f'✅ SpaCy working, processed {len(doc)} tokens')
"

# Check disk space
echo "Checking disk space..."
df -h /workspace

# Check memory
echo "Checking memory..."
free -h

echo "🎉 Setup test completed!"
EOF

chmod +x /workspace/juraxis_data/test_setup.sh

# Create monitoring dashboard
echo "📊 Creating monitoring dashboard..."
cat > /workspace/juraxis_data/monitoring/dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Simple monitoring dashboard for Juraxis Pipeline
"""

import time
import psutil
import sqlite3
from pathlib import Path
import json

def get_system_metrics():
    """Get system metrics"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/workspace').percent,
        'timestamp': time.time()
    }

def get_pipeline_metrics():
    """Get pipeline metrics from database"""
    db_path = Path('/workspace/juraxis_data/juraxis_cache.db')
    if not db_path.exists():
        return {'documents': 0, 'embeddings': 0}
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        
        return {
            'documents': doc_count,
            'embeddings': embedding_count
        }
    except Exception as e:
        return {'documents': 0, 'embeddings': 0, 'error': str(e)}
    finally:
        conn.close()

def main():
    """Main monitoring loop"""
    print("📊 Juraxis Pipeline Monitor")
    print("=" * 40)
    
    while True:
        try:
            system_metrics = get_system_metrics()
            pipeline_metrics = get_pipeline_metrics()
            
            print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💻 CPU: {system_metrics['cpu_percent']:.1f}%")
            print(f"🧠 Memory: {system_metrics['memory_percent']:.1f}%")
            print(f"💾 Disk: {system_metrics['disk_percent']:.1f}%")
            print(f"📄 Documents: {pipeline_metrics.get('documents', 0):,}")
            print(f"🔢 Embeddings: {pipeline_metrics.get('embeddings', 0):,}")
            
            time.sleep(30)  # Update every 30 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
EOF

chmod +x /workspace/juraxis_data/monitoring/dashboard.py

# Final setup
echo "🎯 Final setup..."
cd /workspace/juraxis_data

# Create logs directory
mkdir -p logs

# Set permissions
chown -R root:root /workspace/juraxis_data
chmod -R 755 /workspace/juraxis_data

# Create completion marker
echo "$(date): Juraxis RunPod setup completed successfully" > setup_complete.txt

# Display final information
echo ""
echo "🎉 Juraxis RunPod Setup Complete (FIXED)!"
echo "=================================================="
echo "📍 Workspace: /workspace/juraxis_data"
echo "🗄️  Database: /workspace/juraxis_data/juraxis_cache.db"
echo "🔍 Qdrant: http://localhost:6333"
echo "📊 Monitoring: python3 /workspace/juraxis_data/monitoring/dashboard.py"
echo ""
echo "🚀 To run the pipeline:"
echo "   cd /workspace/juraxis_data"
echo "   ./run_pipeline.sh"
echo ""
echo "🧪 To test the setup:"
echo "   cd /workspace/juraxis_data"
echo "   ./test_setup.sh"
echo ""
echo "📋 Configuration file: /workspace/juraxis_data/.env"
echo ""
echo "✅ Setup completed successfully!"

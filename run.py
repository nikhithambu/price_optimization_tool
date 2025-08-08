#!/usr/bin/env python3
"""
Startup script for Price Optimization Tool
This script will check if the model exists, train it if needed, and start the Flask app.
"""

import os
import sys
import subprocess

def check_model_exists():
    """Check if model.pkl exists"""
    return os.path.exists('model.pkl')

def train_model():
    """Run the model training script"""
    print("🔄 Training model...")
    try:
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True, check=True)
        print("✅ Model training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def start_app():
    """Start the Flask application"""
    print("🚀 Starting Flask application...")
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Flask app: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

def main():
    print("=" * 60)
    print("🎯 PRICE OPTIMIZATION TOOL STARTUP")
    print("=" * 60)
    
    # Check if model exists
    if not check_model_exists():
        print("📋 Model not found. Training new model...")
        if not train_model():
            print("❌ Cannot start application without trained model.")
            sys.exit(1)
    else:
        print("✅ Model found!")
    
    # Start the Flask application
    print("\n🌐 Starting web application at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    start_app()

if __name__ == "__main__":
    main()

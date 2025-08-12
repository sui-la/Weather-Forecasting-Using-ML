#!/usr/bin/env python3
"""
Weather Prediction System - Backend Server Runner
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

# Change to backend directory
os.chdir(backend_path)

# Import and run the Flask app
from app import app

if __name__ == '__main__':
    print("🌤️  Weather Prediction System - Backend Server")
    print("=" * 50)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1) 
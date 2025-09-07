#!/usr/bin/env python3
"""
Weather Prediction System - Frontend Server
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Change to frontend directory
frontend_path = Path(__file__).parent / 'frontend'
os.chdir(frontend_path)

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == '__main__':
    print("🌤️  Weather Prediction System - Frontend Server")
    print("=" * 50)
    print(f"Starting HTTP server on port {PORT}...")
    print(f"Frontend will be available at: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"✅ Server started successfully!")
            print(f"📱 Open your browser and go to: http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1) 
#!/usr/bin/env python3
"""
Simple HTTP Server for Alphabet Image Display Application
Serves the Text-to-Sign web interface
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_simple_server():
    """Start the simple HTTP server"""
    PORT = 8001
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("SIMPLE HTTP SERVER FOR TEXT-TO-SIGN")
    print("=" * 60)
    print(f"Starting server on port {PORT}...")
    print(f"Serving files from: {script_dir}")
    print()
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"SUCCESS: Server started at http://localhost:{PORT}")
            print("Text-to-Sign application is now available!")
            print()
            print("Features available:")
            print("- Type text to see sign language images")
            print("- Voice input support")
            print("- Word history")
            print("- ASL learning mode")
            print()
            print("Press Ctrl+C to stop the server")
            print("=" * 60)
            
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}')
                print("Browser opened automatically")
            except:
                print(f"Please manually open: http://localhost:{PORT}")
            
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 10048:  # Port already in use
            print(f"ERROR: Port {PORT} is already in use")
            print("Please stop any other server using this port and try again")
        else:
            print(f"ERROR: Failed to start server: {e}")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")

if __name__ == "__main__":
    start_simple_server()

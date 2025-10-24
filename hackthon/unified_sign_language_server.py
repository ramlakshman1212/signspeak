#!/usr/bin/env python3
"""
Unified Sign Language Server
Combines Text-to-Sign and Advanced Sign-to-Text in a single server
"""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
import subprocess
import os
import sys
import threading
import time
from pathlib import Path

app = Flask(__name__)

class UnifiedSignLanguageServer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.abcd_dir = self.base_dir / "abcd"
        self.sign_lang_dir = self.base_dir / "Sign-Language-To-Text-Conversion-main"
        self.advanced_app_process = None
        self.is_advanced_running = False

    def start_advanced_app(self):
        """Start the Advanced Sign-to-Text application"""
        try:
            if self.is_advanced_running:
                return {"success": False, "message": "Advanced Sign-to-Text is already running"}
            
            # Check if the directory exists
            if not self.sign_lang_dir.exists():
                return {"success": False, "message": "Sign language directory not found"}
            
            # Check if the application file exists
            app_file = self.sign_lang_dir / "AdvancedGestureRecognitionApp.py"
            if not app_file.exists():
                return {"success": False, "message": "AdvancedGestureRecognitionApp.py not found"}
            
            # Change to the sign language directory
            original_dir = os.getcwd()
            os.chdir(self.sign_lang_dir)
            
            # Start the application
            self.advanced_app_process = subprocess.Popen([sys.executable, "AdvancedGestureRecognitionApp.py"])
            self.is_advanced_running = True
            
            # Change back to original directory
            os.chdir(original_dir)
            
            return {"success": True, "message": "Advanced Sign-to-Text application started successfully! Check your taskbar for the application window."}
            
        except Exception as e:
            return {"success": False, "message": f"Failed to start application: {str(e)}"}

    def stop_advanced_app(self):
        """Stop the Advanced Sign-to-Text application"""
        try:
            if self.advanced_app_process and self.is_advanced_running:
                self.advanced_app_process.terminate()
                self.advanced_app_process = None
                self.is_advanced_running = False
                return {"success": True, "message": "Advanced Sign-to-Text application stopped successfully"}
            else:
                return {"success": False, "message": "Advanced Sign-to-Text application is not running"}
        except Exception as e:
            return {"success": False, "message": f"Failed to stop application: {str(e)}"}

    def get_status(self):
        """Get the current status of the applications"""
        return {
            "text_to_sign": True,  # Always available through this server
            "advanced_sign_to_text": self.is_advanced_running,
            "message": "Advanced Sign-to-Text is running" if self.is_advanced_running else "Advanced Sign-to-Text is not running"
        }

# Create server instance
server = UnifiedSignLanguageServer()

# Serve static files from abcd directory
@app.route('/alphabets/<path:filename>')
def serve_alphabets(filename):
    """Serve alphabet images"""
    return send_from_directory(server.abcd_dir / 'alphabets', filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve data images"""
    return send_from_directory(server.abcd_dir / 'data', filename)

@app.route('/style.css')
def serve_css():
    """Serve CSS file"""
    return send_from_directory(server.abcd_dir, 'style.css')

@app.route('/script.js')
def serve_js():
    """Serve JavaScript file"""
    return send_from_directory(server.abcd_dir, 'script.js')

# API endpoints
@app.route('/api/start-advanced-app', methods=['POST'])
def start_advanced_app():
    """API endpoint to start the Advanced Sign-to-Text application"""
    result = server.start_advanced_app()
    return jsonify(result)

@app.route('/api/stop-advanced-app', methods=['POST'])
def stop_advanced_app():
    """API endpoint to stop the Advanced Sign-to-Text application"""
    result = server.stop_advanced_app()
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get the current status"""
    result = server.get_status()
    return jsonify(result)

# Main interface
@app.route('/')
def index():
    """Serve the unified interface"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Sign Language System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .apps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .app-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-align: center;
        }

        .app-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }

        .app-card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.8rem;
        }

        .app-card p {
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }

        .app-button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }

        .app-button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .app-button.secondary {
            background: linear-gradient(45deg, #28a745, #20c997);
        }

        .app-button.advanced {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        }

        .app-button.stop {
            background: linear-gradient(45deg, #dc3545, #c82333);
        }

        .status-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .status-item {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background: #f8f9fa;
        }

        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin: 0 auto 10px;
            background: #28a745;
        }

        .status-indicator.offline {
            background: #dc3545;
        }

        .message-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            display: none;
        }

        .message-box.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }

        .footer {
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.8;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .apps-grid {
                grid-template-columns: 1fr;
            }
            
            .app-card {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤟 Unified Sign Language System</h1>
            <p>Text-to-Sign + Advanced Sign-to-Text in One Server</p>
        </div>

        <div class="status-section">
            <h2 style="text-align: center; margin-bottom: 20px; color: #667eea;">System Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-indicator" id="text-to-sign-status"></div>
                    <h4>Text-to-Sign</h4>
                    <p>Available (Built-in)</p>
                </div>
                <div class="status-item">
                    <div class="status-indicator offline" id="sign-to-text-status"></div>
                    <h4>Advanced Sign-to-Text</h4>
                    <p id="sign-to-text-message">Not started</p>
                </div>
            </div>
        </div>

        <div class="apps-grid">
            <div class="app-card">
                <h2>📝 Text to Sign</h2>
                <p>Convert text to sign language images. Type any word and see the corresponding sign language alphabet images.</p>
                <button class="app-button" onclick="openTextToSign()">Open Text-to-Sign</button>
                <button class="app-button secondary" onclick="checkTextToSignStatus()">Check Status</button>
            </div>

            <div class="app-card">
                <h2>📷 Advanced Sign to Text</h2>
                <p>Advanced gesture recognition that can detect all letters A-Z. Click the button below to start it.</p>
                <button class="app-button advanced" onclick="startAdvancedApp()" id="start-advanced-btn">Start Advanced Sign-to-Text</button>
                <button class="app-button stop" onclick="stopAdvancedApp()" id="stop-advanced-btn" style="display: none;">Stop Advanced App</button>
                <button class="app-button secondary" onclick="checkAdvancedStatus()">Check Status</button>
            </div>
        </div>

        <div class="message-box" id="message-box"></div>

        <div class="footer">
            <p>Unified Sign Language System - Single Server Solution</p>
        </div>
    </div>

    <script>
        let advancedAppRunning = false;

        function showMessage(message, isError = false) {
            const messageBox = document.getElementById('message-box');
            messageBox.textContent = message;
            messageBox.className = 'message-box' + (isError ? ' error' : '');
            messageBox.style.display = 'block';
            setTimeout(() => {
                messageBox.style.display = 'none';
            }, 5000);
        }

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusIndicator = document.getElementById('sign-to-text-status');
                    const statusMessage = document.getElementById('sign-to-text-message');
                    const startBtn = document.getElementById('start-advanced-btn');
                    const stopBtn = document.getElementById('stop-advanced-btn');
                    
                    if (data.advanced_sign_to_text) {
                        statusIndicator.classList.remove('offline');
                        statusMessage.textContent = 'Running';
                        startBtn.style.display = 'none';
                        stopBtn.style.display = 'inline-block';
                        advancedAppRunning = true;
                    } else {
                        statusIndicator.classList.add('offline');
                        statusMessage.textContent = 'Not started';
                        startBtn.style.display = 'inline-block';
                        stopBtn.style.display = 'none';
                        advancedAppRunning = false;
                    }
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                });
        }

        function openTextToSign() {
            // Open Text-to-Sign in a new tab
            window.open('/text-to-sign', '_blank');
        }

        function startAdvancedApp() {
            showMessage('Starting Advanced Sign-to-Text application...');
            
            fetch('/api/start-advanced-app', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage(data.message);
                    updateStatus();
                } else {
                    showMessage(data.message, true);
                }
            })
            .catch(error => {
                showMessage('Error starting application: ' + error.message, true);
            });
        }

        function stopAdvancedApp() {
            showMessage('Stopping Advanced Sign-to-Text application...');
            
            fetch('/api/stop-advanced-app', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage(data.message);
                    updateStatus();
                } else {
                    showMessage(data.message, true);
                }
            })
            .catch(error => {
                showMessage('Error stopping application: ' + error.message, true);
            });
        }

        function checkTextToSignStatus() {
            document.getElementById('text-to-sign-status').classList.remove('offline');
            showMessage('SUCCESS: Text-to-Sign is available!');
        }

        function checkAdvancedStatus() {
            updateStatus();
            if (advancedAppRunning) {
                showMessage('Advanced Sign-to-Text application is running. Check your taskbar for the application window.');
            } else {
                showMessage('Advanced Sign-to-Text application is not running. Click "Start Advanced Sign-to-Text" to start it.');
            }
        }

        // Auto-check status on page load and every 5 seconds
        window.onload = function() {
            checkTextToSignStatus();
            updateStatus();
            setInterval(updateStatus, 5000);
        };
    </script>
</body>
</html>
    ''')

# Text-to-Sign interface - serve the EXACT original interface
@app.route('/text-to-sign')
def text_to_sign():
    """Serve the EXACT original Text-to-Sign interface"""
    # Read the original index.html file
    index_file = server.abcd_dir / 'index.html'
    with open(index_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    print("=" * 80)
    print("UNIFIED SIGN LANGUAGE SERVER")
    print("=" * 80)
    print("Starting unified server with both features...")
    print("Server will be available at http://localhost:5000")
    print()
    print("Features:")
    print("1. Text-to-Sign: http://localhost:5000/text-to-sign")
    print("2. Advanced Sign-to-Text: Use the button in main interface")
    print("3. Main Interface: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)

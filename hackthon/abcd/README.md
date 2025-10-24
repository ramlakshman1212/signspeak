# 🔤 Alphabet Image Display Application

A beautiful, modern web application that displays alphabet images based on user input text, with **word history preservation** - previous words and their images are kept visible!

## ✨ Features

- **📚 Word History**: Previous words and their images are preserved and visible
- **🎨 Modern Web Interface**: Beautiful, responsive design with hover effects
- **⚡ Real-time Updates**: Images update automatically as you type
- **🖼️ Image Display**: Shows corresponding alphabet images in horizontal sequence
- **📱 Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **🎯 Interactive Elements**: Hover effects, click to view in modal
- **⌨️ Keyboard Shortcuts**: Ctrl+Enter to show, Esc to clear, Ctrl+K to focus input
- **🔍 Error Handling**: Beautiful placeholders for missing images
- **🗑️ Clear Function**: One-click clear with visual feedback
- **📊 Statistics**: Live word and letter counts
- **📁 Export**: Export your word history as JSON
- **💡 Smart Status**: Real-time status updates with emojis

## Requirements

- Python 3.6+ (for the local server)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- `alphabets/` folder with A.jpg through Z.jpg images

## 🚀 Quick Start (Web Version - Recommended)

1. **Start the web server:**
```bash
python server.py
```

2. **Open your browser:**
   - The application will automatically open at `http://localhost:8000`
   - Or manually navigate to `http://localhost:8000`

3. **Use the application:**
   - Type any text in the input box (e.g., "hello world")
   - Click "Show Images" or press **Ctrl+Enter**
   - **Previous words are automatically preserved!**
   - Click on any word card to view in full-screen modal
   - Use **Esc** or "Clear All" to remove all words
   - Export your word history as JSON

## 🖥️ Desktop Version (Alternative)

If you prefer the desktop version:

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the desktop application:**
```bash
python alphabet_display.py
```

## How it Works

- The application extracts only alphabetic characters from your input
- Spaces and special characters are ignored
- Each letter is converted to uppercase
- Corresponding images are loaded from the `alphabets/` folder
- Images are displayed horizontally in sequence
- Missing images show as red placeholders with the letter

## Example

Input: "Hello World!"
Result: Displays H.jpg, E.jpg, L.jpg, L.jpg, O.jpg, W.jpg, O.jpg, R.jpg, L.jpg, D.jpg

## File Structure

```
project/
├── alphabet_display.py    # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── alphabets/            # Folder containing alphabet images
    ├── A.jpg
    ├── B.jpg
    ├── ...
    └── Z.jpg
```




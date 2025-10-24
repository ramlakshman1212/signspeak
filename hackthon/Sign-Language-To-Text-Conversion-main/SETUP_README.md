# Sign Language To Text Conversion

A real-time sign language recognition system that converts American Sign Language (ASL) gestures to text using computer vision and deep learning.

## Features

- Real-time sign language recognition using webcam
- Support for all 26 letters (A-Z) plus blank gesture
- Multi-layer CNN model for improved accuracy
- GUI interface with live video feed
- Word suggestion system using Hunspell
- Data collection tools for training/testing

## Requirements

- Python 3.7+
- Webcam
- Windows/Linux/macOS

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the folder creation script to set up the data structure:
   ```bash
   python FoldersCreation.py
   ```

## Usage

### Running the Application

To run the main sign language recognition application:
```bash
python Application.py
```

### Data Collection

To collect training data:
```bash
python TrainingDataCollection.py
```

To collect testing data:
```bash
python TestingDataCollection.py
```

### Model Training

To train or retrain the models, open the Jupyter notebook:
```bash
jupyter notebook Models/Model.ipynb
```

## How to Use

1. **Launch the application**: Run `Application.py`
2. **Position your hand**: Place your hand in the blue rectangle on the screen
3. **Make gestures**: Perform ASL letter gestures
4. **View results**: The recognized letters will appear in the GUI
5. **Use suggestions**: Click the suggestion buttons to correct words

## Data Collection Instructions

1. Run the data collection script
2. Position your hand in the blue rectangle
3. Press the corresponding letter key on your keyboard to capture the gesture
4. Press 'ESC' to exit

## Model Architecture

The system uses multiple CNN models:
- **Main Model**: Recognizes all 27 classes (A-Z + blank)
- **DRU Model**: Specialized for D, R, U letters
- **TKDI Model**: Specialized for T, K, D, I letters  
- **SMN Model**: Specialized for S, M, N letters

## File Structure

```
├── Application.py              # Main application
├── FoldersCreation.py         # Creates data folder structure
├── TrainingDataCollection.py  # Collects training data
├── TestingDataCollection.py   # Collects testing data
├── Models/                    # Model files and training notebook
├── dataSet/                   # Training and testing data
├── images/                    # Documentation images
└── requirements.txt           # Python dependencies
```

## Troubleshooting

- **Camera not working**: Ensure your webcam is connected and not being used by another application
- **Model files missing**: Run the training notebook to generate model files
- **Poor recognition**: Ensure good lighting and clear hand gestures
- **Dependencies issues**: Make sure all packages in requirements.txt are installed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.




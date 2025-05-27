# Real-Time Face Recognition System

A comprehensive face recognition system with CUDA acceleration, multiple detection models, and interactive Jupyter notebook interface.

## 🚀 Features

- **Real-time face detection and recognition** using webcam
- **Multiple detection models**: Haar Cascade, DNN MobileNet, MTCNN (optional)
- **CUDA acceleration** with automatic CPU fallback
- **Interactive Jupyter notebook** interface with widgets
- **Face database management** with CRUD operations
- **Performance monitoring** and benchmarking
- **Anti-spoofing** features (basic movement detection)
- **Export/Import** functionality for face databases

## 📋 Requirements

- Python 3.13+
- NVIDIA GPU with CUDA support (optional)
- Webcam for face capture
- 4GB+ RAM recommended

## 🛠️ Installation

1. **Clone or download this project**
2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models (if needed):**
   ```bash
   python download_models.py
   ```

## 🎯 Usage

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter lab
# Open face_recognition_system.ipynb
# Run cells 1-8 in order
```

### Option 2: Python Scripts
```python
from face_recognition_system import FaceRecognitionSystem

# Initialize system
face_system = FaceRecognitionSystem()

# Add face from webcam
# (Interactive - will open camera window)

# Start real-time recognition
# (See notebook for complete examples)
```

## 📁 Project Structure

```
Face Recognition System/
├── face_recognition_system.py    # Main system class
├── face_database.py             # Database management
├── cuda_utils.py               # CUDA utilities & benchmarking
├── download_models.py          # Model download script
├── face_recognition_system.ipynb # Interactive notebook
├── requirements.txt            # Dependencies
├── models/                    # Face detection models
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── face_database/            # Face storage
│   ├── face_encodings.pkl   # Face encodings
│   ├── face_metadata.json  # Face metadata
│   └── images/             # Face images
└── .venv/                  # Virtual environment
```

## 🔧 Configuration

### Detection Models
- **Haar Cascade**: Fast, good for real-time (~200ms)
- **DNN MobileNet**: Neural network, more accurate (~3ms with GPU)
- **MTCNN**: Most accurate, requires `facenet_pytorch` package

### Performance Settings
- **Recognition Tolerance**: 0.3-0.9 (lower = stricter matching)
- **Confidence Threshold**: 0.1-0.9 (minimum detection confidence)
- **Threading**: Enable for better real-time performance

## 📊 Performance

### Tested Configuration
- **GPU**: NVIDIA RTX 3050 (4GB VRAM)
- **CPU**: 12 cores @ 2000 MHz
- **RAM**: 16GB

### Results
- **FPS**: 15-30 fps (real-time)
- **Detection Time**: 3-200ms depending on model
- **Recognition Time**: <50ms per face
- **Database**: Scales to 1000+ faces

## 🚨 Troubleshooting

### Common Issues

**1. Camera not opening:**
- Check camera permissions
- Ensure no other apps are using camera
- Try different camera index (0, 1, 2...)

**2. CUDA errors:**
- System automatically falls back to CPU
- Install CUDA toolkit for full GPU support
- Check GPU compatibility

**3. Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.13+ recommended)

**4. Poor recognition:**
- Add more face samples per person
- Adjust recognition tolerance
- Ensure good lighting during capture

## 🔐 Privacy & Security

- **Local Processing**: All face data stays on your machine
- **No Cloud**: No data sent to external servers
- **Encrypted Storage**: Face encodings stored in binary format
- **Data Control**: Easy export/import and deletion of face data

## 📝 License

This project is for educational and personal use. Face recognition technology should be used responsibly and in compliance with local privacy laws.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements!

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Jupyter notebook comments
3. Check system logs in `face_recognition.log`

---

**Built with ❤️ using OpenCV, PyTorch, and Jupyter** 
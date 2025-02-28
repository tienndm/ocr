# 🔍 Transformer-based OCR System

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A state-of-the-art Optical Character Recognition (OCR) system that uses transformer architecture for accurate text recognition from images.

## ✨ Features

- **Transformer Architecture**: Utilizes encoder-decoder transformer model for robust text recognition
- **High Accuracy**: Optimized for realistic OCR scenarios
- **Easy Integration**: Simple API for incorporating OCR capabilities into any application
- **GPU Acceleration**: Supports CUDA for faster inference
- **Customizable**: Can be fine-tuned for specific domains and text styles

## 📋 Requirements

- Python 3.7+
- PyTorch 1.8+
- Pillow
- torchvision
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ocr.git
cd ocr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained model:
```bash
mkdir -p checkpoints
# Download model from your preferred storage location
# Example: wget https://your-domain.com/models/best_model.pth -O checkpoints/best_model.pth
```

## 🚀 Usage

### Basic Usage

```python
from inference.realistic_predictor import RealisticPredictor

# Initialize the predictor
predictor = RealisticPredictor(
    model_path="checkpoints/best_model.pth",
    vocab_path="data/vocab/vocab.json",
    device="cuda"  # Use "cpu" if CUDA is not available
)

# Predict from an image file
result = predictor.predict("path/to/image.jpg")
print("Recognized text:", result)

# Or use with a PIL Image
from PIL import Image
image = Image.open("path/to/image.jpg").convert("RGB")
result = predictor.predict(image)
print("Recognized text:", result)
```

### Command-line Usage

```bash
python -m inference.realistic_predictor --image path/to/image.jpg --model checkpoints/best_model.pth --vocab data/vocab/vocab.json
```

## 📂 Project Structure

```
ocr/
├── data/
│   └── vocab/
│       └── vocab.json
├── inference/
│   └── realistic_predictor.py
├── models/
│   └── ocr.py
├── checkpoints/
│   └── best_model.pth
├── README.md
└── requirements.txt
```

## 🔄 Training

To train your own model:

1. Prepare your training data in the required format
2. Configure training parameters in the config file
3. Run the training script:
```bash
python train.py --config configs/train_config.yaml
```

## 📈 Performance

The model achieves the following performance metrics:

- **Accuracy**: XX% on standard OCR benchmarks
- **Speed**: XX images per second on NVIDIA V100
- **Support**: Handles multiple languages and fonts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Thanks to the authors of the original transformer architecture
- Dataset providers for training and evaluation
- Open source community for valuable tools and libraries

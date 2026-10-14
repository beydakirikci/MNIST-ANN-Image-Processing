# MNIST Digit Classification with ANN & Image Processing

A deep learning project that classifies handwritten digits (0–9) from the **MNIST dataset** using an **Artificial Neural Network (ANN)**. The pipeline includes classical **image processing** techniques (OpenCV) as preprocessing steps before feeding the data into the neural network built with **TensorFlow/Keras**.

## 📸 Image Processing Pipeline

The project applies the following preprocessing stages to each 28×28 grayscale image:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | **Histogram Equalization** | Contrast enhancement |
| 2 | **Gaussian Blur** (5×5) | Noise reduction |
| 3 | **Canny Edge Detection** | Edge extraction |
| 4 | **Flatten + Normalize** | Convert to 784-dim vector, scale to [0, 1] |

![Image Processing Stages](Ekran%20görüntüsü%202026-10-14%20194906.png)

## 🧠 ANN Architecture

```
Input (784) → Dense(128, ReLU) → Dropout(0.5) → Dense(64, ReLU) → Dense(10, Softmax)
```

- **Optimizer:** Adam (learning rate = 0.0001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Training:** 50 epochs, batch size 32
- **Regularization:** 50% Dropout to prevent overfitting

## 🛠️ Tech Stack

- **Python 3.14**
- **TensorFlow / Keras** — Model building & training
- **OpenCV** — Image processing
- **Matplotlib** — Visualization
- **NumPy** — Numerical operations

## 🚀 Getting Started

### Prerequisites

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python ann.py
```

## 📊 Output & Results

The script outputs:
1. **Image Processing Stages** (Original → Histogram Equalization → Gaussian Blur → Canny Edges)
2. **ANN Model Training** for 50 epochs
3. **Evaluation Metrics** (Test Loss & Test Accuracy)
4. **Learning Curves** showing training and validation loss/accuracy progression

### 📈 Training History (Loss & Accuracy)

![Training History](Ekran%20görüntüsü%202026-10-14%20194802.png)

## 📁 Project Structure

```
├── ann.py              # Main script (preprocessing + model + training)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 📝 License

This project is open source and available for educational purposes.
